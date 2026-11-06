//! A one-deep pipeline around [`crate::ff_pcm::PcmDecoder::decode`], so the demux thread's
//! `avcodec_send_packet`/`avcodec_receive_frame` work for a software-decoded audio track (DTS,
//! TrueHD, FLAC, ALAC, LPCM — `devcaps::APP_DECODED`) runs on its own thread instead of
//! serialized with container parsing and the network read that feeds it.
//!
//! **What moves, and what does not.** Only the decode CALL — the CPU-bound part — runs on the
//! worker. `ff.rs`'s demux loop keeps every byte of its existing audio-push logic exactly where
//! it is: `AuBatcher` batching, `audio_push_gate`'s cross-lane headroom check, `drain_wire`
//! backpressure, and a decode failure's exact fatal-break semantics. The worker is handed an
//! owned, independently ref-counted `AVPacket` (`av_packet_clone` — the demuxer's own `pkt` is
//! reused by the next `av_read_frame` the instant this call returns, so the worker cannot borrow
//! it) and hands back an owned `Vec<u8>`; the demux thread copies that into the SAME `AuBatcher`
//! and pushes it through the SAME `aq_push_gated` call it always has.
//!
//! **Why depth exactly one.** `submit` is a `sync_channel(1)`: a second submit before the first
//! result is collected blocks the caller. That is deliberate rather than a missing buffer — a
//! deeper queue would let the demux thread race arbitrarily far ahead of decode, and on a
//! `pcm_lossy` stream `PcmDecoder`'s own `ErrorBudget` (consecutive-failure counter) assumes
//! calls are answered in the order they were made; one in flight at a time keeps that invariant
//! true with no reordering possible. The caller is expected to `try_collect` a PREVIOUS packet's
//! result before (or instead of) submitting a new one — see `ff.rs`'s audio arm, which collects
//! on every audio packet and only submits when nothing was collected.
//!
//! **Seeking and teardown never race the worker.** [`PcmPipeline::reset`] first drains any
//! outstanding submission with a BLOCKING collect (discarding its result — it is pre-seek audio),
//! then resets the decoder on the worker thread, and only returns once that has happened. The
//! demux thread cannot observe a stale in-flight result after a seek, and the worker never touches
//! the decoder concurrently with itself. [`PcmPipeline::drain_final`] does the same blocking
//! collect at EOF/teardown so the last submitted packet's audio is not silently dropped.
//!
//! **What this deliberately does NOT try to fix:** a genuinely stalled connection while the
//! audio lane is parked waiting for queue space no longer gets `ff.rs`'s `drain_wire`
//! keep-the-socket-warm polling for pushes the worker makes on the demux thread's behalf — those
//! pushes pass a plain parking gate. That is `aq_push_gated`'s ordinary behaviour; it only matters
//! on a lane that is already both full and parked, which an audio lane sized against its own
//! bitrate reaches rarely. See `docs/` for the full writeup — `aq_push_gated` is used for the
//! batched AU exactly where the original inline call was.

use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::ff::AVPacket;

/// `*mut AVPacket` is not `Send`; this crate already carries the identical justification for the
/// same pattern (`player::threads::SendPtr`) — the pointer is an owned, exclusively-held `AVPacket`
/// for the one channel send that moves it to the worker, which is the only thread that ever reads
/// it again.
struct OwnedPacket(*mut AVPacket);
unsafe impl Send for OwnedPacket {}

/// What the worker thread needs from a decoder. Generic so the pipeline's threading and
/// backpressure are host-testable with a fake — [`PcmDecoderAdapter`] is the only production
/// implementor, and it is the only place this module touches real FFmpeg.
trait PacketDecoder: Send {
    /// Decode one packet. Implementations that take ownership of `pkt` (the production adapter
    /// does, via `av_packet_free`) must free it before returning; a test fake that never actually
    /// dereferences the pointer has nothing to free.
    fn decode(&mut self, pkt: *mut AVPacket, pts: i64) -> Result<(i64, Vec<u8>), String>;

    /// The demuxer sought: drop whatever the decoder carries from before it.
    fn reset(&mut self);
}

/// The only production [`PacketDecoder`]: owns the real decoder and the one packet-freeing step
/// a cloned `AVPacket` needs once decode has read it.
struct PcmDecoderAdapter(crate::ff_pcm::PcmDecoder);

impl PacketDecoder for PcmDecoderAdapter {
    fn decode(&mut self, mut pkt: *mut AVPacket, pts: i64) -> Result<(i64, Vec<u8>), String> {
        // Safety: `pkt` is this call's own clone (`av_packet_clone` at the submit site), held
        // live by nothing else — freeing it here, win or lose, is exactly the ownership
        // `av_packet_clone` handed over.
        let result = unsafe { self.0.decode(pkt as *const AVPacket, pts) }
            .map(|(stamp, pcm)| (stamp, pcm.to_vec()));
        unsafe { crate::ff::av_packet_free(&mut pkt) };
        result
    }

    fn reset(&mut self) {
        self.0.reset_stream();
    }
}

enum Request {
    Decode(OwnedPacket, i64),
    Reset,
}

enum Response {
    Decoded(Result<(i64, Vec<u8>), String>),
    Reset,
}

/// A decode in flight: [`PcmPipeline::submit`] leaves this for [`PcmPipeline::try_collect`] or a
/// blocking drain to answer. [`PcmPipeline::reset`] has no equivalent of its own — it sends its
/// `Request::Reset` and blocks on the one response inline, so nothing is ever left outstanding
/// across a call boundary for a reset the way a decode's answer is.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Outstanding {
    None,
    Decode,
}

/// The demux-thread handle onto the worker. One per engine run — a fresh `PcmPipeline` on a
/// fresh open or a reload, exactly like `PcmDecoder` was constructed fresh before this existed.
pub struct PcmPipeline {
    req_tx: mpsc::SyncSender<Request>,
    resp_rx: mpsc::Receiver<Response>,
    handle: Option<JoinHandle<()>>,
    outstanding: Outstanding,
}

impl PcmPipeline {
    /// Spawn the worker, moving `decoder` onto it. From this call on, the decoder's thread is
    /// fixed for the pipeline's lifetime — `PcmDecoder`'s internal state (the error budget, the
    /// converter's filter tail) is exactly as single-threaded as it always was, just on a
    /// different single thread.
    pub fn spawn(decoder: crate::ff_pcm::PcmDecoder) -> PcmPipeline {
        Self::spawn_with(PcmDecoderAdapter(decoder))
    }

    fn spawn_with(decoder: impl PacketDecoder + 'static) -> PcmPipeline {
        let (req_tx, req_rx) = mpsc::sync_channel::<Request>(1);
        let (resp_tx, resp_rx) = mpsc::channel::<Response>();
        let handle = std::thread::Builder::new()
            .name("pcm-decode".into())
            .spawn(move || Self::run(decoder, req_rx, resp_tx))
            .ok();
        PcmPipeline {
            req_tx,
            resp_rx,
            handle,
            outstanding: Outstanding::None,
        }
    }

    fn run(mut decoder: impl PacketDecoder, req_rx: mpsc::Receiver<Request>, resp_tx: mpsc::Sender<Response>) {
        while let Ok(req) = req_rx.recv() {
            let resp = match req {
                Request::Decode(OwnedPacket(pkt), pts) => Response::Decoded(decoder.decode(pkt, pts)),
                Request::Reset => {
                    decoder.reset();
                    Response::Reset
                }
            };
            if resp_tx.send(resp).is_err() {
                // The pipeline was dropped mid-call (teardown raced the worker, not the thread
                // spawn): nothing left to answer, and nothing left to decode for.
                return;
            }
        }
    }

    /// Submit a packet for decode and return immediately — never calls into FFmpeg on this
    /// thread. `false` means the worker is gone (it never panics in normal operation; a crashed
    /// worker is a bug, not a condition this type recovers from) — the caller treats it exactly
    /// like `PcmDecoder::decode` returning `Err` used to: a fatal demux failure.
    ///
    /// Caller contract: at most one submission outstanding. Collect (or drain) the previous one
    /// first; `sync_channel(1)` makes a violation block rather than silently reorder, which would
    /// turn into a deadlock the first time it happened in testing rather than a quiet corruption
    /// later — the louder failure is the one worth having here.
    pub fn submit(&mut self, pkt: *mut AVPacket, pts: i64) -> bool {
        debug_assert_eq!(self.outstanding, Outstanding::None, "a submit raced an uncollected one");
        let ok = self.req_tx.send(Request::Decode(OwnedPacket(pkt), pts)).is_ok();
        if ok {
            self.outstanding = Outstanding::Decode;
        }
        ok
    }

    /// Collect a previously submitted decode if the worker has finished it; `None` means either
    /// nothing is outstanding or the worker has not finished yet — the two look the same to the
    /// caller, which already treats "no AU yet" as business as usual on every other packet.
    pub fn try_collect(&mut self) -> Option<Result<(i64, Vec<u8>), String>> {
        if self.outstanding != Outstanding::Decode {
            return None;
        }
        match self.resp_rx.try_recv() {
            Ok(Response::Decoded(r)) => {
                self.outstanding = Outstanding::None;
                Some(r)
            }
            Ok(Response::Reset) => unreachable!("a Decode request answers Decoded, never Reset"),
            Err(_) => None,
        }
    }

    /// Block for a submission already in flight, discarding its result. Used only before a
    /// `reset` (the result is pre-seek audio, which the caller is about to throw away regardless)
    /// and at teardown through [`drain_final`](Self::drain_final), which wants the AU instead.
    fn collect_blocking(&mut self) -> Option<Result<(i64, Vec<u8>), String>> {
        if self.outstanding != Outstanding::Decode {
            return None;
        }
        self.outstanding = Outstanding::None;
        match self.resp_rx.recv() {
            Ok(Response::Decoded(r)) => Some(r),
            Ok(Response::Reset) => unreachable!("a Decode request answers Decoded, never Reset"),
            Err(_) => None,
        }
    }

    /// The last submitted packet's audio, blocking until the worker has it — called once, at
    /// EOF/teardown, before the caller's final `AuBatcher::take()` flush. Skipping this would
    /// silently drop up to one packet's worth of audio from the very end of every stream this
    /// pipeline ever decoded.
    pub fn drain_final(&mut self) -> Option<Result<(i64, Vec<u8>), String>> {
        self.collect_blocking()
    }

    /// A seek: drop whatever the worker carries from the old position (in flight or already
    /// decoded-but-uncollected — either way it is pre-seek audio) and reset the decoder, blocking
    /// until the reset has actually happened on the worker thread. Returns only once it is safe
    /// for the demux thread to resume feeding packets at the new position — no result can arrive
    /// out of order, because nothing is ever outstanding across this call.
    pub fn reset(&mut self) {
        let _ = self.collect_blocking();
        if self.req_tx.send(Request::Reset).is_err() {
            return;
        }
        match self.resp_rx.recv() {
            Ok(Response::Reset) => {}
            Ok(Response::Decoded(_)) => unreachable!("a Reset request answers Reset, never Decoded"),
            Err(_) => {}
        }
    }
}

impl Drop for PcmPipeline {
    fn drop(&mut self) {
        // Dropping `req_tx` ends the worker's `recv()` loop; join so a torn-down engine never
        // leaves a `pcm-decode` thread outliving it (the same shape `engine::teardown` already
        // keeps for every other worker — see `player/CLAUDE.md`'s threading model).
        drop(std::mem::replace(&mut self.req_tx, mpsc::sync_channel(1).0));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A fake decoder for the pipeline's own tests: never dereferences `pkt` (tests pass
    /// `std::ptr::null_mut()`), so these run with no FFmpeg and no real `AVPacket` anywhere.
    struct FakeDecoder {
        /// `decode`'s answer, consumed in order; a call past the end panics the test loudly.
        script: std::collections::VecDeque<Result<(i64, Vec<u8>), String>>,
        resets: Arc<AtomicUsize>,
        /// Lets a test block the worker inside `decode` until the test releases it, so a
        /// `try_collect` while decode is still "running" is observable as `None` rather than a
        /// race that usually finishes first.
        hold: Option<std::sync::mpsc::Receiver<()>>,
    }

    impl PacketDecoder for FakeDecoder {
        fn decode(&mut self, _pkt: *mut AVPacket, _pts: i64) -> Result<(i64, Vec<u8>), String> {
            if let Some(hold) = &self.hold {
                let _ = hold.recv();
            }
            self.script
                .pop_front()
                .expect("the test's script has no answer for this call")
        }

        fn reset(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn spawn_fake(
        script: Vec<Result<(i64, Vec<u8>), String>>,
    ) -> (PcmPipeline, Arc<AtomicUsize>) {
        let resets = Arc::new(AtomicUsize::new(0));
        let pipeline = PcmPipeline::spawn_with(FakeDecoder {
            script: script.into(),
            resets: Arc::clone(&resets),
            hold: None,
        });
        (pipeline, resets)
    }

    #[test]
    fn a_submitted_packet_is_collectible_once_the_worker_answers() {
        let (mut p, _resets) = spawn_fake(vec![Ok((42, vec![1, 2, 3]))]);
        p.submit(std::ptr::null_mut(), 42);
        let got = loop {
            if let Some(r) = p.try_collect() {
                break r;
            }
        };
        assert_eq!(got, Ok((42, vec![1, 2, 3])));
    }

    #[test]
    fn try_collect_answers_none_before_the_worker_has_finished_and_again_once_nothing_is_outstanding() {
        let (tx, rx) = std::sync::mpsc::channel();
        let resets = Arc::new(AtomicUsize::new(0));
        let mut p = PcmPipeline::spawn_with(FakeDecoder {
            script: vec![Ok((1, vec![9]))].into(),
            resets: Arc::clone(&resets),
            hold: Some(rx),
        });
        p.submit(std::ptr::null_mut(), 1);
        // The worker is parked in `decode`, waiting on `hold` — nothing to collect yet, and that
        // must read as "nothing yet", not as an error or a panic.
        assert_eq!(p.try_collect(), None);
        assert_eq!(p.try_collect(), None, "collecting nothing twice is still nothing");
        tx.send(()).unwrap();
        let got = loop {
            if let Some(r) = p.try_collect() {
                break r;
            }
        };
        assert_eq!(got, Ok((1, vec![9])));
        // Nothing is outstanding anymore — the SAME non-blocking answer, for the other reason.
        assert_eq!(p.try_collect(), None);
    }

    #[test]
    fn a_decode_error_is_handed_back_exactly_once() {
        let (mut p, _resets) = spawn_fake(vec![Err("avcodec_send_packet failed (-22)".into())]);
        p.submit(std::ptr::null_mut(), 7);
        let got = loop {
            if let Some(r) = p.try_collect() {
                break r;
            }
        };
        assert_eq!(got, Err("avcodec_send_packet failed (-22)".into()));
    }

    #[test]
    fn reset_drains_an_outstanding_submission_before_resetting_the_decoder() {
        let (mut p, resets) = spawn_fake(vec![Ok((1, vec![1])), Ok((99, vec![9]))]);
        p.submit(std::ptr::null_mut(), 1);
        // Pre-seek audio in flight, never collected — reset must wait for and discard it, not
        // leave it to surface after the decoder has already been reset for the new position.
        p.reset();
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        assert_eq!(p.try_collect(), None, "the drained result must not still be collectible");
        // The worker is healthy afterwards: a fresh submit still answers.
        p.submit(std::ptr::null_mut(), 99);
        let got = loop {
            if let Some(r) = p.try_collect() {
                break r;
            }
        };
        assert_eq!(got, Ok((99, vec![9])));
    }

    #[test]
    fn reset_with_nothing_outstanding_still_resets_the_decoder() {
        let (mut p, resets) = spawn_fake(vec![]);
        p.reset();
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn drain_final_returns_the_last_submission_instead_of_dropping_it() {
        let (mut p, _resets) = spawn_fake(vec![Ok((5, vec![7, 7, 7]))]);
        p.submit(std::ptr::null_mut(), 5);
        assert_eq!(p.drain_final(), Some(Ok((5, vec![7, 7, 7]))));
        // Already collected: a second drain at a later EOF-ish call site must not block forever
        // or hand back a stale answer.
        assert_eq!(p.drain_final(), None);
    }

    #[test]
    fn drain_final_with_nothing_outstanding_answers_none_at_once() {
        let (mut p, _resets) = spawn_fake(vec![]);
        assert_eq!(p.drain_final(), None);
    }

    #[test]
    fn dropping_the_pipeline_joins_the_worker_rather_than_leaking_it() {
        let (p, _resets) = spawn_fake(vec![]);
        let handle_is_some = p.handle.is_some();
        assert!(handle_is_some, "the worker must have spawned for this test to mean anything");
        drop(p); // must return — a hang here is the bug this test exists to catch
    }
}
