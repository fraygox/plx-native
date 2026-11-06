//! A one-deep pipeline around [`crate::ff_pcm::PcmDecoder::decode`], so the decode of an
//! app-decoded audio track (the codecs of `devcaps::APP_DECODED`: the FFmpeg
//! `avcodec_send_packet`/`avcodec_receive_frame` work and the `PcmConverter` downmix and resample
//! after it) runs on its own thread instead of serialized with container parsing and the network
//! read that feeds it.
//!
//! **What moves, and what does not.** Only the decode CALL runs on the worker. `ff.rs`'s demux
//! loop keeps the rest of its audio-push logic where it was: `AuBatcher` batching,
//! `audio_push_gate`'s cross-lane headroom check, `drain_wire` backpressure, the pre-audio headroom
//! marks, and a decode failure's fatal break. The worker makes no pushes. It is handed a
//! [`ClonedPacket`] (an `av_packet_clone`: the demuxer's own `pkt` is reused by the next
//! `av_read_frame` as soon as the arm returns, so the worker cannot borrow it) and hands back an
//! owned `Vec<u8>`, which the demux thread puts through the same `AuBatcher` and the same
//! `aq_push_gated` call as before.
//!
//! **One decode in flight, by construction.** The demux thread's only way to hand the worker a
//! packet is [`PcmPipeline::exchange`]: it first collects the decode already in flight (waiting
//! for it if the worker is still running it), and only then submits the new packet. So at most
//! one decode is ever outstanding, results come back in the order the packets were read (which
//! `PcmDecoder`'s `ErrorBudget`, a consecutive-failure counter, relies on), and no call sequence
//! can queue a second one behind the first. Decode N overlaps everything the demux thread does
//! until it next collects: on an interleaved file, mostly video.
//!
//! **Collected early, not only at the next audio packet.** An `exchange` returns the PREVIOUS
//! audio packet's audio, and nothing on the first one after the open or a seek, so a decode
//! collected only there would reach the lane one audio packet late. That lag is not harmless: the
//! video lane's pre-audio headroom (`ff::PreAudioHeadroom`) decides whether a full video lane may
//! park or must fail the open by whether audio has been pushed, and a late track whose first
//! packet is still on the worker when the video lane fills would fail an open that inline decode
//! passed. So the demux thread also calls [`PcmPipeline::poll`] before every video push: it
//! collects a finished decode without waiting, and waits for one still running only when that
//! video push would park. `poll` never submits, so the one-in-flight rule holds.
//!
//! **Seeking and teardown.** [`PcmPipeline::reset`] collects and discards the decode in flight
//! (pre-seek audio), resets the decoder on the worker, and returns once that has happened, so the
//! first `exchange` after a seek starts with nothing outstanding. [`PcmPipeline::drain_final`]
//! collects the last decode at EOF, so the stream's last audio packet is not dropped. Dropping the
//! pipeline closes the request channel and joins the worker through `plx_base::task::join`.
//!
//! **A refused worker is not a failed stream.** The worker starts through
//! `plx_base::task::spawn`; if the OS refuses the thread, the pipeline keeps the decoder and
//! decodes on the demux thread inside `exchange`, as the demuxer did before this module existed
//! (with no lag: `exchange` then returns the packet's own audio).

use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::ff::AVPacket;
use crate::player::threads::SendPtr;

/// The worker's thread name (`plx_base::task` keeps 15 bytes of it; this fits).
const WORKER: &str = "pcm-decode";

/// What the pipeline reports once the worker has died (a panic in the decoder) or never took the
/// decoder.
const GONE: &str = "the pcm-decode worker is gone";

/// What the worker needs from a decoder. Generic so the pipeline's threading is host-testable
/// with a fake; [`PcmDecoderAdapter`] is the only production implementor and the only place this
/// module calls into the decoder.
pub(crate) trait PacketDecoder: Send + 'static {
    /// The packet `decode` consumes. Dropping one that was never decoded releases it.
    type Packet: Send + 'static;

    /// Decode one packet into `(pts, pcm)`.
    fn decode(&mut self, pkt: Self::Packet, pts: i64) -> Result<(i64, Vec<u8>), String>;

    /// The demuxer sought: drop whatever the decoder carries from before it.
    fn reset(&mut self);
}

/// An `av_packet_clone` of a demuxed packet: its own reference to the payload, released
/// (`av_packet_free`) when it drops, on whichever thread that is.
pub(crate) struct ClonedPacket(SendPtr<AVPacket>);

impl ClonedPacket {
    /// Clone `pkt`; `None` when FFmpeg could not allocate the copy.
    ///
    /// # Safety
    /// `pkt` is a valid packet.
    pub(crate) unsafe fn of(pkt: *const AVPacket) -> Option<ClonedPacket> {
        let clone = crate::ff::av_packet_clone(pkt);
        (!clone.is_null()).then(|| ClonedPacket(SendPtr(clone)))
    }
}

impl Drop for ClonedPacket {
    fn drop(&mut self) {
        // Safety: the pointer is this value's own `av_packet_clone`, referenced by nothing else.
        unsafe { crate::ff::av_packet_free(&mut self.0 .0) };
    }
}

/// The production [`PacketDecoder`]: the real decoder, fed cloned packets.
pub(crate) struct PcmDecoderAdapter(crate::ff_pcm::PcmDecoder);

impl PacketDecoder for PcmDecoderAdapter {
    type Packet = ClonedPacket;

    fn decode(&mut self, pkt: ClonedPacket, pts: i64) -> Result<(i64, Vec<u8>), String> {
        // Safety: `pkt` holds its own reference to the packet until it drops at the end of this
        // call, after the decoder has read it.
        unsafe { self.0.decode(pkt.0 .0, pts) }.map(|(stamp, pcm)| (stamp, pcm.to_vec()))
    }

    fn reset(&mut self) {
        self.0.reset_stream();
    }
}

/// One packet's decoded audio, as [`PcmPipeline::exchange`] hands it back.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Decoded {
    /// The pts the packet was handed over with: the demuxed packet this audio belongs to.
    pub packet_pts: i64,
    /// The decoder's own stamp for the audio, which goes to the `AuBatcher`.
    pub pcm_pts: i64,
    pub pcm: Vec<u8>,
}

enum Request<P> {
    Decode(P, i64),
    Reset,
}

enum Response {
    Decoded(Result<(i64, Vec<u8>), String>),
    Reset,
}

/// The demux thread's side of a running worker.
struct Worker<P> {
    /// `None` once the worker is gone, and inside `drop`, which closes it to end the worker's loop.
    req_tx: Option<mpsc::SyncSender<Request<P>>>,
    resp_rx: mpsc::Receiver<Response>,
    handle: Option<JoinHandle<()>>,
    /// The pts of the one decode in flight, if any.
    in_flight: Option<i64>,
}

impl<P> Worker<P> {
    /// Wait for the decode in flight, if there is one.
    fn collect(&mut self) -> Result<Option<Decoded>, String> {
        let Some(packet_pts) = self.in_flight.take() else {
            return Ok(None);
        };
        match self.resp_rx.recv() {
            Ok(resp) => Self::decoded(packet_pts, resp),
            Err(_) => self.gone(),
        }
    }

    /// The decode in flight if the worker has finished it; with `wait`, wait for it if it has not.
    fn poll(&mut self, wait: bool) -> Result<Option<Decoded>, String> {
        let Some(packet_pts) = self.in_flight else {
            return Ok(None);
        };
        if wait {
            return self.collect();
        }
        match self.resp_rx.try_recv() {
            Ok(resp) => {
                self.in_flight = None;
                Self::decoded(packet_pts, resp)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.in_flight = None;
                self.gone()
            }
        }
    }

    fn decoded(packet_pts: i64, resp: Response) -> Result<Option<Decoded>, String> {
        match resp {
            Response::Decoded(r) => r.map(|(pcm_pts, pcm)| Some(Decoded { packet_pts, pcm_pts, pcm })),
            Response::Reset => unreachable!("a Decode request answers Decoded, never Reset"),
        }
    }

    /// The worker is gone (it panicked). Close the request side too, so every later call fails
    /// here at once: otherwise a send could still land in the channel of a worker that is
    /// unwinding (`run`'s `resp_tx` drops before its `req_rx`) and read as a submitted decode.
    fn gone<T>(&mut self) -> Result<T, String> {
        self.req_tx = None;
        Err(GONE.to_owned())
    }

    /// Hand the worker a packet. Only called with nothing in flight (`exchange` collects first),
    /// so the one-slot channel is empty and the send never waits.
    fn submit(&mut self, pkt: P, pts: i64) -> Result<(), String> {
        debug_assert!(self.in_flight.is_none(), "submit with a decode still in flight");
        let sent = self.req_tx.as_ref().is_some_and(|tx| tx.send(Request::Decode(pkt, pts)).is_ok());
        if !sent {
            return self.gone();
        }
        self.in_flight = Some(pts);
        Ok(())
    }

    fn reset(&mut self) {
        // Pre-seek audio, or a dead worker's error: either way nothing the caller wants.
        let _ = self.collect();
        if !self.req_tx.as_ref().is_some_and(|tx| tx.send(Request::Reset).is_ok()) {
            self.req_tx = None;
            return; // the worker is gone; the next `exchange` reports it
        }
        match self.resp_rx.recv() {
            Ok(Response::Reset) => {}
            Err(_) => self.req_tx = None,
            Ok(Response::Decoded(_)) => unreachable!("a Reset request answers Reset, never Decoded"),
        }
    }
}

impl<P> Drop for Worker<P> {
    fn drop(&mut self) {
        // Closing the request channel ends the worker's `recv` loop (after any decode it is in);
        // `task::join` logs a worker that panicked and a join that stalls.
        self.req_tx = None;
        if let Some(handle) = self.handle.take() {
            plx_base::task::join(WORKER, handle);
        }
    }
}

enum Mode<D: PacketDecoder> {
    Worker(Worker<D::Packet>),
    /// The worker could not start: decode on the caller's thread.
    Inline(D),
}

/// The demux thread's handle onto the decoder. One per demux open, like the `PcmDecoder` it owns.
pub(crate) struct PcmPipeline<D: PacketDecoder = PcmDecoderAdapter> {
    mode: Mode<D>,
}

impl PcmPipeline {
    /// Start the worker and move `decoder` onto it, or keep the decoder here if the OS refuses
    /// the thread.
    pub(crate) fn start(decoder: crate::ff_pcm::PcmDecoder) -> PcmPipeline {
        PcmPipeline::start_with(PcmDecoderAdapter(decoder), |what, work| plx_base::task::spawn(what, work))
    }
}

impl<D: PacketDecoder> PcmPipeline<D> {
    /// [`start`](PcmPipeline::start) with the spawn seam as a parameter, so a test can refuse it.
    fn start_with(
        decoder: D,
        spawn: impl FnOnce(&'static str, Box<dyn FnOnce() + Send>) -> Option<JoinHandle<()>>,
    ) -> Self {
        // The decoder follows the thread rather than riding in its closure, so a refused spawn
        // (which drops the closure unrun) leaves it here for the inline fallback.
        let (dec_tx, dec_rx) = mpsc::sync_channel::<D>(1);
        let (req_tx, req_rx) = mpsc::sync_channel::<Request<D::Packet>>(1);
        let (resp_tx, resp_rx) = mpsc::channel::<Response>();
        let handle = spawn(
            WORKER,
            Box::new(move || {
                if let Ok(decoder) = dec_rx.recv() {
                    run(decoder, req_rx, resp_tx);
                }
            }),
        );
        let inline = |decoder: D| {
            crate::player::log("ff: pcm-decode worker refused; decoding PCM on the demux thread");
            PcmPipeline { mode: Mode::Inline(decoder) }
        };
        let Some(handle) = handle else {
            return inline(decoder);
        };
        // The worker waits for the decoder; the one-slot channel takes it without blocking.
        match dec_tx.send(decoder) {
            Ok(()) => PcmPipeline {
                mode: Mode::Worker(Worker { req_tx: Some(req_tx), resp_rx, handle: Some(handle), in_flight: None }),
            },
            Err(mpsc::SendError(decoder)) => {
                plx_base::task::join(WORKER, handle);
                inline(decoder)
            }
        }
    }

    /// Whether decode runs on the worker (`false`: the worker was refused, decode is inline).
    pub(crate) fn off_thread(&self) -> bool {
        matches!(self.mode, Mode::Worker(_))
    }

    /// Collect the decode in flight without submitting anything: `Ok(None)` when nothing is in
    /// flight or (without `wait`) the worker is still on it; with `wait`, block until it is done.
    /// Always `Ok(None)` inline, where `exchange` returns each packet's audio itself. `Err` as for
    /// [`exchange`](Self::exchange).
    pub(crate) fn poll(&mut self, wait: bool) -> Result<Option<Decoded>, String> {
        match &mut self.mode {
            Mode::Worker(w) => w.poll(wait),
            Mode::Inline(_) => Ok(None),
        }
    }

    /// Hand over the packet at `pts` and get back the previous packet's audio: the decode in
    /// flight is collected first (waiting for the worker if it is still decoding), then `pkt` is
    /// submitted. `Ok(None)` when nothing was in flight. `Err` is a decode failure of the
    /// collected packet (`pkt` is then not submitted) or a dead worker; the caller treats both as
    /// it treated `PcmDecoder::decode`'s `Err`: fatal to the demux.
    pub(crate) fn exchange(&mut self, pkt: D::Packet, pts: i64) -> Result<Option<Decoded>, String> {
        match &mut self.mode {
            Mode::Worker(w) => {
                let previous = w.collect()?;
                w.submit(pkt, pts)?;
                Ok(previous)
            }
            Mode::Inline(d) => d
                .decode(pkt, pts)
                .map(|(pcm_pts, pcm)| Some(Decoded { packet_pts: pts, pcm_pts, pcm })),
        }
    }

    /// At EOF: the last submitted packet's audio, waiting for the worker if it is still decoding
    /// it. `Ok(None)` when nothing is in flight.
    pub(crate) fn drain_final(&mut self) -> Result<Option<Decoded>, String> {
        match &mut self.mode {
            Mode::Worker(w) => w.collect(),
            Mode::Inline(_) => Ok(None),
        }
    }

    /// A seek: discard the decode in flight (pre-seek audio) and reset the decoder, returning once
    /// the reset has happened.
    pub(crate) fn reset(&mut self) {
        match &mut self.mode {
            Mode::Worker(w) => w.reset(),
            Mode::Inline(d) => d.reset(),
        }
    }
}

/// The worker's loop: answer each request in order until the request channel closes.
fn run<D: PacketDecoder>(mut decoder: D, req_rx: mpsc::Receiver<Request<D::Packet>>, resp_tx: mpsc::Sender<Response>) {
    while let Ok(req) = req_rx.recv() {
        let resp = match req {
            Request::Decode(pkt, pts) => Response::Decoded(decoder.decode(pkt, pts)),
            Request::Reset => {
                decoder.reset();
                Response::Reset
            }
        };
        if resp_tx.send(resp).is_err() {
            return; // the pipeline is gone: nobody left to answer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// A decoder with no FFmpeg behind it: the packet at `pts` decodes to `(pts, [pts as u8])`,
    /// unless the test says that pts fails or panics.
    #[derive(Default)]
    struct FakeDecoder {
        resets: Arc<AtomicUsize>,
        /// The pts of every `decode` call, in call order.
        calls: Arc<Mutex<Vec<i64>>>,
        /// When set, each `decode` waits for one message first, so the test decides when the
        /// worker finishes.
        hold: Option<mpsc::Receiver<()>>,
        fail_at: Option<i64>,
        panic_at: Option<i64>,
    }

    impl PacketDecoder for FakeDecoder {
        type Packet = ();

        fn decode(&mut self, (): (), pts: i64) -> Result<(i64, Vec<u8>), String> {
            if let Some(hold) = &self.hold {
                let _ = hold.recv();
            }
            self.calls.lock().unwrap().push(pts);
            if self.panic_at == Some(pts) {
                panic!("the fake decoder panics at {pts}, as the test asked");
            }
            if self.fail_at == Some(pts) {
                return Err(format!("bad packet {pts}"));
            }
            Ok((pts, vec![pts as u8]))
        }

        fn reset(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn decoded(pts: i64) -> Decoded {
        Decoded { packet_pts: pts, pcm_pts: pts, pcm: vec![pts as u8] }
    }

    fn start(fake: FakeDecoder) -> PcmPipeline<FakeDecoder> {
        let p = PcmPipeline::start_with(fake, |what, work| plx_base::task::spawn(what, work));
        assert!(p.off_thread(), "the worker must have started for this test to mean anything");
        p
    }

    /// Regression: the demux arm used to `try_collect`, get `None` while the worker was still
    /// decoding, and submit the next packet anyway: two decodes outstanding, and a seek's `reset`
    /// then met a stale `Decoded` (`unreachable!`). `exchange` is the only way in now, and a
    /// packet that arrives while the previous one is still decoding waits for it instead of
    /// queueing behind it.
    #[test]
    fn an_exchange_while_the_worker_is_still_decoding_waits_for_it_and_a_reset_after_is_clean() {
        let (release, hold) = mpsc::channel();
        let fake = FakeDecoder { hold: Some(hold), ..FakeDecoder::default() };
        let resets = Arc::clone(&fake.resets);
        let mut p = start(fake);
        assert_eq!(p.exchange((), 1), Ok(None));
        // Taken before the releaser exists, so its 50 ms sleep is wholly inside the measured span
        // however late the OS schedules this thread after the spawn.
        let t0 = Instant::now();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            release.send(()).unwrap(); // decode 1 finishes
            release.send(()).unwrap(); // decode 2 (the reset discards it)
            release.send(()).unwrap(); // decode 3
        });
        assert_eq!(p.exchange((), 2), Ok(Some(decoded(1))));
        assert!(t0.elapsed() >= Duration::from_millis(40), "the second exchange must wait for decode 1");
        p.reset();
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        assert_eq!(p.exchange((), 3), Ok(None));
        assert_eq!(p.drain_final(), Ok(Some(decoded(3))));
        releaser.join().unwrap();
    }

    #[test]
    fn each_exchange_returns_the_previous_packets_audio_and_drain_final_the_last() {
        let fake = FakeDecoder::default();
        let calls = Arc::clone(&fake.calls);
        let mut p = start(fake);
        assert_eq!(p.drain_final(), Ok(None), "nothing in flight yet");
        assert_eq!(p.exchange((), 10), Ok(None));
        assert_eq!(p.exchange((), 20), Ok(Some(decoded(10))));
        assert_eq!(p.exchange((), 30), Ok(Some(decoded(20))));
        assert_eq!(p.drain_final(), Ok(Some(decoded(30))));
        assert_eq!(p.drain_final(), Ok(None), "the last decode is handed back once");
        assert_eq!(*calls.lock().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn a_reset_after_several_exchanges_discards_only_the_decode_in_flight() {
        let fake = FakeDecoder::default();
        let resets = Arc::clone(&fake.resets);
        let calls = Arc::clone(&fake.calls);
        let mut p = start(fake);
        assert_eq!(p.exchange((), 1), Ok(None));
        for pts in 2..=5 {
            assert_eq!(p.exchange((), pts), Ok(Some(decoded(pts - 1))));
        }
        p.reset();
        p.reset();
        assert_eq!(resets.load(Ordering::SeqCst), 2);
        assert_eq!(p.exchange((), 100), Ok(None), "packet 5 was pre-seek audio");
        assert_eq!(p.exchange((), 101), Ok(Some(decoded(100))));
        assert_eq!(p.drain_final(), Ok(Some(decoded(101))));
        assert_eq!(*calls.lock().unwrap(), vec![1, 2, 3, 4, 5, 100, 101]);
    }

    #[test]
    fn a_decode_error_comes_back_on_the_next_exchange_which_then_submits_nothing() {
        let fake = FakeDecoder { fail_at: Some(1), ..FakeDecoder::default() };
        let calls = Arc::clone(&fake.calls);
        let mut p = start(fake);
        assert_eq!(p.exchange((), 1), Ok(None));
        assert_eq!(p.exchange((), 2), Err("bad packet 1".to_owned()));
        assert_eq!(p.drain_final(), Ok(None), "packet 2 was never submitted");
        assert_eq!(*calls.lock().unwrap(), vec![1]);
    }

    #[test]
    fn the_last_packets_decode_error_comes_back_from_drain_final() {
        let mut p = start(FakeDecoder { fail_at: Some(2), ..FakeDecoder::default() });
        assert_eq!(p.exchange((), 1), Ok(None));
        assert_eq!(p.exchange((), 2), Ok(Some(decoded(1))));
        assert_eq!(p.drain_final(), Err("bad packet 2".to_owned()));
    }

    /// A worker that died (here the decoder panicked) is a failure on every later call, never a
    /// hang; a reset on it returns, and dropping the pipeline still joins. Deterministic: the
    /// first `Err` closes the request side, so a later submit cannot race the worker's unwind
    /// into a send that succeeds.
    #[test]
    fn a_dead_worker_fails_every_exchange_instead_of_hanging() {
        let mut p = start(FakeDecoder { panic_at: Some(1), ..FakeDecoder::default() });
        assert_eq!(p.exchange((), 1), Ok(None));
        assert_eq!(p.exchange((), 2), Err(GONE.to_owned()));
        assert_eq!(p.exchange((), 3), Err(GONE.to_owned()));
        p.reset();
        assert_eq!(p.drain_final(), Ok(None));
        drop(p);
    }

    /// `poll` collects a finished decode without submitting anything, waits only when asked to,
    /// and leaves the next `exchange` with nothing to collect.
    #[test]
    fn poll_collects_the_decode_in_flight_and_waits_only_when_asked() {
        let (release, hold) = mpsc::channel();
        let fake = FakeDecoder { hold: Some(hold), ..FakeDecoder::default() };
        let calls = Arc::clone(&fake.calls);
        let mut p = start(fake);
        assert_eq!(p.poll(false), Ok(None), "nothing in flight");
        assert_eq!(p.poll(true), Ok(None), "nothing in flight, nothing to wait for");
        assert_eq!(p.exchange((), 1), Ok(None));
        assert_eq!(p.poll(false), Ok(None), "decode 1 is held: not finished, not waited for");
        release.send(()).unwrap();
        assert_eq!(p.poll(true), Ok(Some(decoded(1))), "waited for decode 1");
        assert_eq!(p.poll(false), Ok(None), "handed back once");
        assert_eq!(p.exchange((), 2), Ok(None), "decode 1 was already collected");
        release.send(()).unwrap();
        // Without waiting: finished soon after the release, so a bounded spin must see it.
        let until = Instant::now() + Duration::from_secs(10);
        let got = loop {
            match p.poll(false) {
                Ok(None) if Instant::now() < until => std::thread::yield_now(),
                other => break other,
            }
        };
        assert_eq!(got, Ok(Some(decoded(2))));
        assert_eq!(p.drain_final(), Ok(None));
        assert_eq!(*calls.lock().unwrap(), vec![1, 2], "poll never submits");
    }

    #[test]
    fn poll_on_a_dead_worker_fails_and_so_does_every_later_call() {
        let mut p = start(FakeDecoder { panic_at: Some(1), ..FakeDecoder::default() });
        assert_eq!(p.exchange((), 1), Ok(None));
        assert_eq!(p.poll(true), Err(GONE.to_owned()));
        assert_eq!(p.exchange((), 2), Err(GONE.to_owned()));
        assert_eq!(p.poll(false), Ok(None), "nothing was submitted");
    }

    /// The OS refused the thread: the decoder stays on the caller's thread and each packet is
    /// decoded inside its own `exchange`, with no lag.
    #[test]
    fn a_refused_worker_decodes_inline_on_the_callers_thread() {
        let fake = FakeDecoder::default();
        let resets = Arc::clone(&fake.resets);
        let mut p = PcmPipeline::start_with(fake, |_, _| None);
        assert!(!p.off_thread());
        assert_eq!(p.poll(true), Ok(None), "inline, nothing is ever in flight");
        assert_eq!(p.exchange((), 7), Ok(Some(decoded(7))));
        assert_eq!(p.exchange((), 8), Ok(Some(decoded(8))));
        assert_eq!(p.drain_final(), Ok(None));
        p.reset();
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_the_pipeline_with_a_decode_in_flight_joins_the_worker() {
        let (release, hold) = mpsc::channel();
        let mut p = start(FakeDecoder { hold: Some(hold), ..FakeDecoder::default() });
        assert_eq!(p.exchange((), 1), Ok(None));
        release.send(()).unwrap();
        drop(p); // must return: a hang here is the leak this test exists to catch
    }
}
