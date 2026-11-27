//! Wait-Before-Prep Flight-1 GPU overlap (`VICTORY_FLIGHT1=1`) and opt-in
//! async submission mode (`VICTORY_ASYNC_SUBMIT=1`).
//!
//! At most one GPU submission (`InFlight`) is in flight at a time. While the
//! host GPU executes Flight N, the title's CPU threads immediately return from
//! submission and record Flight N+1 into fresh `HostList`s (`list::restart`
//! sees `Arc::strong_count > 1` and splits off spare lists). Before Flight N+1
//! touches any shared CPU/GPU state (`evict_unused`, `sync_heaps`,
//! `bindless::uploads` / `texture::recycle`, `Shadow::update`, `Mark::arm`),
//! `finish_flight_locked` waits for Flight N's fence, checks its `Mark`,
//! releases worker staging (`Uploads::release_worker`), copies back GPU writes,
//! lands texture readbacks, runs `after` closures and bottom-of-pipe writes,
//! returns the upload list, recycles its `HostList`s, and delivers any attached
//! title fence signals.
//!
//! A background `victory-flight` thread waits on `flight_event` without
//! holding `HELD` or `Host::submitted`, then acquires `Host::submitted` to
//! land the completed flight so fence waiters and BoP pollers wake promptly
//! even when the title submits no further work.
//!
//! With `VICTORY_FLIGHT1_PREWARM=1`, Flight N+1's wait for Flight N is not
//! idle: the buffer copies its lists ask for that Flight N cannot write are
//! brought up to date first (`Host::prewarm`), which is most of what its
//! preparation copies, so preparing overlaps the GPU. The `overlap:` frame
//! line says how much of the host GPU's time the CPU waited for.
//!
//! With `VICTORY_FLIGHTS=2`, Flight N+1 may be prepared and given to the
//! host GPU while Flight N is still in the air, and flights land strictly
//! in order. Per flight: its lists, upload list and staging, refresh batch
//! and heaps (as with one), its mark (a second one for the flight beside),
//! the Signals attached behind it, what it may write (declared ranges, the
//! copies its heaps view for unordered access), and the number retirement
//! counts it by (its own, not the last one given). What stays one for all
//! and so keeps Flight N+1 waiting for Flight N's landing: the write-mark
//! bitmap (taken whole, by plain reads and writes, at a landing, while the
//! GPU may set bits: a flight beside another marks no writes), the copies'
//! pending ranges and snapshots (it writes no copy Flight N writes, and
//! refreshes of those wait until it lands), the upload list's staging,
//! barrier lists and split marks, title-memory programs (which write any
//! copy). Read marks are taken at a landing too; one a flight sets as they
//! are taken is set again by its next read.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::{Batch, Holding, Host, HostList, Mark, Step};
use crate::object::This;

/// Whether Wait-Before-Prep Flight-1 overlap is enabled (`VICTORY_FLIGHT1=1`,
/// or implicitly enabled when `VICTORY_ASYNC_SUBMIT=1` unless
/// `VICTORY_FLIGHT1=0` is explicitly set; disabled when `VICTORY_GPU_TIMING`
/// is on).
pub(crate) fn flight1_enabled() -> bool {
    #[cfg(test)]
    match TEST_FLIGHT1.load(Ordering::Acquire) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if super::timing() {
            return false;
        }
        let on = match std::env::var("VICTORY_FLIGHT1").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => async_submit_env(),
        };
        if on {
            tracing::info!(
                "Flight-1 GPU overlap enabled (VICTORY_FLIGHT1=1): wait-before-prep with background flight finisher"
            );
        }
        on
    })
}

/// Whether submissions are queued to the dedicated `victory-submit` thread
/// (`VICTORY_ASYNC_SUBMIT=1`, off by default).
pub(crate) fn async_submit_enabled() -> bool {
    #[cfg(test)]
    match TEST_ASYNC_SUBMIT.load(Ordering::Acquire) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = async_submit_env();
        if on {
            tracing::info!(
                "Async submission worker enabled (VICTORY_ASYNC_SUBMIT=1): submissions prepared and launched on victory-submit"
            );
        }
        on
    })
}

fn async_submit_env() -> bool {
    std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1")
}

/// Whether, while a flight runs on the host GPU, the buffer copies the next
/// submission's lists ask for are brought up to date ([`Host::prewarm`]):
/// `VICTORY_FLIGHT1_PREWARM=1`, off by default, and nothing without
/// Flight-1. The fork's `VICTORY_ASYNC_PREWARM` (halfwave50 02618e8, after
/// Belobug's design), here in the wait for Flight N that Flight N+1 starts
/// with.
pub(crate) fn prewarm_enabled() -> bool {
    #[cfg(test)]
    if TEST_PREWARM.with(std::cell::Cell::get) {
        return flight1_enabled();
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flight1_enabled()
        && *ON.get_or_init(|| {
            let on = crate::switch::on("VICTORY_FLIGHT1_PREWARM");
            if on {
                tracing::info!(
                    "VICTORY_FLIGHT1_PREWARM: while a flight runs on the host GPU, the buffer copies the next \
                     submission's lists ask for are brought up to date, but for those the flight may write"
                );
            }
            on
        })
}

#[cfg(test)]
thread_local! {
    /// `VICTORY_FLIGHT1_PREWARM` on for this thread's submissions.
    pub(crate) static TEST_PREWARM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// `VICTORY_FLIGHTS=2` for this thread's submissions.
    pub(crate) static TEST_FLIGHTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// `VICTORY_FLIGHT_HEAP_SYNC=1` for this thread's submissions.
    pub(crate) static TEST_HEAP_SYNC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How many flights may be in the air at once: `VICTORY_FLIGHTS=<n>` (1 or
/// 2; 1 by default), with Flight-1. With 2, the next submission is
/// prepared and given to the host GPU beside the flight before it when
/// nothing of it can see or disturb what that flight owes
/// ([`Host::beside_refusal`]); flights land strictly in order.
pub(crate) fn flights_limit() -> usize {
    #[cfg(test)]
    if let n @ 1.. = TEST_FLIGHTS.with(std::cell::Cell::get) {
        return n.min(2);
    }
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_FLIGHTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 2);
        if n > 1 {
            tracing::info!(
                "VICTORY_FLIGHTS={n}: with Flight-1, a submission that cannot disturb the flight before it is \
                 prepared and given to the host GPU beside it (its heap sync too); flights land in order"
            );
        }
        n
    })
}

/// Whether, with one flight in the air at most, the next submission's
/// descriptor heap sync runs while the flight before it still runs, before
/// it is landed (`VICTORY_FLIGHT_HEAP_SYNC=1`, with Flight-1). With
/// `VICTORY_FLIGHTS=2` a submission prepared beside a flight syncs the
/// heaps beside it anyway.
pub(crate) fn heap_sync_in_wait() -> bool {
    #[cfg(test)]
    if TEST_HEAP_SYNC.with(std::cell::Cell::get) {
        return flight1_enabled();
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    flight1_enabled()
        && *ON.get_or_init(|| {
            let on = crate::switch::on("VICTORY_FLIGHT_HEAP_SYNC");
            if on {
                tracing::info!(
                    "VICTORY_FLIGHT_HEAP_SYNC: the next submission's descriptor heap sync runs while the flight \
                     before it is still in the air"
                );
            }
            on
        })
}

/// The fewest host submissions a retired object outlives
/// (`crate::host::advance_retired`): with flights beside each other, an
/// object retired while two are in the air may be used by the later one.
pub(crate) fn bury_floor() -> u64 {
    if flight1_enabled() && flights_limit() > 1 {
        flights_limit() as u64
    } else {
        0
    }
}

/// What kept a submission from going to the host GPU beside the flight
/// before it (the overlap line).
#[derive(Clone, Copy, Debug)]
enum Refusal {
    /// As many flights in the air as `VICTORY_FLIGHTS` allows.
    Full,
    /// Coalesced pieces' barrier lists (one set, reused).
    Barriers,
    /// A flight runs programs that use title memory, or this one does.
    Memory,
    /// The flight in the air has texture uploads (their staging is reused).
    Uploads,
    /// It names a list the flight in the air runs.
    SameList,
    /// A step of ours (any code).
    Host,
    /// Its shaders mark the pages they write (one bitmap, taken whole when
    /// a flight lands).
    Marks,
    /// It writes a copy the flight in the air may write.
    Copy,
    /// A title copy or fill over memory the flight in the air may write.
    TitleCopy,
    /// Found while preparing: a refresh of a copy the flight in the air may
    /// write, or split native calls; those in the air land first.
    Late,
}

const REFUSALS: usize = Refusal::Late as usize + 1;

/// Since the last frame line: submissions given beside a flight, and of
/// those that were not with a flight in the air, why ([`Refusal`]); the
/// most flights in the air at once; heap syncs and their microseconds run
/// in the wait for a flight (`VICTORY_FLIGHT_HEAP_SYNC`).
static BESIDE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static REFUSED: [AtomicU64; REFUSALS] = [const { AtomicU64::new(0) }; REFUSALS];

/// Since the start (tests): submissions given beside a flight, and refused.
#[cfg(test)]
static BESIDE_TOTAL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// [`BESIDE_TOTAL`] and heap syncs run in the wait so far (tests).
#[cfg(test)]
pub(crate) fn beside_counts() -> [u64; 3] {
    [
        BESIDE_TOTAL[0].load(Ordering::Relaxed),
        BESIDE_TOTAL[1].load(Ordering::Relaxed),
        HEAP_SYNCS_TOTAL.load(Ordering::Relaxed),
    ]
}

#[cfg(test)]
static HEAP_SYNCS_TOTAL: AtomicU64 = AtomicU64::new(0);

fn refused(why: Refusal) {
    REFUSED[why as usize].fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    BESIDE_TOTAL[1].fetch_add(1, Ordering::Relaxed);
}

/// While a submission is prepared beside flights still in the air
/// ([`Host::execute_flight1`] with `VICTORY_FLIGHTS=2`): what they may write,
/// so that refreshes of those copies wait until they land
/// ([`defer_beside`]), and the refreshes so put off.
struct Beside {
    /// Copies the flights' heaps view for unordered access.
    uav: std::collections::HashSet<usize>,
    /// Title memory whose copies hold shader writes kept from it, joined.
    kept: Vec<(usize, usize)>,
    deferred: Vec<(Arc<crate::shadow::Shadow>, usize, usize)>,
}

thread_local! {
    static BESIDE_PREP: std::cell::RefCell<Option<Beside>> = const { std::cell::RefCell::new(None) };
}

/// Whether this thread prepares a submission beside a flight in the air.
pub(crate) fn beside() -> bool {
    BESIDE_PREP.with(|b| b.borrow().is_some())
}

/// A refresh asked for while preparing beside flights in the air: true
/// (and noted, done once they have landed) if one of them may write the
/// copy. Its bytes there are title memory's older ones until its copy-back,
/// and the refresh would put them over what its shaders wrote.
pub(crate) fn defer_beside(copy: &Arc<crate::shadow::Shadow>, address: usize, len: usize) -> bool {
    BESIDE_PREP.with(|b| {
        let mut b = b.borrow_mut();
        let Some(b) = b.as_mut() else {
            return false;
        };
        let (start, bytes) = copy.range();
        let (start, end) = (start as usize, (start + bytes) as usize);
        let kept = b
            .kept
            .get(b.kept.partition_point(|&(_, e)| e <= start))
            .is_some_and(|&(s, _)| s < end);
        if copy.busy() || kept || b.uav.contains(&(Arc::as_ptr(copy) as usize)) {
            b.deferred.push((copy.clone(), address, len));
            return true;
        }
        false
    })
}

/// A second submission mark, for the flight given beside another (each
/// flight's mark copies its own number).
struct SecondMark(Option<Mark>);
// SAFETY: as `Host`'s own mark: D3D12 objects are free-threaded, and its
// mapped buffers are used under the submission lock only.
unsafe impl Send for SecondMark {}
// SAFETY: as above.
unsafe impl Sync for SecondMark {}
static SECOND_MARK: std::sync::OnceLock<SecondMark> = std::sync::OnceLock::new();

/// Since the last frame line, every host submission (the overlap line):
/// `[0]` how many, `[1]` of them flights, `[2]` microseconds the host GPU
/// ran them (from their launch to when they were seen to have run), `[3]`
/// microseconds a thread of the title's or ours waited for it to. What ran
/// and was not waited for ran while the CPU did other work.
static OVERLAP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A host submission ran on the GPU for `ran`, `waited` of it waited for by
/// a thread of the title's or ours (the overlap line).
pub(super) fn ran_for(ran: Duration, waited: Duration, flight: bool) {
    OVERLAP[0].fetch_add(1, Ordering::Relaxed);
    OVERLAP[1].fetch_add(u64::from(flight), Ordering::Relaxed);
    OVERLAP[2].fetch_add(ran.as_micros() as u64, Ordering::Relaxed);
    OVERLAP[3].fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
}

/// Since the last frame line, of [`Host::prewarm`]: times it had a running
/// flight and lists to look at; of those, times it did nothing as a running
/// list uses title memory, and times the GPU finished first; copies brought
/// up to date and their bytes; copies left alone as the flight may write
/// them, and as their memory is not tracked; microseconds.
static EARLY: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// [`EARLY`] summed since the start (tests: the frame line takes it in
/// turns).
#[cfg(test)]
static EARLY_TOTAL: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// [`EARLY_TOTAL`] (tests).
#[cfg(test)]
pub(crate) fn early_counts() -> [u64; 8] {
    EARLY_TOTAL.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// Tests that hold `victory-flight` off landing, while above 0.
#[cfg(test)]
static LANDER_HELD: AtomicU64 = AtomicU64::new(0);

/// Tests: `victory-flight` lands nothing until dropped (a flight the GPU has
/// run stays in the air until something else lands it).
#[cfg(test)]
pub(crate) struct LanderHeld;

#[cfg(test)]
impl LanderHeld {
    pub(crate) fn on() -> LanderHeld {
        LANDER_HELD.fetch_add(1, Ordering::AcqRel);
        LanderHeld
    }
}

#[cfg(test)]
impl Drop for LanderHeld {
    fn drop(&mut self) {
        LANDER_HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Tests: the host queue runs nothing given to it after this is made until
/// [`Gate::open`] (or the drop): a flight given meanwhile stays in the air.
/// A watchdog opens it after 2 s: another test's submission behind the gate
/// holds the submission lock while it waits, and the thread that would open
/// it may be waiting for that lock. [`Gate::held`] says whether it was the
/// test that opened it (else what it saw "while held" may not have been).
#[cfg(test)]
pub(crate) struct Gate {
    fence: GateFence,
    opened: Arc<std::sync::atomic::AtomicBool>,
    by_watchdog: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
#[derive(Clone)]
struct GateFence(windows::Win32::Graphics::Direct3D12::ID3D12Fence);

// SAFETY: D3D12 fences are free-threaded.
#[cfg(test)]
unsafe impl Send for GateFence {}

#[cfg(test)]
impl GateFence {
    fn open(&self, opened: &std::sync::atomic::AtomicBool) -> bool {
        if opened.swap(true, Ordering::AcqRel) {
            return false;
        }
        // SAFETY: our fence.
        unsafe { self.0.Signal(1) }.expect("the gate opens");
        true
    }
}

#[cfg(test)]
impl Gate {
    pub(crate) fn hold(host: &Host) -> Gate {
        // SAFETY: plain creation; our queue waits on our fence.
        let fence = unsafe {
            let fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence = host
                .device
                .CreateFence(
                    0,
                    windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE,
                )
                .expect("a gate fence");
            host.queue()
                .Wait(&fence, 1)
                .expect("the queue waits on the gate");
            GateFence(fence)
        };
        let opened = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let by_watchdog = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (f, o, w) = (fence.clone(), opened.clone(), by_watchdog.clone());
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(2));
            if f.open(&o) {
                w.store(true, Ordering::Release);
            }
        });
        Gate {
            fence,
            opened,
            by_watchdog,
        }
    }

    pub(crate) fn open(&mut self) {
        self.fence.open(&self.opened);
    }

    /// Whether the watchdog has not opened it (so far).
    pub(crate) fn held(&self) -> bool {
        !self.by_watchdog.load(Ordering::Acquire)
    }
}

#[cfg(test)]
impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
    }
}

/// The copies that took pages early, held until the flight that ran
/// meanwhile is landed (`crate::shadow::forget_early`).
static EARLY_HELD: std::sync::Mutex<Vec<Arc<crate::shadow::Shadow>>> =
    std::sync::Mutex::new(Vec::new());

/// The frame lines of host submissions (`frames` since the last one): how
/// much of the host GPU's time the CPU waited for and how much ran hidden
/// behind its work, and with the prewarm on, what it did.
pub(super) fn log(frames: f64) {
    let per = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / frames;
    let [n, flights, ran, waited] = OVERLAP.each_ref().map(per);
    let (ran, waited) = (ran / 1000.0, waited / 1000.0);
    let hidden = (ran - waited).max(0.0);
    let [beside, most, syncs, sync_us] =
        [0, 1, 2, 3].map(|i| BESIDE[i].swap(0, Ordering::Relaxed) as f64);
    let refused: [f64; REFUSALS] = std::array::from_fn(|i| per(&REFUSED[i]));
    tracing::info!(
        "overlap: {n:.1} host submissions/frame ({flights:.1} of them flights, VICTORY_FLIGHT1; {:.1} given beside \
         another, VICTORY_FLIGHTS, at most {most:.0} in the air); the GPU ran them {ran:.1} ms/frame; threads \
         waited {waited:.1} ms/frame for it; {hidden:.1} ms/frame ({:.0}%) ran hidden behind CPU work; kept from \
         going beside a flight by (in the air already, barrier lists, title memory, uploads, the same list, a \
         step of ours, write marks, a copy it writes, a title copy, found while preparing): {refused:.1?}/frame; \
         heap syncs in the wait: {:.1}/frame, {:.1} ms/frame",
        beside / frames,
        if ran > 0.0 { 100.0 * hidden / ran } else { 0.0 },
        syncs / frames,
        sync_us / 1000.0 / frames,
    );
    if !prewarm_enabled() {
        return;
    }
    let early: [f64; 8] = std::array::from_fn(|i| per(&EARLY[i]));
    let marked = per(&crate::shadow::EARLY_MARKED);
    tracing::info!(
        "prewarm: {:.1} ms/frame of the next lists' preparation done while the host GPU ran a flight, in {:.1} \
         goes/frame ({:.1} of them did nothing: a running list uses title memory; {:.1} stopped as the GPU \
         finished first): {:.0} buffer copies/frame brought up to date early ({:.2} MB/frame), {:.0} left alone as \
         the flight may write them and {:.0} as their memory is not tracked; {marked:.1} pages/frame taken early \
         were marked as written by the flight's shaders all the same (the title's bytes kept; 0 if the rule holds)",
        early[7] / 1000.0,
        early[0],
        early[1],
        early[2],
        early[3],
        early[4] / 1e6,
        early[5],
        early[6],
    );
}

/// What [`Host::prewarm`] gathers: the ranges of each copy to bring up to
/// date (ranges less than a page apart as one, as a submission's own
/// refreshes), and the copies left alone.
#[derive(Default)]
struct Early {
    wanted: Vec<(Arc<crate::shadow::Shadow>, crate::ranges::Ranges)>,
    /// Where each copy is in `wanted`, by copy.
    index: std::collections::HashMap<usize, usize>,
    /// Copies the flight's heaps view for unordered access.
    written: std::collections::HashSet<usize>,
    /// Title memory whose copies hold shader writes kept from it: where
    /// each stretch starts and ends, in order, none touching the next.
    kept: Vec<(usize, usize)>,
    /// Copies decided against, and how many: as the flight may write them,
    /// as their memory is not tracked.
    left: std::collections::HashSet<usize>,
    alone: [u64; 2],
}

impl Early {
    fn add(&mut self, copy: &Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        // Nothing to copy, as for a submission's own refreshes.
        if len == 0 || copy.clean() {
            return;
        }
        let key = Arc::as_ptr(copy) as usize;
        if self.left.contains(&key) {
            return;
        }
        let i = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                let (start, bytes) = copy.range();
                let (start, end) = (start as usize, (start + bytes) as usize);
                let why = if copy.dirty_bit().is_none() {
                    Some(1)
                } else if !copy.idle()
                    || self.written.contains(&key)
                    || self
                        .kept
                        .get(self.kept.partition_point(|&(_, e)| e <= start))
                        .is_some_and(|&(s, _)| s < end)
                {
                    Some(0)
                } else {
                    None
                };
                if let Some(why) = why {
                    self.left.insert(key);
                    self.alone[why] += 1;
                    return;
                }
                self.wanted.push((copy.clone(), Default::default()));
                self.index.insert(key, self.wanted.len() - 1);
                self.wanted.len() - 1
            }
        };
        self.wanted[i]
            .1
            .insert(address, address + len, crate::watch::PAGE);
    }
}

#[cfg(test)]
static TEST_FLIGHT1: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
static TEST_ASYNC_SUBMIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
static TEST_MODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) struct ModeGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev_flight1: u8,
    prev_async: u8,
}

#[cfg(test)]
impl ModeGuard {
    pub(crate) fn new(flight1: bool, async_submit: bool) -> Self {
        let lock = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::queue::drain_submit(crate::queue::Why::Present);
        if let Some(h) = super::host() {
            h.finish_flight_now();
        }
        let prev_flight1 = TEST_FLIGHT1.swap(if flight1 { 2 } else { 1 }, Ordering::AcqRel);
        let prev_async = TEST_ASYNC_SUBMIT.swap(if async_submit { 2 } else { 1 }, Ordering::AcqRel);
        Self {
            _lock: lock,
            prev_flight1,
            prev_async,
        }
    }
}

#[cfg(test)]
impl Drop for ModeGuard {
    fn drop(&mut self) {
        crate::queue::drain_submit(crate::queue::Why::Present);
        if let Some(h) = super::host() {
            h.finish_flight_now();
        }
        TEST_FLIGHT1.store(self.prev_flight1, Ordering::Release);
        TEST_ASYNC_SUBMIT.store(self.prev_async, Ordering::Release);
    }
}

/// Protected by `Host::flight` (leaf lock strictly below `HELD` and
/// `Host::submitted`, above only `Fence::waiters`).
#[derive(Default)]
pub(super) struct FlightState {
    /// The launched flights awaiting wait + landing, oldest first (one, or
    /// two with `VICTORY_FLIGHTS=2`); the oldest is taken by whichever
    /// thread (`victory-flight` or a submission) lands it under
    /// `Host::submitted`, and they land in that order.
    in_flight: std::collections::VecDeque<InFlight>,
    /// Flights taken out of `in_flight` and being landed (0 or 1: landing
    /// holds `Host::submitted`). A flight counts as active from its launch
    /// until its post-GPU copy-back, BoP writes and attached signals have
    /// all completed; while one is, `attach_signal` queues behind the newest.
    landing: usize,
    /// Title fences attached while the flight being landed was the only
    /// one: delivered after its own, at the end of that landing.
    landing_signals: Vec<(usize, u64)>,
    /// Whether the background `victory-flight` finisher thread has been spawned.
    worker_started: bool,
}

impl FlightState {
    fn active(&self) -> bool {
        !self.in_flight.is_empty() || self.landing > 0
    }
}

struct InFlight {
    fence_value: u64,
    segments: Vec<Arc<HostList>>,
    /// Whether one of its lists runs programs that use title memory, which
    /// write host copies at addresses they compute (`crate::memory`).
    memory: bool,
    /// Title memory its lists declared that their shaders write (address,
    /// end), and the copies its heaps view for unordered access, with their
    /// whole ranges: what a flight beside it must not write or copy over.
    writes: Vec<(usize, usize)>,
    uav: std::collections::HashSet<usize>,
    /// Title fences attached behind it (each kept alive via `fence::keep`),
    /// delivered when it lands, after its copy-back.
    signals: Vec<(usize, u64)>,
    /// Which mark it ends with (0 the host's, 1 [`SECOND_MARK`]).
    slot: usize,
    uploads: Option<crate::bindless::Uploads>,
    batch: Batch,
    heaps: Vec<Arc<crate::bindless::Viewed>>,
    split_count: usize,
    started: std::time::Instant,
    prepared: std::time::Instant,
    launched: std::time::Instant,
    waited: bool,
}

// SAFETY: Every field is `Send` (D3D12 objects are free-threaded; `HostList`
// and `Uploads` — including its owned upload list and `ReadyDrain` — are
// `Send`; `Batch` holds `Arc<Shadow>` and ranges).
unsafe impl Send for InFlight {}

impl Host {
    /// Runs `segments` with Wait-Before-Prep Flight-1 overlap when enabled:
    /// finishes any prior flight first, prepares and launches `segments` on
    /// the host queue, and returns immediately while the GPU executes.
    /// Falls back to synchronous execution when Flight-1 is disabled or when
    /// per-submission diagnostics tracing is active on this frame.
    pub(crate) fn execute_flight1(&self, segments: Vec<Arc<HostList>>, barrier_before: Vec<bool>) {
        if segments.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(not(feature = "diagnostics"))]
        let traced = false;
        if !flight1_enabled() || traced {
            self.execute_sync_segments(segments, barrier_before);
            return;
        }

        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        // While Flight N-1 still runs, the buffer copies these lists ask for
        // that it cannot write are brought up to date
        // (`VICTORY_FLIGHT1_PREWARM`): their preparation below finds them done.
        if prewarm_enabled() {
            self.prewarm(&segments);
        }
        // And the heap sync (`VICTORY_FLIGHT_HEAP_SYNC`), before the flight
        // is landed: the sync below finds what the title wrote since.
        if heap_sync_in_wait() && flights_limit() == 1 {
            self.sync_in_wait();
        }
        // Wait-Before-Prep: drain Flight N-1 completely before Flight N
        // touches any shared Shadow, descriptor heap, upload staging pool, or
        // Mark buffer; with `VICTORY_FLIGHTS=2`, only what the flights in the
        // air keep it from going beside.
        loop {
            let in_air = self
                .flight
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .in_flight
                .len();
            if in_air == 0 {
                break;
            }
            let why = if in_air >= flights_limit() {
                Some(Refusal::Full)
            } else {
                self.beside_refusal(&segments, &barrier_before)
            };
            let Some(why) = why else { break };
            if flights_limit() > 1 {
                refused(why);
            }
            self.finish_flight_locked(&mut submitted, None, None);
        }
        let mut beside = self.in_air() > 0;

        let prep = {
            let run_refs: Vec<&HostList> = segments.iter().map(|s| &**s).collect();
            if beside {
                self.enter_beside();
            }
            self.prepare_submission(&submitted, &run_refs, &barrier_before, false)
        };
        let deferred = BESIDE_PREP
            .with(|b| b.borrow_mut().take())
            .map_or_else(Vec::new, |b| b.deferred);
        // Found while preparing: what still keeps it from going beside. The
        // flights in the air land first, then the refreshes put off.
        if beside
            && (!deferred.is_empty()
                || !prep.native_boundaries.is_empty()
                || prep.heaps.iter().any(|h| {
                    let mut uav = false;
                    crate::bindless::written(h, |_| uav = true);
                    uav
                }))
        {
            refused(Refusal::Late);
            self.land_all_locked(&mut submitted);
            beside = false;
        }
        for (copy, address, len) in deferred {
            copy.update(address, len);
        }
        let slot = if beside {
            let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            BESIDE[0].fetch_add(1, Ordering::Relaxed);
            #[cfg(test)]
            BESIDE_TOTAL[0].fetch_add(1, Ordering::Relaxed);
            BESIDE[1].fetch_max(state.in_flight.len() as u64 + 1, Ordering::Relaxed);
            state.in_flight.back().map_or(0, |f| 1 - f.slot)
        } else {
            BESIDE[1].fetch_max(1, Ordering::Relaxed);
            0
        };

        let mut native_boundaries = prep.native_boundaries;
        let mark = self.mark_for(slot);
        if mark.is_none() {
            native_boundaries.clear();
        }
        let mut split_pool = (!native_boundaries.is_empty())
            .then(|| self.split_marks.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(pool) = split_pool.as_deref_mut() {
            if let Some(marks) = pool.as_mut()
                && let Err(error) = (marks.len()..native_boundaries.len()).try_for_each(|_| {
                    marks.push(Mark::new(self)?);
                    Ok::<_, windows::core::Error>(())
                })
            {
                tracing::warn!(
                    %error,
                    "could not make split submission marks, keeping the native batch together"
                );
                *pool = None;
            }
            if pool.is_none() {
                native_boundaries.clear();
            }
        }
        let split_marks = split_pool
            .as_deref()
            .and_then(Option::as_deref)
            .unwrap_or(&[]);

        *submitted += 1;
        let fence_value = *submitted;
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, fence_value, 0);

        let mut group = prep.lists;
        if let Some(mark) = mark {
            mark.arm(fence_value);
            group.push(mark.list());
        }
        let queue = self.queue();
        let launched = std::time::Instant::now();
        // SAFETY: closed lists; fence and events are ours.
        let waited = unsafe {
            let mut first = 0;
            for (&end, m) in native_boundaries.iter().zip(split_marks) {
                m.arm(fence_value);
                let mut marked = group[first..end].to_vec();
                marked.push(m.list());
                #[cfg(test)]
                super::tests::native_call(marked.len());
                queue.ExecuteCommandLists(&marked);
                first = end;
            }
            #[cfg(test)]
            super::tests::native_call(group.len() - first);
            queue.ExecuteCommandLists(&group[first..]);
            crate::awake::sent(&self.device);
            match queue.Signal(&self.fence, fence_value).and_then(|()| {
                self.fence.SetEventOnCompletion(fence_value, self.event)?;
                self.fence
                    .SetEventOnCompletion(fence_value, self.flight_event)
            }) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(error = %e, "host flight submission not signalled");
                    false
                }
            }
        };
        drop(split_pool);

        let split_count = native_boundaries.len();
        let memory = segments.iter().any(|l| {
            l.deferred()
                .copies
                .iter()
                .any(|s| matches!(s, Step::Memory(_)))
        });
        let (writes, uav) = writes_of(&segments, &prep.heaps);
        let flight = InFlight {
            fence_value,
            segments,
            memory,
            writes,
            uav,
            signals: Vec::new(),
            slot,
            uploads: prep.uploads,
            batch: prep.batch,
            heaps: prep.heaps,
            split_count,
            started: prep.started,
            prepared: prep.prepared,
            launched,
            waited,
        };
        {
            let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert!(state.in_flight.len() < flights_limit() && state.landing == 0);
            state.in_flight.push_back(flight);
            if !state.worker_started {
                state.worker_started = true;
                spawn_flight_worker();
            }
        }
        self.flight_wake.notify_one();
    }

    /// Runs `segments` synchronously and recycles them afterward. Used by the
    /// submit worker when `VICTORY_ASYNC_SUBMIT=1` with `VICTORY_FLIGHT1=0`
    /// and as the fallback for traced frames.
    pub(crate) fn execute_sync_segments(
        &self,
        segments: Vec<Arc<HostList>>,
        barrier_before: Vec<bool>,
    ) {
        let lists: Vec<&HostList> = segments.iter().map(|s| &**s).collect();
        self.execute_sync(&lists, &barrier_before);
        drop(lists);
        segments.into_iter().for_each(crate::list::recycle);
    }

    /// Attaches a title fence signal behind the active flight if one is in the
    /// air. Returns `true` if attached (the caller must NOT call `fence::set`
    /// immediately), or `false` if no flight is active.
    ///
    /// If `already_kept` is `false`, this calls `fence::keep(fence)` before
    /// storing it.
    pub(crate) fn attach_signal(&self, fence: This, value: u64, already_kept: bool) -> bool {
        let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        if !state.active() {
            return false;
        }
        if !already_kept {
            crate::fence::keep(fence);
        }
        // Behind the newest flight: the queue's work before the Signal is
        // in it or in one before it, which land first.
        match state.in_flight.back_mut() {
            Some(newest) => newest.signals.push((fence as usize, value)),
            None => state.landing_signals.push((fence as usize, value)),
        }
        true
    }

    /// Whether a flight is currently in the air or landing.
    pub(crate) fn has_in_flight(&self) -> bool {
        self.flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active()
    }

    /// How many flights are in the air (not being landed).
    pub(crate) fn in_air(&self) -> usize {
        self.flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .in_flight
            .len()
    }

    /// Waits for and lands every active flight now, oldest first. Returns
    /// `true` if a flight was landed by this call.
    pub(crate) fn finish_flight_now(&self) -> bool {
        if !self.has_in_flight() {
            return false;
        }
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.land_all_locked(&mut submitted)
    }

    /// Lands every flight in the air, oldest first, under
    /// `Host::submitted`: whether there was one.
    pub(super) fn land_all_locked(&self, submitted: &mut u64) -> bool {
        let mut landed = false;
        while self.finish_flight_locked(submitted, None, None) {
            landed = true;
        }
        landed
    }

    /// The mark a flight in slot `slot` ends with.
    fn mark_for(&self, slot: usize) -> Option<&Mark> {
        if slot == 0 {
            return self.mark();
        }
        SECOND_MARK
            .get_or_init(|| {
                SecondMark(
                    Mark::new(self)
                        .inspect_err(|e| tracing::warn!(error = %e, "no second submission mark"))
                        .ok(),
                )
            })
            .0
            .as_ref()
    }

    /// Whether `next` (the lists of a submission) may be prepared and given
    /// to the host GPU beside the flights in the air, and if not, why: it
    /// sets no write marks (the bitmap is taken whole, by plain reads and
    /// writes, when a flight lands, and the GPU may set bits meanwhile), it
    /// writes no copy and title-copies over no memory a flight in the air
    /// may write, and nothing of it is shared with them (barrier lists,
    /// lists, staging of uploads). Its refreshes of copies a flight may
    /// write are put off until that flight lands ([`defer_beside`]).
    fn beside_refusal(&self, next: &[Arc<HostList>], barrier_before: &[bool]) -> Option<Refusal> {
        if barrier_before.iter().any(|&b| b) {
            return Some(Refusal::Barriers);
        }
        let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        let flights = &state.in_flight;
        if flights.iter().any(|f| f.memory) {
            return Some(Refusal::Memory);
        }
        if flights.iter().any(|f| f.uploads.is_some()) {
            return Some(Refusal::Uploads);
        }
        if next.iter().any(|l| {
            flights
                .iter()
                .any(|f| f.segments.iter().any(|s| Arc::ptr_eq(s, l)))
        }) {
            return Some(Refusal::SameList);
        }
        let over = |at: usize, len: usize| {
            flights
                .iter()
                .any(|f| f.writes.iter().any(|&(a, e)| a < at + len && at < e))
        };
        let uav = |copy: &Arc<crate::shadow::Shadow>| {
            let key = Arc::as_ptr(copy) as usize;
            flights.iter().any(|f| f.uav.contains(&key))
        };
        for list in next {
            for step in &list.deferred().copies {
                let why = match step {
                    Step::Host(_) => Some(Refusal::Host),
                    Step::Memory(_) => Some(Refusal::Memory),
                    Step::GpuWrite(_, _, _, true, _) => Some(Refusal::Marks),
                    Step::GpuWrite(copy, ..) if copy.busy() || uav(copy) => Some(Refusal::Copy),
                    Step::Heap(viewed) => {
                        let mut marks = false;
                        crate::bindless::written(viewed, |_| marks = true);
                        marks.then_some(Refusal::Marks)
                    }
                    &Step::Copy(dst, _, len) | &Step::Move(dst, _, len) if over(dst, len) => {
                        Some(Refusal::TitleCopy)
                    }
                    &Step::Fill(_, src, len) if over(src, len) => Some(Refusal::TitleCopy),
                    _ => None,
                };
                if why.is_some() {
                    return why;
                }
            }
        }
        None
    }

    /// Starts this thread's preparation beside the flights in the air
    /// ([`defer_beside`] then puts off refreshes of what they may write).
    fn enter_beside(&self) {
        let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        let uav = state
            .in_flight
            .iter()
            .flat_map(|f| f.uav.iter().copied())
            .collect();
        drop(state);
        let mut kept = crate::memory::kept();
        kept.sort_unstable();
        let mut joined: Vec<(usize, usize)> = Vec::new();
        for (at, n) in kept {
            match joined.last_mut() {
                Some((_, end)) if at <= *end => *end = (*end).max(at + n),
                _ => joined.push((at, at + n)),
            }
        }
        BESIDE_PREP.with(|b| {
            *b.borrow_mut() = Some(Beside {
                uav,
                kept: joined,
                deferred: Vec::new(),
            });
        });
    }

    /// `VICTORY_FLIGHT_HEAP_SYNC`: the descriptor heap sync, while the
    /// flight in the air still runs (under the submission lock, before it
    /// is landed). What a sync makes are the slots whose records the title
    /// rewrote since the one before the flight was prepared, which is for
    /// work it gave after: on the console the GPU reads the heap while
    /// earlier work runs, so the flight does not read those slots (the
    /// title's threads write views there meanwhile already). What a slot
    /// viewed before is retired for `bury_lag` submissions; a texture view
    /// left null until read forgets its read mark, which the flight's
    /// landing then takes or a later read sets again.
    fn sync_in_wait(&self) {
        let value = {
            let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            match state.in_flight.back() {
                Some(f) if f.waited => f.fence_value,
                _ => return,
            }
        };
        // SAFETY: a plain query of our fence.
        if unsafe { self.fence.GetCompletedValue() } >= value {
            return;
        }
        let started = Instant::now();
        crate::descriptor::sync_heaps();
        BESIDE[2].fetch_add(1, Ordering::Relaxed);
        BESIDE[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(test)]
        HEAP_SYNCS_TOTAL.fetch_add(1, Ordering::Relaxed);
    }

    /// Completes any active flight under `Host::submitted`. When
    /// `only_if_fence` is `Some(v)`, only lands if the active flight matches
    /// fence value `v` (used by `victory-flight` so it never blocks on a newer
    /// flight while holding `Host::submitted`). `waited_at`, when provided by
    /// `victory-flight`, records the exact instant the GPU fence completed so
    /// lock-wait latency is never billed as GPU execution time.
    pub(super) fn finish_flight_locked(
        &self,
        submitted: &mut u64,
        only_if_fence: Option<u64>,
        waited_at: Option<std::time::Instant>,
    ) -> bool {
        let flight = {
            let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            match state.in_flight.front() {
                Some(f) if only_if_fence.is_none_or(|v| f.fence_value == v) => {
                    state.landing += 1;
                    state.in_flight.pop_front()
                }
                _ => return false,
            }
        };
        let Some(flight) = flight else {
            return false;
        };

        let mut waited = Duration::ZERO;
        let gpu_done = if flight.waited {
            // SAFETY: plain fence completion query.
            if unsafe { self.fence.GetCompletedValue() } < flight.fence_value {
                let asked = Instant::now();
                self.wait_for(flight.fence_value);
                waited = asked.elapsed();
                std::time::Instant::now()
            } else {
                waited_at.unwrap_or_else(std::time::Instant::now)
            }
        } else {
            std::time::Instant::now()
        };
        let gpu_duration = gpu_done.saturating_duration_since(flight.launched);
        ran_for(gpu_duration, waited, true);
        #[cfg(test)]
        super::tests::native_call_group(flight.waited);

        let mark = self.mark_for(flight.slot);
        let split_failed = if flight.split_count > 0 {
            let pool = self.split_marks.lock().unwrap_or_else(|e| e.into_inner());
            pool.as_deref().is_some_and(|marks| {
                marks
                    .iter()
                    .take(flight.split_count)
                    .any(|m| !m.saw(flight.fence_value))
            })
        } else {
            false
        };
        // This flight's own number: another may be in the air after it
        // (`VICTORY_FLIGHTS=2`), so not the last one given, unless a check
        // of the queue ran after it (then every earlier one has run).
        let mut done = flight.fence_value;
        if flight.waited
            && let Some(mark) = mark
            && (!mark.saw(flight.fence_value) || split_failed)
            // SAFETY: a plain query.
            && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
        {
            self.lost(submitted, mark);
            done = *submitted;
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, flight.fence_value, 0);
        super::SUBMISSIONS_DONE.fetch_max(done, Ordering::AcqRel);
        // The GPU fence has completed: return worker-owned staging chunks now
        // at the same point as the synchronous path.
        if let Some(u) = &flight.uploads {
            u.release_worker();
        }

        {
            let run_refs: Vec<&HostList> = flight
                .uploads
                .as_ref()
                .map(|u| u.list())
                .into_iter()
                .chain(flight.segments.iter().map(|s| &**s))
                .collect();
            self.finish_post_gpu(
                &run_refs,
                flight.batch,
                flight.heaps,
                flight.started,
                flight.prepared,
                gpu_duration,
                None,
                done,
            );
        }

        // What copies took early while it ran is as anything else now (its
        // copy-back, above, kept the title's bytes on such pages).
        crate::shadow::forget_early();
        EARLY_HELD.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // Return the upload list to `bindless::LIST` and recycle host lists
        // only after the GPU fence and post-GPU work have finished.
        drop(flight.uploads);
        flight.segments.into_iter().for_each(crate::list::recycle);

        // Deliver every title fence signal queued behind this flight (then
        // those attached while it was landed as the only one) atomically
        // before it stops counting as active, so later `queue::signal` calls
        // can never overtake an attached signal; those behind a later
        // flight wait for it.
        let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        let after = std::mem::take(&mut state.landing_signals);
        for (fence, value) in flight.signals.into_iter().chain(after) {
            crate::fence::set(fence as This, value);
            crate::fence::let_go(fence as This);
        }
        state.landing -= 1;
        true
    }
}

impl Host {
    /// While the flight on the host GPU still runs: brings up to date the
    /// buffer copies `next` (the lists of the submission after it) will ask
    /// for, as their own preparation does after landing it (their refresh
    /// steps, and what their heaps' views reach), so that preparation finds
    /// them done. Under the submission lock, before the flight is landed;
    /// stops when the GPU has run it.
    ///
    /// A copy is left alone, for the lists' own preparation, if the flight
    /// may write it or its early refresh could differ from the one after:
    ///
    /// - a list submitted declared that its shaders write it (a range
    ///   pending or a snapshot kept, `Shadow::idle`): title memory there is
    ///   older than the copy until the copy-back;
    /// - a heap the flight's lists take views from views it for unordered
    ///   access: their shaders may write it with no list declaring it;
    /// - every copy, if a flight's list uses title memory: its programs
    ///   write copies at addresses they compute;
    /// - it holds shader writes kept out of title memory until the frame's
    ///   end (`crate::memory::kept`): a refresh drops them from the pages
    ///   the title wrote, and the flight may read them;
    /// - its memory is not tracked or is aliased: copied whole at each use,
    ///   or never.
    ///
    /// The others take the pages the title wrote since they last took them.
    /// What the flight reads of them it read from pages its own preparation
    /// took; a page the title writes after that, while the GPU reads it, is
    /// a race on the console too, where the GPU reads title memory itself.
    /// A page taken early that the flight's shaders marked as written all
    /// the same is counted and keeps the title's bytes
    /// (`Shadow::kept_early`).
    ///
    /// Not done early: steps that write title memory or upload memory
    /// (their order with the flight's copy-back counts), texture uploads
    /// (the staging the flight's upload list reads is reused), the heaps'
    /// descriptor sync and the title-memory tables (the flight's shaders
    /// read both), steps of ours.
    pub(super) fn prewarm(&self, next: &[Arc<HostList>]) {
        let (value, memory, heaps) = {
            let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            let Some(newest) = state.in_flight.back().filter(|f| f.waited) else {
                return;
            };
            let heaps: Vec<_> = state
                .in_flight
                .iter()
                .flat_map(|f| f.heaps.iter().cloned())
                .collect();
            (
                newest.fence_value,
                state.in_flight.iter().any(|f| f.memory),
                heaps,
            )
        };
        // SAFETY: a plain query of our fence.
        let ran = || unsafe { self.fence.GetCompletedValue() } >= value;
        if ran() {
            return;
        }
        let started = Instant::now();
        EARLY[0].fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        EARLY_TOTAL[0].fetch_add(1, Ordering::Relaxed);
        if memory {
            EARLY[1].fetch_add(1, Ordering::Relaxed);
            #[cfg(test)]
            EARLY_TOTAL[1].fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut early = Early::default();
        for heap in &heaps {
            crate::bindless::written(heap, |copy| {
                early.written.insert(Arc::as_ptr(copy) as usize);
            });
        }
        // In order and joined, to be searched once per copy.
        let mut kept = crate::memory::kept();
        kept.sort_unstable();
        for (at, n) in kept {
            match early.kept.last_mut() {
                Some((_, end)) if at <= *end => *end = (*end).max(at + n),
                _ => early.kept.push((at, at + n)),
            }
        }
        let mut cut = false;
        for list in next {
            if ran() {
                cut = true;
                break;
            }
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(copy, address, len) => early.add(copy, *address, *len),
                    Step::Slot(viewed, index) => {
                        crate::bindless::refresh_slot(viewed, *index, |c, a, l| early.add(c, a, l))
                    }
                    Step::Heap(viewed) => {
                        crate::bindless::refresh(viewed, |c, a, l| early.add(c, a, l))
                    }
                    _ => {}
                }
            }
        }
        let (mut copies, mut bytes) = (0, 0);
        let mut held = Vec::with_capacity(early.wanted.len());
        // The fence is asked at most every 200 microseconds.
        let mut asked = Instant::now();
        for (copy, ranges) in early.wanted {
            if cut {
                break;
            }
            if asked.elapsed() >= Duration::from_micros(200) {
                asked = Instant::now();
                if ran() {
                    cut = true;
                    break;
                }
            }
            let mut taken = Some(0);
            for (from, to) in ranges.iter() {
                taken = taken.and_then(|n| Some(n + copy.update_early(from, to - from)?));
            }
            match taken {
                Some(n) => {
                    copies += 1;
                    bytes += n as u64;
                    held.push(copy);
                }
                None => early.alone[0] += 1,
            }
        }
        // Held until the flight is landed: the pages they took are noted by
        // their addresses.
        EARLY_HELD
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(held);
        EARLY[2].fetch_add(u64::from(cut), Ordering::Relaxed);
        let counted = [
            0,
            0,
            u64::from(cut),
            copies,
            bytes,
            early.alone[0],
            early.alone[1],
            started.elapsed().as_micros() as u64,
        ];
        for (i, n) in counted.into_iter().enumerate().skip(2) {
            EARLY[i].fetch_add(n, Ordering::Relaxed);
            #[cfg(test)]
            EARLY_TOTAL[i].fetch_add(n, Ordering::Relaxed);
        }
    }
}

/// What the lists of a flight may write (`InFlight::writes`, `uav`): the
/// title memory they declared that their shaders write, and the copies
/// their heaps view for unordered access, with their whole ranges.
fn writes_of(
    segments: &[Arc<HostList>],
    heaps: &[Arc<crate::bindless::Viewed>],
) -> (Vec<(usize, usize)>, std::collections::HashSet<usize>) {
    let mut writes = Vec::new();
    for list in segments {
        for step in &list.deferred().copies {
            if let &Step::GpuWrite(_, at, len, _, _) = step {
                writes.push((at, at + len));
            }
        }
    }
    let mut uav = std::collections::HashSet::new();
    for heap in heaps {
        crate::bindless::written(heap, |copy| {
            if uav.insert(Arc::as_ptr(copy) as usize) {
                let (start, len) = copy.range();
                writes.push((start as usize, (start + len) as usize));
            }
        });
    }
    (writes, uav)
}

fn spawn_flight_worker() {
    let _ = std::thread::Builder::new()
        .name("victory-flight".into())
        .spawn(|| {
            let Some(host) = super::host() else {
                return;
            };
            loop {
                let (fence_value, waited) = {
                    let mut state = host.flight.lock().unwrap_or_else(|e| e.into_inner());
                    while state.in_flight.is_empty() {
                        state = host
                            .flight_wake
                            .wait(state)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                    let f = state.in_flight.front().expect("in_flight present");
                    (f.fence_value, f.waited)
                };
                // A test that needs a flight the GPU has run left unlanded.
                #[cfg(test)]
                if LANDER_HELD.load(Ordering::Acquire) > 0 {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                // Wait on the GPU fence WITHOUT holding `HELD` or
                // `Host::submitted`.
                if waited {
                    host.wait_for_on_event(host.flight_event, fence_value, 250);
                }
                let gpu_done = std::time::Instant::now();
                // Skip acquiring `Host::submitted` if `victory-submit` already
                // claimed and began landing this flight while we waited.
                let still_mine = host
                    .flight
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .in_flight
                    .front()
                    .is_some_and(|f| f.fence_value == fence_value);
                if !still_mine {
                    continue;
                }
                let landed = {
                    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
                    let _holding = Holding::enter();
                    host.finish_flight_locked(&mut submitted, Some(fence_value), Some(gpu_done))
                };
                // Strictly AFTER dropping `Host::submitted`: wake held work or
                // the submit worker in case this flight's BoP writes or fence
                // signals unblocked another queue.
                if landed {
                    crate::queue::on_flight_landed();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::backend::{HostGpu, HostMemory};
    use crate::host::host;
    use crate::object::{S_OK, This, slot};
    use crate::queue::Written;

    fn make_queue_and_fence() -> (This, This) {
        let q = crate::queue::create_direct_for_test();
        let mut f = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut f
            ),
            S_OK
        );
        (q, f)
    }

    fn queue_signal(q: This, f: This, value: u64) {
        let hr = crate::queue::signal_for_test(q, f, value);
        assert_eq!(hr, S_OK);
    }

    fn queue_execute(q: This, lists: &[This]) {
        crate::queue::execute_for_test(q, lists);
    }

    fn make_title_list() -> (This, This) {
        let null = std::ptr::null();
        let alloc_iid = crate::object::guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = crate::object::guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let mut alloc = null;
        assert_eq!(
            crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list),
            S_OK
        );
        (alloc, list)
    }

    fn close_title_list(list: This) {
        let close = slot::<extern "system" fn(This) -> i32>(list, 0x48);
        assert_eq!(close(list), S_OK);
    }

    fn reset_title_list(list: This, alloc: This) {
        let reset = slot::<extern "system" fn(This, This, This) -> i32>(list, 0x50);
        assert_eq!(reset(list, alloc, std::ptr::null()), S_OK);
    }

    fn wait_fence(f: This, target: u64, timeout: Duration) -> bool {
        let start = Instant::now();
        while crate::fence::value(f) < target {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// Under `VICTORY_FLIGHT1=1`, a GPU copy + Shadow copy-back + `after`
    /// callback + bottom-of-pipe write + attached `queue::signal` all land in
    /// order via the background `victory-flight` finisher without requiring a
    /// second submission.
    #[test]
    fn flight1_delivers_signals_bop_writes_and_copy_back_in_order() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);

        let (q, f) = make_queue_and_fence();
        let mut title_buf = vec![0u32; 16];
        let title_addr = title_buf.as_mut_ptr() as u64;
        let shadow = crate::shadow::find(title_addr, 64).expect("shadow");

        // Prepare source upload buffer containing 0xCAFE_BABE.
        let (src_buf, src_ptr) = host.upload_buffer(64).expect("upload buffer");
        // SAFETY: mapped 64-byte upload buffer.
        unsafe {
            src_ptr.cast::<u32>().write_volatile(0xCAFE_BABE);
        }

        let seg = Arc::new(host.new_list().expect("host list"));
        // SAFETY: open host list, valid D3D12 buffers.
        unsafe {
            seg.list
                .CopyBufferRegion(&shadow.buffer, shadow.offset(title_addr), &src_buf, 0, 4);
        }
        seg.keep(&src_buf);
        shadow.gpu_writes(&seg, title_addr, 4, false);

        let after_ran = Arc::new(AtomicBool::new(false));
        let after_flag = after_ran.clone();
        let title_ptr_num = title_addr as usize;
        let saw_copy_back_in_after = Arc::new(AtomicBool::new(false));
        let saw_cb = saw_copy_back_in_after.clone();
        let mut bop_word: u64 = 0;
        let bop_ptr_num = (&raw mut bop_word) as usize;
        let saw_bop_not_yet_written = Arc::new(AtomicBool::new(false));
        let saw_bop = saw_bop_not_yet_written.clone();

        seg.defer_after(move || {
            after_flag.store(true, Ordering::Release);
            // SAFETY: title_buf and bop_word outlive the flight.
            unsafe {
                let copied = (title_ptr_num as *const u32).read_volatile();
                if copied == 0xCAFE_BABE {
                    saw_cb.store(true, Ordering::Release);
                }
                if (bop_ptr_num as *const u64).read_volatile() == 0 {
                    saw_bop.store(true, Ordering::Release);
                }
            }
        });
        seg.defer_write(bop_ptr_num as *mut u8, Written::Value(0x1234_5678), 8);
        seg.close();

        host.execute_flight1(vec![seg], vec![]);
        queue_signal(q, f, 7);

        // Background `victory-flight` must finish the flight and deliver the
        // attached fence signal even with no further submissions.
        assert!(
            wait_fence(f, 7, Duration::from_secs(5)),
            "victory-flight did not deliver attached fence signal"
        );
        assert!(after_ran.load(Ordering::Acquire), "defer_after did not run");
        assert!(
            saw_copy_back_in_after.load(Ordering::Acquire),
            "copy_back must complete before defer_after"
        );
        assert!(
            saw_bop_not_yet_written.load(Ordering::Acquire),
            "BoP writes must run after defer_after"
        );
        // SAFETY: flight has landed and fence reached 7.
        assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 0x1234_5678);
        assert_eq!(title_buf[0], 0xCAFE_BABE);
        assert!(!host.has_in_flight());
    }

    /// `VICTORY_FLIGHT1_PREWARM`: while a flight is held on the host GPU
    /// (its queue waits on a fence of ours), the next submission's
    /// preparation takes the pages the title wrote of a copy the flight
    /// only reads, and leaves the copy the flight declared it writes alone;
    /// once the flight lands, the next one's own refresh does that one.
    #[test]
    fn prewarm_takes_only_copies_the_flight_cannot_write() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        let page = crate::watch::PAGE;
        let [read, written] = [0, 1].map(|_| {
            let p = crate::watch::tests::pool(1);
            crate::shadow::register(p, p as u64, page as u64);
            let copy = crate::shadow::find(p as u64, page as u64).expect("a host copy");
            copy.update(p, page);
            (p, copy)
        });
        // Flight N reads one copy and declares that its shaders write the
        // other.
        let n = Arc::new(host.new_list().expect("host list"));
        read.1.refresh(&n, read.0 as u64, 256);
        written.1.refresh(&n, written.0 as u64, 256);
        written.1.gpu_writes(&n, written.0 as u64, 256, false);
        n.close();
        // The GPU holds it until the gate opens.
        let mut gate = Gate::hold(host);
        host.execute_flight1(vec![n], vec![]);
        assert!(host.has_in_flight());
        // The title writes both copies' memory meanwhile.
        for (p, _) in [&read, &written] {
            // SAFETY: bytes of our pools.
            unsafe { (*p as *mut u32).write_volatile(0x7177_0001) };
        }
        let m = Arc::new(host.new_list().expect("host list"));
        read.1.refresh(&m, read.0 as u64, 256);
        written.1.refresh(&m, written.0 as u64, 256);
        m.close();
        let opener = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            gate.open();
        });
        let before = early_counts();
        TEST_PREWARM.with(|p| p.set(true));
        // Prewarms beside flight N, then waits for the gate and lands it.
        host.execute_flight1(vec![m], vec![]);
        TEST_PREWARM.with(|p| p.set(false));
        opener.join().unwrap();
        host.finish_flight_now();
        let after = early_counts();
        let d: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        // (Unless another test's submission held this thread off the
        // submission lock past the gate's opening: then the GPU may have
        // run the flight before the go, and nothing is said.)
        if d[0] == 1 && d[2] == 0 {
            assert!(
                d[3] >= 1,
                "the copy the flight only reads was taken early: {d:?}"
            );
            assert!(
                d[5] >= 1,
                "the copy the flight writes was left alone: {d:?}"
            );
        }
        for (p, copy) in [&read, &written] {
            assert_eq!(
                copy.read_host(*p as u64, 1)[0],
                0x7177_0001,
                "the next flight read the title's write"
            );
            crate::shadow::unregister(*p, *p as u64);
        }
    }

    /// `VICTORY_FLIGHTS=2`: a submission whose shaders mark the pages they
    /// write does not go beside the flight in the air (the bitmap is taken
    /// whole when that one lands); one that only reads goes beside it, and
    /// both land in order.
    #[test]
    fn write_marks_keep_a_submission_from_going_beside() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        TEST_FLIGHTS.with(|n| n.set(2));
        let page = crate::watch::PAGE;
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, page as u64);
        let copy = crate::shadow::find(p as u64, page as u64).expect("a host copy");
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let list = |name: &'static str, marked: Option<bool>| {
            let l = Arc::new(host.new_list().expect("host list"));
            copy.refresh(&l, p as u64, 256);
            if let Some(marked) = marked {
                copy.gpu_writes(&l, p as u64, 256, marked);
            }
            let log = order.clone();
            l.defer_after(move || log.lock().unwrap().push(name));
            l.close();
            l
        };
        let mut gate = Gate::hold(host);
        let before_beside = beside_counts()[0];
        host.execute_flight1(vec![list("first", None)], vec![]);
        // Reads only: beside the first.
        host.execute_flight1(vec![list("reads", None)], vec![]);
        let (in_air, after_reads) = (host.in_air(), beside_counts()[0]);
        if gate.held() {
            assert_eq!(in_air, 2, "two flights in the air");
            assert_eq!(after_reads, before_beside + 1);
        }
        // Marks its writes: the two in the air land first (the gate opens
        // meanwhile).
        let opener = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            gate.open();
        });
        host.execute_flight1(vec![list("marks", Some(true))], vec![]);
        opener.join().unwrap();
        assert_eq!(beside_counts()[0], after_reads, "it never goes beside");
        host.finish_flight_now();
        TEST_FLIGHTS.with(|n| n.set(0));
        assert_eq!(*order.lock().unwrap(), ["first", "reads", "marks"]);
        crate::shadow::unregister(p, p as u64);
    }

    /// `Host::quiet()` drains any active Flight-1 work cleanly before
    /// returning `Quiet` (preserving D3DMetal Issue #31's GPU-idle guarantee
    /// around shader-visible heap null-fill) with zero deadlock.
    #[test]
    fn flight1_quiet_waits_for_in_flight_before_heap_creation() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);

        let mut bop_word: u32 = 0;
        let seg = Arc::new(host.new_list().expect("host list"));
        seg.defer_write((&raw mut bop_word).cast(), Written::Value(99), 4);
        seg.close();

        host.execute_flight1(vec![seg], vec![]);
        // `quiet()` locks `submitted` and finishes any in-flight work.
        // Force a direct `finish_flight_now` check via `quiet()` (note: off
        // Wine `quiet()` returns None, so also test `finish_flight_now`
        // directly when off Wine).
        let q = host.quiet();
        if q.is_some() {
            assert!(
                !host.has_in_flight(),
                "Host::quiet() returned while a GPU flight was still active"
            );
            // SAFETY: flight has landed before `quiet()` returned.
            assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 99);
            drop(q);
        } else {
            host.finish_flight_now();
            assert!(!host.has_in_flight());
            // SAFETY: flight has landed after `finish_flight_now()`.
            assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 99);
        }
    }

    /// Reproduces the 2-submission shadow overlap:
    /// Flight 1 reads `OLD` (`0x11223300`) from a `Shadow` into `out1`, then
    /// the CPU immediately mutates title memory to `NEW` (`0xAABBCC00`) and
    /// submits Flight 2 reading the same `Shadow` into `out2`.
    /// Because Flight-1 enforces Wait-Before-Prep, Flight 1 MUST observe
    /// `OLD` and Flight 2 MUST observe `NEW` (never `NEW, NEW`) across 10
    /// consecutive iterations.
    #[test]
    fn flight1_never_clobbers_in_flight_shadow_data() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        // One flight in the air, whatever `VICTORY_FLIGHTS` says: with two,
        // the next submission is prepared beside the flight, and its
        // refresh of a copy the flight only reads takes the title's newer
        // bytes there (`Host::beside_refusal`), as the prewarm does: a
        // title rewriting what a submission it gave still reads races the
        // GPU on the console too.
        TEST_FLIGHTS.with(|n| n.set(1));

        // Allocate a page-aligned title buffer so Shadow tracks it cleanly.
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: valid layout; freed after all flights finish.
        let title_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let title_addr = title_ptr as u64;
        let shadow = crate::shadow::find(title_addr, 4096).expect("shadow");

        let (out1_buf, out1_ptr) = host.mapped_buffer(64, true).expect("out1");
        let (out2_buf, out2_ptr) = host.mapped_buffer(64, true).expect("out2");

        for iter in 0..10u32 {
            let old_val = 0x1122_3300u32 | iter;
            let new_val = 0xAABB_CC00u32 | iter;
            // SAFETY: valid 4096-byte title buffer and 64-byte readback buffers.
            unsafe {
                title_ptr.cast::<u32>().write_volatile(old_val);
                out1_ptr.cast::<u32>().write_volatile(0);
                out2_ptr.cast::<u32>().write_volatile(0);
            }

            let seg1 = Arc::new(host.new_list().expect("seg1"));
            shadow.refresh(&seg1, title_addr, 64);
            // SAFETY: open list, valid buffers.
            unsafe {
                seg1.list.CopyBufferRegion(
                    &out1_buf,
                    0,
                    &shadow.buffer,
                    shadow.offset(title_addr),
                    4,
                );
            }
            seg1.keep(&out1_buf);
            seg1.close();

            // Launch Flight 1 asynchronously!
            host.execute_flight1(vec![seg1], vec![]);

            // Immediately overwrite title memory with NEW while Flight 1 is
            // still in flight (exactly the #273 overlap window).
            // SAFETY: valid title buffer.
            unsafe {
                title_ptr.cast::<u32>().write_volatile(new_val);
            }

            let seg2 = Arc::new(host.new_list().expect("seg2"));
            shadow.refresh(&seg2, title_addr, 64);
            // SAFETY: open list, valid buffers.
            unsafe {
                seg2.list.CopyBufferRegion(
                    &out2_buf,
                    0,
                    &shadow.buffer,
                    shadow.offset(title_addr),
                    4,
                );
            }
            seg2.keep(&out2_buf);
            seg2.close();

            // Flight 2 MUST wait for Flight 1 before refreshing `shadow`.
            host.execute_flight1(vec![seg2], vec![]);
            host.finish_flight_now();

            // SAFETY: both flights have landed.
            let got1 = unsafe { out1_ptr.cast::<u32>().read_volatile() };
            // SAFETY: both flights have landed.
            let got2 = unsafe { out2_ptr.cast::<u32>().read_volatile() };
            assert_eq!(
                (got1, got2),
                (old_val, new_val),
                "iter {iter}: Shadow.data clobbered in flight! got ({got1:#x}, {got2:#x}), expected ({old_val:#x}, {new_val:#x})"
            );
        }

        // SAFETY: all flights landed; free title buffer.
        unsafe { std::alloc::dealloc(title_ptr, layout) };
        TEST_FLIGHTS.with(|n| n.set(0));
    }

    /// Under `VICTORY_FLIGHT1=1` + `VICTORY_HEAP_WORKER=1`, `bindless::Uploads`
    /// keeps both the upload list and the worker `ReadyDrain` alive across the
    /// flight until the GPU fence completes, then calls `release_worker()` and
    /// returns the upload list to the reuse pool so subsequent submissions see
    /// the uploaded texture and cleanly reset the dirty upload list.
    #[test]
    fn flight1_uploads_releases_heap_worker_drain_after_fence() {
        use crate::compute::{Param, srv, uav};
        use windows::core::s;

        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        let _worker = crate::heap_worker::EnableForTest::new();

        const PAGE_BYTES: usize = 64 * 1024;
        const LINEAR: u32 = 1;
        const SHADER: &str = r#"
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t0);
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, texel);
}
"#;

        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 16, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let slot_at = |i: u64| title_heap.start + 32 * i;
        let heap_host = title_heap.host().expect("host view heap");

        let pool = crate::watch::tests::pool(1);
        let addr_a = pool as u64;
        // SAFETY: dedicated pool page.
        unsafe {
            std::ptr::write_bytes(addr_a as *mut u8, 0x44, PAGE_BYTES);
        }
        let placed_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, LINEAR, 0, 0];
        let mut desc_a = [0u32; 16];
        (desc_a[0], desc_a[1], desc_a[2], desc_a[5]) = (28, 4, 0x1688, 1);
        (desc_a[10], desc_a[11]) = (addr_a as u32, (addr_a >> 32) as u32);

        let dst_desc: [u32; 14] = [1, 0, 0, 0, 256, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let props = [1u32, 0, 0, 0, 0];
        let res_iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut dst = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                props.as_ptr().cast(),
                0,
                dst_desc.as_ptr().cast(),
                0,
                std::ptr::null(),
                &res_iid,
                &mut dst,
            ),
            S_OK
        );
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let uav_desc: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav_desc, slot_at(0));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_a, slot_at(1));

        _worker.step_heap(title_heap);
        assert!(crate::heap_worker::has_ready());

        let (root, pso) =
            crate::compute::pipeline(SHADER, s!("flight_worker"), 0, &[Param::Table(&[uav(0, 1), srv(0, 1)])])
                .unwrap();
        let seg = Arc::new(host.new_list().unwrap());
        // SAFETY: open list, live objects.
        unsafe {
            seg.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            seg.list.SetComputeRootSignature(&root);
            seg.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            seg.list.SetPipelineState(&pso);
            seg.list.Dispatch(1, 1, 1);
        }
        seg.defer_heap(&heap_host.viewed);
        seg.close();

        let (q, f) = make_queue_and_fence();
        host.execute_flight1(vec![seg], vec![]);
        queue_signal(q, f, 1);

        assert!(
            wait_fence(f, 1, Duration::from_secs(5)),
            "victory-flight did not land worker upload flight"
        );
        assert!(!host.has_in_flight());
        assert!(heap_host.kept(1), "slot 1 published at flight submission");
        // Not asserted: `has_ready` is false only until the flight's kick of the
        // heap worker finds more to stage, which depends on timing. The
        // read-back below shows the staged upload ran.
        let out = crate::shadow::find(dst_bytes as u64, 256).unwrap();
        assert_eq!(out.read_host(dst_bytes as u64, 4), [0x44, 0x44, 0x44, 0x44]);

        // A follow-up empty flight exercises resetting the returned dirty upload list.
        let seg2 = Arc::new(host.new_list().unwrap());
        seg2.close();
        host.execute_flight1(vec![seg2], vec![]);
        host.finish_flight_now();
        assert!(!host.has_in_flight());

        crate::object::release_ref(heap);
        crate::object::release_ref(dst);
        crate::watch::tests::free(pool);
    }

    /// Under `VICTORY_ASYNC_SUBMIT=1` + `VICTORY_FLIGHT1=1`, submits
    /// multiple lists with immediate title `Reset` (exercising `list::restart`
    /// early-reset splitting), interleaved `Signal`s, and `drain_submit`
    /// (as called by `PresentX`), verifying strict FIFO completion order.
    #[test]
    fn async_submit_drains_cleanly_at_present_and_preserves_order() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, true);

        let (q, f) = make_queue_and_fence();
        let (alloc, list) = make_title_list();

        let mut words = [0u64; 6];
        for (i, slot_word) in words.iter_mut().enumerate() {
            if i > 0 {
                reset_title_list(list, alloc);
            }
            let seg = crate::list::host_list(list).expect("host list");
            let val = (i as u64 + 1) * 100;
            seg.defer_write((slot_word as *mut u64).cast(), Written::Value(val), 8);
            close_title_list(list);
            queue_execute(q, &[list]);
            queue_signal(q, f, i as u64 + 1);
        }

        // Simulates PresentX draining `victory-submit` + any active flight.
        crate::queue::drain_submit(crate::queue::Why::Present);

        assert_eq!(crate::fence::value(f), 6);
        assert!(!host.has_in_flight());
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(w, (i as u64 + 1) * 100, "submission {i} BoP mismatch");
        }
    }
}
