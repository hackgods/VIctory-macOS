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
//! copies back GPU writes, runs `after` closures and bottom-of-pipe writes,
//! recycles its `HostList`s, and delivers any attached title fence signals.
//!
//! A background `victory-flight` thread waits on `flight_event` without
//! holding `HELD` or `Host::submitted`, then acquires `Host::submitted` to
//! land the completed flight so fence waiters and BoP pollers wake promptly
//! even when the title submits no further work.

use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::Ordering;

use std::sync::atomic::AtomicU64;

use super::{Batch, Holding, Host, HostList, Mark, Step};
use crate::object::This;

/// Whether the next submission's buffer copies are brought up to date beside
/// the flight in the air (`VICTORY_PREP_BESIDE=1`, with Flight-1 and without
/// the async submit worker; off by default).
///
/// Flight-1 lands the flight before it prepares the next submission, so the
/// whole preparation follows the copy-back. Most of it has nothing to do with
/// the flight: of the bytes a world frame's refreshes copy, about a tenth lie
/// in pages the submission before wrote. With the switch, the thread that
/// asks for a submission first refreshes what its lists ask for outside the
/// ranges the flight may write ([`InFlight::writes`]), while the GPU still
/// runs the flight; then it lands the flight itself and prepares as before,
/// and the refreshes it asks for again find those pages current.
///
/// A refresh there is the one the preparation makes later
/// ([`crate::shadow::Shadow::update`]: the pages the title wrote since the
/// copy took them, guarded again), only earlier, so a page the title writes
/// afterwards is copied again and the submission reads the same bytes as
/// without the switch. What the flight may write is left alone until its
/// copy-back: its declared ranges and those armed for it
/// ([`arm_undeclared`]), as whole 64 KiB pages. Descriptors, uploads, title
/// copies and snapshots are not moved.
///
/// Work held behind the flight's bottom-of-pipe writes is run by the thread
/// that asked for it too: it refreshes for the held lists, lands the flight
/// and goes on, with no wake-up of `victory-flight` in between.
pub(crate) fn prep_beside_enabled() -> bool {
    #[cfg(test)]
    match TEST_PREP_BESIDE.load(Ordering::Acquire) {
        1 => return false,
        2 => return flight1_enabled(),
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = flight1_enabled()
            && !async_submit_env()
            && std::env::var_os("VICTORY_PREP_BESIDE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "VICTORY_PREP_BESIDE=1: buffer copies the flight in the air cannot write are refreshed beside it"
            );
        }
        on
    })
}

#[cfg(test)]
static TEST_PREP_BESIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests: `VICTORY_PREP_BESIDE` on or off until dropped (under a
/// [`ModeGuard`], which serializes the tests that set modes).
#[cfg(test)]
pub(crate) struct BesideGuard(u8);

#[cfg(test)]
impl BesideGuard {
    pub(crate) fn new(on: bool) -> Self {
        Self(TEST_PREP_BESIDE.swap(if on { 2 } else { 1 }, Ordering::AcqRel))
    }
}

#[cfg(test)]
impl Drop for BesideGuard {
    fn drop(&mut self) {
        TEST_PREP_BESIDE.store(self.0, Ordering::Release);
    }
}

/// Since the last frame line, of [`Host::prepare_beside`]: `[0]` times it
/// ran beside a flight still on the GPU, `[1]` times the GPU had finished
/// first (nothing done), `[2]` bytes its refreshes copied, `[3]` bytes asked
/// for and left as the flight may write them, `[4]` microseconds, `[5]`
/// flights landed by the thread whose held work waited for them, `[6]`
/// microseconds landings waited for the GPU (with or without the switch).
static BESIDE: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// [`BESIDE`] since the start (tests).
static BESIDE_TOTAL: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn beside(i: usize, n: u64) {
    BESIDE[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    BESIDE_TOTAL[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// [`BESIDE_TOTAL`] (tests).
#[cfg(test)]
pub(crate) fn beside_counts() -> [u64; 7] {
    BESIDE_TOTAL.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// The frame line of `VICTORY_PREP_BESIDE`, per frame over `frames`: how
/// much of the frame's refreshes ran beside a flight and how much after its
/// landing (`refreshed`: all the bytes refreshes copied since the last line).
pub(super) fn log_beside(frames: f64, refreshed: u64) {
    if !flight1_enabled() {
        return;
    }
    let b = BESIDE
        .each_ref()
        .map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed));
    let per = |n: u64| n as f64 / frames.max(1.0);
    tracing::info!(
        "prepared beside a flight per frame (VICTORY_PREP_BESIDE={}): {:.1} goes beside a flight still on the GPU, \
         {:.1} found it finished; refreshes copied {:.2} MB beside it and {:.2} MB after its landing, left {:.2} MB \
         asked for in ranges it may write; {:.1} ms; {:.1} flights landed by the thread whose held work waited for them; \
         landings still waited {:.1} ms for the GPU",
        u8::from(prep_beside_enabled()),
        per(b[0]),
        per(b[1]),
        per(b[2]) / 1e6,
        per(refreshed.saturating_sub(b[2])) / 1e6,
        per(b[3]) / 1e6,
        per(b[4]) / 1e3,
        per(b[5]),
        per(b[6]) / 1e3,
    );
}

/// `ranges` sorted and joined, each widened to whole 64 KiB pages.
fn pages_joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let page = crate::watch::PAGE;
    for r in &mut ranges {
        *r = (r.0 / page * page, r.1.div_ceil(page).saturating_mul(page));
    }
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// The parts of `[from, to)` outside the sorted, joined `writes`.
fn outside(writes: &[(usize, usize)], from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = from;
    let first = writes.partition_point(|w| w.1 <= from);
    for &(a, b) in &writes[first..] {
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

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

/// Before a flight is given to the host GPU: the pages its shaders may
/// write with no list declaring them are armed (`Shadow::arm_undeclared`),
/// the unordered views of its heaps (the bindless path) and every copy the
/// page tables of its title-memory programs name. Their writes are marked
/// and go back by the pages marked (`Shadow::copy_back_marked`), and a page
/// the title wrote since the copy took it, and that nothing saved, keeps
/// the title's bytes whole, the shaders' there dropped. Waited for, the
/// title's thread that submits cannot write meanwhile; a flight lets it go
/// on, and it writes a frame's data into pages the GPU writes too (on the
/// console both land, byte by byte). Armed, the first title write to such
/// a page saves its bytes from before, and the copy-back keeps both. The
/// copies armed, to disarm when it lands.
fn arm_undeclared(
    segments: &[Arc<HostList>],
    heaps: &[Arc<crate::bindless::Viewed>],
) -> (Vec<Arc<crate::shadow::Shadow>>, Vec<(usize, usize)>) {
    let mut armed: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
    let mut ranges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut arm = |copy: &Arc<crate::shadow::Shadow>, address: usize, len: usize| {
        copy.arm_undeclared(address, len);
        ranges.push((address, address.saturating_add(len)));
        if seen.insert(Arc::as_ptr(copy) as usize) {
            armed.push(copy.clone());
        }
    };
    for heap in heaps {
        crate::bindless::written_ranges(heap, &mut arm);
    }
    for list in segments {
        for step in &list.deferred().copies {
            if let &Step::Memory(key) = step {
                crate::memory::copies(key, |copy| {
                    let (start, len) = copy.range();
                    arm(copy, start as usize, len as usize);
                });
            }
        }
    }
    (armed, ranges)
}

/// Title memory a flight of `segments` may write, as sorted, joined whole
/// pages: the ranges armed for it (`armed`, from [`arm_undeclared`]) and
/// those its lists declare their shaders write.
fn flight_writes(segments: &[Arc<HostList>], mut armed: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    for list in segments {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(_, address, len, _, _) = step {
                armed.push((*address, address.saturating_add(*len)));
            }
        }
    }
    pages_joined(armed)
}

/// Tests: the host queue waits on a fence of ours from [`Gate::hold`] to
/// [`Gate::open`] (or the drop): a flight given meanwhile stays in the air.
/// A watchdog opens it after 2 s: another test's submission behind the gate
/// holds the submission lock while it waits, and the thread that would open
/// it may be waiting for that lock.
#[cfg(test)]
pub(crate) struct Gate {
    fence: GateFence,
    opened: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
#[derive(Clone)]
struct GateFence(windows::Win32::Graphics::Direct3D12::ID3D12Fence);

// SAFETY: D3D12 fences are free-threaded.
#[cfg(test)]
unsafe impl Send for GateFence {}

#[cfg(test)]
impl GateFence {
    fn open(&self, opened: &std::sync::atomic::AtomicBool) {
        if !opened.swap(true, Ordering::AcqRel) {
            // SAFETY: our fence.
            unsafe { self.0.Signal(1) }.expect("the gate opens");
        }
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
        let (f, o) = (fence.clone(), opened.clone());
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            f.open(&o);
        });
        Gate { fence, opened }
    }

    pub(crate) fn open(&mut self) {
        self.fence.open(&self.opened);
    }

    /// Whether it still holds the queue (the watchdog has not opened it):
    /// what a test saw so far, it saw with the flight in the air.
    pub(crate) fn held(&self) -> bool {
        !self.opened.load(Ordering::Acquire)
    }
}

#[cfg(test)]
impl Drop for Gate {
    fn drop(&mut self) {
        self.fence.open(&self.opened);
    }
}

/// Protected by `Host::flight` (leaf lock strictly below `HELD` and
/// `Host::submitted`, above only `Fence::waiters`).
#[derive(Default)]
pub(super) struct FlightState {
    /// True from the moment a flight is launched until its post-GPU copy-back,
    /// BoP writes, and attached signals have all completed. While `true`,
    /// `attach_signal` queues behind the active flight instead of signalling
    /// immediately.
    active: bool,
    /// The launched flight awaiting wait + landing; taken once by whichever
    /// thread (`victory-flight` or the next submission) lands it under
    /// `Host::submitted`.
    in_flight: Option<InFlight>,
    /// Title fences queued behind the active flight (each kept alive via
    /// `fence::keep`). Delivered atomically before clearing `active`.
    pending_signals: Vec<(usize, u64)>,
    /// Whether the background `victory-flight` finisher thread has been spawned.
    worker_started: bool,
}

struct InFlight {
    fence_value: u64,
    segments: Vec<Arc<HostList>>,
    /// Copies whose pages it armed for its shaders' undeclared writes
    /// ([`arm_undeclared`]), disarmed once it has landed.
    armed: Vec<Arc<crate::shadow::Shadow>>,
    /// Title memory its shaders may write ([`flight_writes`]): what a
    /// preparation beside it leaves alone ([`Host::prepare_beside`]).
    writes: Vec<(usize, usize)>,
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
// and `Uploads` are `Send`; `Batch` holds `Arc<Shadow>` and ranges).
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
        // `VICTORY_PREP_BESIDE`: what these lists ask for and the flight in
        // the air cannot write is refreshed while the GPU still runs it.
        self.prepare_beside_locked(&segments);
        // Wait-Before-Prep: drain Flight N-1 completely before Flight N
        // touches any shared Shadow, descriptor heap, upload staging pool, or
        // Mark buffer.
        self.finish_flight_locked(&mut submitted, None, None);

        let prep = {
            let run_refs: Vec<&HostList> = segments.iter().map(|s| &**s).collect();
            self.prepare_submission(&submitted, &run_refs, &barrier_before, false)
        };

        let mut native_boundaries = prep.native_boundaries;
        let mark = self.mark();
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
        // Armed before the GPU may write: a title write from here on saves
        // its page.
        let (armed, armed_ranges) = arm_undeclared(&segments, &prep.heaps);
        let writes = flight_writes(&segments, armed_ranges);
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
        let flight = InFlight {
            fence_value,
            segments,
            armed,
            writes,
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
            debug_assert!(!state.active && state.in_flight.is_none());
            state.active = true;
            state.in_flight = Some(flight);
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
    pub(crate) fn execute_sync_segments(&self, segments: Vec<Arc<HostList>>, barrier_before: Vec<bool>) {
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
        if !state.active {
            return false;
        }
        if !already_kept {
            crate::fence::keep(fence);
        }
        state.pending_signals.push((fence as usize, value));
        true
    }

    /// Whether a flight is currently in the air or landing.
    pub(crate) fn has_in_flight(&self) -> bool {
        self.flight.lock().unwrap_or_else(|e| e.into_inner()).active
    }

    /// [`Host::prepare_beside_locked`] for lists still held behind the
    /// flight in the air (`crate::queue`), which the caller then lands.
    pub(crate) fn prepare_beside(&self, lists: &[Arc<HostList>]) {
        if !prep_beside_enabled() || lists.is_empty() || !self.has_in_flight() {
            return;
        }
        let _submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.prepare_beside_locked(lists);
    }

    /// The flight in the air was landed by the thread whose held work waited
    /// for it (the frame line).
    pub(crate) fn landed_for_held_work(&self) {
        beside(5, 1);
    }

    /// With `VICTORY_PREP_BESIDE=1` and a flight still on the GPU: brings up
    /// to date the buffer ranges `lists` ask for (their refreshes, table
    /// slots and heaps' views) outside what the flight may write
    /// ([`prep_beside_enabled`]). Called under `Host::submitted`, so the
    /// flight is not landed meanwhile.
    fn prepare_beside_locked(&self, lists: &[Arc<HostList>]) {
        if !prep_beside_enabled() {
            return;
        }
        let (fence_value, writes) = {
            let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            match state.in_flight.as_ref() {
                Some(f) if f.waited => (f.fence_value, f.writes.clone()),
                _ => return,
            }
        };
        // SAFETY: a plain query of our fence.
        if unsafe { self.fence.GetCompletedValue() } >= fence_value {
            beside(1, 1);
            return;
        }
        let started = std::time::Instant::now();
        let mut batch = Batch::default();
        let mut left = 0u64;
        let mut add = |copy: &Arc<crate::shadow::Shadow>, address: usize, len: usize| {
            // Memory that is not tracked is copied whole at every refresh:
            // done here it would be done again by the preparation.
            if copy.clean() || !copy.is_tracked() {
                return;
            }
            // Cut to the copy first: what is asked past its end is no byte
            // of it.
            let (start, bytes) = copy.range();
            let (from, to) = (
                address.max(start as usize),
                address.saturating_add(len).min((start + bytes) as usize),
            );
            if from >= to {
                return;
            }
            let mut ahead = 0;
            for (a, b) in outside(&writes, from, to) {
                batch.add(copy, a, b - a);
                ahead += b - a;
            }
            left += (to - from - ahead) as u64;
        };
        for list in lists {
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(copy, address, len) => add(copy, *address, *len),
                    Step::Slot(viewed, index) => crate::bindless::refresh_slot(viewed, *index, &mut add),
                    Step::Heap(viewed) => crate::bindless::refresh(viewed, &mut add),
                    // Title copies, fills, snapshots, page tables, textures
                    // and host work stay where they are.
                    Step::Copy(..)
                    | Step::Move(..)
                    | Step::Fill(..)
                    | Step::Host(_)
                    | Step::GpuWrite(..)
                    | Step::Texture(..)
                    | Step::Memory(_) => {}
                }
            }
        }
        let copied = batch.flush();
        beside(0, 1);
        beside(2, copied as u64);
        beside(3, left);
        beside(4, started.elapsed().as_micros() as u64);
    }

    /// Waits for and lands any active flight now. Returns `true` if a flight
    /// was landed by this call.
    pub(crate) fn finish_flight_now(&self) -> bool {
        if !self.has_in_flight() {
            return false;
        }
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.finish_flight_locked(&mut submitted, None, None)
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
            match state.in_flight.as_ref() {
                Some(f) if only_if_fence.is_none_or(|v| f.fence_value == v) => {
                    state.in_flight.take()
                }
                _ => return false,
            }
        };
        let Some(flight) = flight else {
            return false;
        };

        let gpu_done = if flight.waited {
            // SAFETY: plain fence completion query.
            if unsafe { self.fence.GetCompletedValue() } < flight.fence_value {
                let asked = std::time::Instant::now();
                self.wait_for(flight.fence_value);
                let now = std::time::Instant::now();
                // What a landing still waits for the GPU (the frame line of
                // `VICTORY_PREP_BESIDE`; reporting the wait left as upstream
                // #687 by AEMONET-Dev does for its prep-ahead).
                beside(6, (now - asked).as_micros() as u64);
                now
            } else {
                waited_at.unwrap_or_else(std::time::Instant::now)
            }
        } else {
            std::time::Instant::now()
        };
        let gpu_duration = gpu_done.saturating_duration_since(flight.launched);
        #[cfg(test)]
        super::tests::native_call_group(flight.waited);

        let mark = self.mark();
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
        if flight.waited
            && let Some(mark) = mark
            && (!mark.saw(flight.fence_value) || split_failed)
            // SAFETY: a plain query.
            && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
        {
            self.lost(submitted, mark);
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, flight.fence_value, 0);

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
            );
        }

        // Its copy-back is done: what it armed is as anything else again.
        for copy in &flight.armed {
            copy.disarm_undeclared();
        }
        // Return the upload list to `bindless::LIST` and recycle host lists
        // only after the GPU fence and post-GPU work have finished.
        drop(flight.uploads);
        flight.segments.into_iter().for_each(crate::list::recycle);

        // Deliver every title fence signal queued behind this flight
        // atomically before clearing `active`, so later `queue::signal` calls
        // can never overtake an attached signal.
        let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        for (fence, value) in state.pending_signals.drain(..) {
            crate::fence::set(fence as This, value);
            crate::fence::let_go(fence as This);
        }
        state.active = false;
        true
    }
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
                    while state.in_flight.is_none() {
                        state = host
                            .flight_wake
                            .wait(state)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                    let f = state.in_flight.as_ref().expect("in_flight present");
                    (f.fence_value, f.waited)
                };
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
                    .as_ref()
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

    /// `VICTORY_PREP_BESIDE`: a list held behind the flight in the air (it
    /// waits for the value the flight writes at its end) is run by the
    /// thread that submitted it, which lands the flight itself: when the
    /// submission returns, the flight has landed and the held list is in
    /// the air, with no landing by `victory-flight` in between.
    #[test]
    fn prep_beside_lands_the_flight_for_work_held_behind_it() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        let _beside = BesideGuard::new(true);
        let (q, f) = make_queue_and_fence();
        let mut bop_word: u64 = 0;
        let bop = (&raw mut bop_word) as usize;

        let mut gate = Gate::hold(host);
        let first = Arc::new(host.new_list().expect("host list"));
        first.defer_write(bop as *mut u8, Written::Value(0x1234_5678), 8);
        first.close();
        host.execute_flight1(vec![first], vec![]);
        assert!(host.has_in_flight());

        // The held list: it waits for the flight's bottom-of-pipe value.
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        let saw_value = Arc::new(AtomicBool::new(false));
        let saw = saw_value.clone();
        let second = Arc::new(host.new_list().expect("host list"));
        second.defer_wait(crate::hold::ValueWait {
            address: bop as u64,
            bytes: 8,
            comparison: 3,
            reference: 0x1234_5678,
            mask: u64::MAX,
        });
        second.defer_after(move || {
            flag.store(true, Ordering::Release);
            // SAFETY: the word outlives the flights.
            if unsafe { (bop as *const u64).read_volatile() } == 0x1234_5678 {
                saw.store(true, Ordering::Release);
            }
        });
        second.close();

        let before = beside_counts();
        // The gate opens while the submitting thread waits for the flight.
        let opener = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            gate.open();
        });
        crate::queue::submit_segment_for_test(q, second);
        opener.join().expect("the gate's thread");
        // The call returned: the first flight landed on this thread and the
        // held list was given to the GPU.
        // SAFETY: the flight has landed.
        assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 0x1234_5678);
        assert_eq!(
            beside_counts()[5],
            before[5] + 1,
            "the submitting thread landed the flight its held work waited for"
        );
        queue_signal(q, f, 3);
        assert!(wait_fence(f, 3, Duration::from_secs(5)), "the held list ran");
        assert!(ran.load(Ordering::Acquire), "its work after it ran");
        assert!(
            saw_value.load(Ordering::Acquire),
            "it ran after the flight's bottom-of-pipe write"
        );
        host.finish_flight_now();
        assert!(!host.has_in_flight());
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
