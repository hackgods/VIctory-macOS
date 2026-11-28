//! `VICTORY_SUB_TIMELINE=1` (off by default): one log line per host
//! submission, for the dependency graph between consecutive submissions.
//!
//! Each line gives when the submission was asked for (microseconds since the
//! first one), how long the calling thread was away since the last
//! submission returned (the title's own time between two), the steps of its
//! preparation, the host call, the wait for the GPU, the copy-back, and what
//! it has to do with the submission before it: of the bytes its refreshes
//! copied from title memory, how many lie in 64 KiB pages the last
//! submission's shaders marked as written, in ranges it declared it writes,
//! and in host copies shaders can write at all; and the title copies whose
//! source the last submission wrote. Those are the bytes a preparation
//! running beside the last submission's GPU work would have to leave until
//! it has landed.
//!
//! Nothing here changes what runs: the hooks return at once when the switch
//! is off, and with it on they only count.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Whether submissions are logged (`VICTORY_SUB_TIMELINE=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SUB_TIMELINE").is_some_and(|v| v == "1");
        if on {
            tracing::info!("VICTORY_SUB_TIMELINE=1: one timeline line per host submission");
        }
        on
    })
}

/// A submission's steps, stamped as each ends.
#[derive(Clone, Copy)]
pub(crate) enum At {
    /// The submission lock taken.
    Locked,
    /// Eviction and view remakes done.
    Evicted,
    /// The heaps' views synced.
    Heaps,
    /// Texture uploads staged.
    Uploads,
    /// The lists' steps run (title copies, refreshes asked, snapshots).
    Steps,
    /// The last refreshes flushed: prepared.
    Prepared,
    /// The GPU has run it.
    Ran,
    /// Copy-back and the work after it done.
    Back,
}
const STAMPS: usize = At::Back as usize + 1;

#[derive(Default)]
struct Run {
    asked: Option<Instant>,
    stamps: [u64; STAMPS],
    lists: usize,
    /// Microseconds in the host's `ExecuteCommandLists` and `Signal`.
    call_us: u64,
    /// Refreshes flushed: ranges, bytes asked, bytes copied.
    refreshes: [u64; 3],
    /// Bytes copied (at most) in pages the last submission marked, in
    /// ranges it declared, and in copies shaders can write.
    copied_in: [u64; 3],
    /// Title copies and fills: how many, their bytes, and the bytes whose
    /// source the last submission marked or declared.
    title_copies: [u64; 3],
    /// What this one writes: pages its shaders marked, ranges it declared.
    marked: Vec<(usize, usize)>,
    declared: Vec<(usize, usize)>,
    /// Bottom-of-pipe writes, work after it, texture read-backs.
    owed: [usize; 3],
}

#[derive(Default)]
struct State {
    epoch: Option<Instant>,
    run: Option<Run>,
    /// The last submission's writes, sorted and joined.
    marked: Vec<(usize, usize)>,
    declared: Vec<(usize, usize)>,
    /// When the last submission returned, and on which thread.
    left: Option<(Instant, u32)>,
    number: u64,
}

static STATE: Mutex<State> = Mutex::new(State {
    epoch: None,
    run: None,
    marked: Vec::new(),
    declared: Vec::new(),
    left: None,
    number: 0,
});
/// Whether a run is being counted (the hooks' quick way out).
static RUNNING: AtomicBool = AtomicBool::new(false);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn with(f: impl FnOnce(&mut Run, &State)) {
    if !RUNNING.load(Relaxed) {
        return;
    }
    let mut state = state();
    if let Some(mut run) = state.run.take() {
        f(&mut run, &state);
        state.run = Some(run);
    }
}

/// Sorted and joined.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
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

/// The bytes of `[from, to)` inside the sorted, joined `ranges`.
fn inside(ranges: &[(usize, usize)], from: usize, to: usize) -> usize {
    let first = ranges.partition_point(|r| r.1 <= from);
    ranges[first..]
        .iter()
        .take_while(|r| r.0 < to)
        .map(|r| r.1.min(to) - r.0.max(from))
        .sum()
}

/// A submission was asked for (before the submission lock).
pub(super) fn asked() -> Option<Instant> {
    on().then(Instant::now)
}

/// The submission asked for at `asked` has the lock.
pub(super) fn begin(asked: Option<Instant>, lists: usize) {
    let Some(asked) = asked else { return };
    let mut state = state();
    state.epoch.get_or_insert(asked);
    let mut run = Run {
        asked: Some(asked),
        lists,
        ..Run::default()
    };
    run.stamps[At::Locked as usize] = asked.elapsed().as_micros() as u64;
    state.run = Some(run);
    RUNNING.store(true, Relaxed);
}

/// A step of the running submission has ended.
pub(super) fn stamp(at: At) {
    with(|run, _| {
        if let Some(asked) = run.asked {
            run.stamps[at as usize] = asked.elapsed().as_micros() as u64;
        }
    });
}

/// The host call took `us`.
pub(super) fn called(us: u64) {
    with(|run, _| run.call_us += us);
}

/// A refresh of `[from, to)` copied `copied` bytes; `writable`: shaders can
/// write the copy through views.
pub(super) fn refreshed(from: usize, to: usize, copied: usize, writable: bool) {
    with(|run, state| {
        run.refreshes[0] += 1;
        run.refreshes[1] += (to - from) as u64;
        run.refreshes[2] += copied as u64;
        run.copied_in[0] += inside(&state.marked, from, to).min(copied) as u64;
        run.copied_in[1] += inside(&state.declared, from, to).min(copied) as u64;
        if writable {
            run.copied_in[2] += copied as u64;
        }
    });
}

/// A title copy or fill of `len` bytes from `src`.
pub(super) fn title_copy(src: usize, len: usize) {
    with(|run, state| {
        run.title_copies[0] += 1;
        run.title_copies[1] += len as u64;
        let to = src.saturating_add(len);
        let hit = inside(&state.marked, src, to).max(inside(&state.declared, src, to));
        run.title_copies[2] += hit as u64;
    });
}

/// The running submission's shaders marked the page at `page`.
pub(crate) fn marked(page: usize) {
    with(|run, _| run.marked.push((page, page + crate::watch::PAGE)));
}

/// The running submission declared that its shaders write `[address,
/// address + len)`, which goes back.
pub(super) fn declared(address: usize, len: usize) {
    with(|run, _| run.declared.push((address, address + len)));
}

/// What the running submission owes the title once it has run:
/// bottom-of-pipe writes, work after it, texture read-backs.
pub(super) fn owed(writes: usize, after: usize, texture_backs: usize) {
    with(|run, _| {
        run.owed[0] += writes;
        run.owed[1] += after;
        run.owed[2] += texture_backs;
    });
}

/// The submission returns: its line is logged, and what it wrote kept for
/// the next one.
pub(super) fn end() {
    if !RUNNING.swap(false, Relaxed) {
        return;
    }
    let mut state = state();
    let Some(run) = state.run.take() else { return };
    let (Some(asked), Some(epoch)) = (run.asked, state.epoch) else {
        return;
    };
    let now = Instant::now();
    // SAFETY: a plain query.
    let thread = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    let (away, same) = state.left.map_or((0, true), |(left, by)| {
        (asked.saturating_duration_since(left).as_micros() as u64, by == thread)
    });
    state.number += 1;
    let s = run.stamps;
    let step = |a: At, b: At| s[b as usize].saturating_sub(s[a as usize]);
    let marked = joined(run.marked);
    let declared = joined(run.declared);
    let bytes = |r: &[(usize, usize)]| r.iter().map(|r| r.1 - r.0).sum::<usize>();
    tracing::info!(
        target: "victory_gpu::host::timeline",
        "sub n={} frame={} thread={thread} same_thread={} lists={} t_us={} away_us={} lock_us={} evict_us={} heaps_us={} \
         uploads_us={} steps_us={} flush_us={} gpu_us={} call_us={} back_us={} tail_us={} refreshes={} asked_bytes={} \
         copied_bytes={} copied_in_prev_marked={} copied_in_prev_declared={} copied_in_writable={} title_copies={} \
         title_copy_bytes={} title_copy_from_prev={} marked_bytes={} declared_bytes={} bop_writes={} after={} \
         texture_backs={}",
        state.number,
        crate::queue::presented(),
        u8::from(same),
        run.lists,
        asked.saturating_duration_since(epoch).as_micros(),
        away,
        s[At::Locked as usize],
        step(At::Locked, At::Evicted),
        step(At::Evicted, At::Heaps),
        step(At::Heaps, At::Uploads),
        step(At::Uploads, At::Steps),
        step(At::Steps, At::Prepared),
        step(At::Prepared, At::Ran),
        run.call_us,
        step(At::Ran, At::Back),
        (now.saturating_duration_since(asked).as_micros() as u64).saturating_sub(s[At::Back as usize]),
        run.refreshes[0],
        run.refreshes[1],
        run.refreshes[2],
        run.copied_in[0],
        run.copied_in[1],
        run.copied_in[2],
        run.title_copies[0],
        run.title_copies[1],
        run.title_copies[2],
        bytes(&marked),
        bytes(&declared),
        run.owed[0],
        run.owed[1],
        run.owed[2],
    );
    state.marked = marked;
    state.declared = declared;
    state.left = Some((now, thread));
}

#[cfg(test)]
mod tests {
    use super::{inside, joined};

    #[test]
    fn ranges_join_and_count_the_bytes_inside() {
        let ranges = joined(vec![(40, 50), (0, 10), (10, 20), (15, 30)]);
        assert_eq!(ranges, [(0, 30), (40, 50)]);
        assert_eq!(inside(&ranges, 0, 100), 40);
        assert_eq!(inside(&ranges, 25, 45), 10);
        assert_eq!(inside(&ranges, 30, 40), 0);
        assert_eq!(inside(&ranges, 50, 60), 0);
        assert_eq!(inside(&[], 0, 10), 0);
    }
}
