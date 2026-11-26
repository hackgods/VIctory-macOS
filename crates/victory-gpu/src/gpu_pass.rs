//! Diagnostics, `VICTORY_GPU_PASS_TIMING=<k>`: where one world frame's GPU
//! time goes, by pass. D3DMetal answers timestamp queries with zeros, so
//! `VICTORY_GPU_TIMING` can time only whole host lists. For world frame `k`
//! (counted, as `VICTORY_CAPTURE_WORLD` counts, from the first frame that
//! ran 500 draws), each host list of the title lists recorded then (a pass,
//! mostly) runs and is waited for on its own: its wall time from
//! submission to the fence is its GPU time, plus a submission's own cost
//! (the line names the least any took). With `VICTORY_GPU_PASS_CUTS=<n>`,
//! up to `n` times the title list is also cut into a host list of its own
//! where it sets another pipeline. Two frames later, the line names the
//! costliest lists with their pipelines, draws and dispatches.
//!
//! Cutting a list makes a new host list carry its state again (as a wait
//! does, `crate::list`), and on Metal a render pass ends at each list's
//! end: passes drawing to one target pay their loads and stores again, so
//! the times are an upper bound, good for ranking.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// The world frame timed, counted from the world's first.
pub(crate) fn requested() -> Option<u64> {
    static FRAME: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *FRAME.get_or_init(|| {
        std::env::var("VICTORY_GPU_PASS_TIMING")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
}

/// Cuts left for the timed frame (`VICTORY_GPU_PASS_CUTS`, none by
/// default): each makes a host list of its own. 3,124 cuts in one world
/// frame (a cut at every pipeline set) made as many host lists, two
/// command allocators and lists each, and the process ran out of memory
/// (run v3-verify: Wine refused a 4 KiB allocation, then the title's crash
/// handler crashed).
static CUTS_LEFT: std::sync::OnceLock<AtomicU64> = std::sync::OnceLock::new();

fn cuts_left() -> &'static AtomicU64 {
    CUTS_LEFT.get_or_init(|| {
        AtomicU64::new(
            std::env::var("VICTORY_GPU_PASS_CUTS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0),
        )
    })
}

/// Whether one more cut may be made (and counts it).
pub(crate) fn take_cut() -> bool {
    cuts_left()
        .fetch_update(Relaxed, Relaxed, |n| n.checked_sub(1))
        .is_ok()
}

/// Sets the cuts left (tests).
#[cfg(test)]
pub(crate) fn allow_cuts(n: u64) {
    cuts_left().store(n, Relaxed);
}

/// The frame number timed, once the world's first frame is known.
fn timed_frame() -> Option<u64> {
    Some(crate::capture::world()? + requested()?)
}

/// Whether lists recorded now are timed (and cut, with cuts left): the
/// frame after the last presented is the timed one.
pub(crate) fn cutting() -> bool {
    timed_frame() == Some(crate::queue::presented() + 1)
}

/// One timed host list: GPU ms, submission, host list id, draws,
/// dispatches, pipelines.
type Run = (f64, u64, u64, u32, u32, String);

static RUNS: Mutex<Vec<Run>> = Mutex::new(Vec::new());
static LOGGED: AtomicU64 = AtomicU64::new(0);

/// A host list cut for timing ran in `wall`, in host submission
/// `submission`.
pub(crate) fn record(
    wall: Duration,
    submission: u64,
    list: u64,
    draws: u32,
    dispatches: u32,
    pipelines: String,
) {
    let ms = wall.as_secs_f64() * 1e3;
    RUNS.lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((ms, submission, list, draws, dispatches, pipelines));
}

/// Frame `n` was presented: two frames after the timed one, its runs are
/// logged (once).
pub(crate) fn presented(n: u64) {
    let Some(timed) = timed_frame() else {
        return;
    };
    if n < timed + 2 || LOGGED.swap(1, Relaxed) != 0 {
        return;
    }
    let runs = std::mem::take(&mut *RUNS.lock().unwrap_or_else(|e| e.into_inner()));
    for line in table(runs, 40) {
        tracing::info!(target: "victory_gpu::pass_timing", "{line}");
    }
}

/// The log lines for `runs`: a summary, then the `top` costliest.
fn table(mut runs: Vec<Run>, top: usize) -> Vec<String> {
    if runs.is_empty() {
        return vec![format!(
            "GPU pass timing: no host list was cut and timed in frame {:?}",
            timed_frame()
        )];
    }
    let total: f64 = runs.iter().map(|r| r.0).sum();
    let least = runs.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let (draws, dispatches) = runs.iter().fold((0u64, 0u64), |(d, c), r| {
        (d + u64::from(r.3), c + u64::from(r.4))
    });
    runs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut lines = vec![format!(
        "GPU pass timing, frame {:?}: {} host lists in {:.1} ms ({draws} draws, {dispatches} dispatches); \
         the least a list took, a submission's own cost: {least:.3} ms",
        timed_frame(),
        runs.len(),
        total
    )];
    for (rank, (ms, submission, list, draws, dispatches, pipelines)) in
        runs.iter().take(top).enumerate()
    {
        lines.push(format!(
            "  #{} {ms:.3} ms ({:.1}%): submission {submission}, host list {list}, {draws} draws, \
             {dispatches} dispatches, {}",
            rank + 1,
            100.0 * ms / total.max(f64::MIN_POSITIVE),
            if pipelines.is_empty() {
                "no pipeline"
            } else {
                pipelines
            }
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::table;

    /// The line names the lists costliest first, with their share of the
    /// frame, and the least any took.
    #[test]
    fn the_table_ranks_runs_by_time() {
        let runs = vec![
            (1.0, 7, 2, 3, 0, "[mesh aa, pixel bb]".to_owned()),
            (4.0, 7, 3, 0, 2, "[compute cc]".to_owned()),
            (0.5, 8, 4, 0, 0, String::new()),
        ];
        let lines = table(runs, 2);
        assert_eq!(lines.len(), 3);
        assert!(
            lines[0].contains("3 host lists in 5.5 ms (3 draws, 2 dispatches)"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("0.500 ms"), "{}", lines[0]);
        assert!(
            lines[1].starts_with("  #1 4.000 ms (72.7%): submission 7, host list 3"),
            "{}",
            lines[1]
        );
        assert!(lines[1].ends_with("[compute cc]"), "{}", lines[1]);
        assert!(lines[2].contains("#2 1.000 ms"), "{}", lines[2]);
    }
}
