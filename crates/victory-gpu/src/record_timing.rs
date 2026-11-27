//! What recording and submitting host lists costs, by how long the lists
//! are (`VICTORY_RECORD_TIMING=1`; off by default). Recording: the time the
//! title's list calls that go through the draw state (`crate::list`'s
//! `with_state`: draws, dispatches, indirect executions, the state they set)
//! spend in VIctory and the host's own recording, per host list, summed by
//! the draws the list holds when it closes. A cost per command that grows
//! with the list's length is recording that grows with the square of it.
//! Submitting: the time in the host's `ExecuteCommandLists` calls, and the
//! longest call with the draws of the submission it ran.

use std::sync::atomic::{AtomicU64, Ordering};

/// Whether recording and submissions are timed (`VICTORY_RECORD_TIMING=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_RECORD_TIMING"))
}

/// The host lists' sizes the counts are kept by: fewer draws than each.
pub(crate) const BUCKETS: [u32; 5] = [100, 500, 1000, 2000, u32::MAX];

/// The bucket of a list of `draws` draws.
pub(crate) fn bucket(draws: u32) -> usize {
    BUCKETS.iter().position(|&below| draws < below).unwrap_or(BUCKETS.len() - 1)
}

/// Per bucket: lists, draws, commands, nanoseconds recording.
static LISTS: [[AtomicU64; 4]; BUCKETS.len()] = [const { [const { AtomicU64::new(0) }; 4] }; BUCKETS.len()];

/// A host list of `draws` draws closed after `commands` timed calls that
/// took `ns` nanoseconds.
pub(crate) fn closed(draws: u32, commands: u32, ns: u64) {
    if commands == 0 {
        return;
    }
    let b = &LISTS[bucket(draws)];
    b[0].fetch_add(1, Ordering::Relaxed);
    b[1].fetch_add(u64::from(draws), Ordering::Relaxed);
    b[2].fetch_add(u64::from(commands), Ordering::Relaxed);
    b[3].fetch_add(ns, Ordering::Relaxed);
}

/// Native submissions: calls, nanoseconds in them, the longest call and the
/// draws of its submission.
static CALLS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// An `ExecuteCommandLists` call took `ns` nanoseconds, in a submission of
/// `draws` draws.
pub(crate) fn submitted(ns: u64, draws: u64) {
    CALLS[0].fetch_add(1, Ordering::Relaxed);
    CALLS[1].fetch_add(ns, Ordering::Relaxed);
    if CALLS[2].fetch_max(ns, Ordering::Relaxed) < ns {
        CALLS[3].store(draws, Ordering::Relaxed);
    }
}

/// The frame line's report over `frames` frames, the counts taken.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let mut parts = Vec::new();
    for (i, b) in LISTS.iter().enumerate() {
        let [lists, draws, commands, ns] = b.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
        if lists == 0 {
            continue;
        }
        let below = BUCKETS[i];
        let size = if below == u32::MAX { format!(">={}", BUCKETS[i - 1]) } else { format!("<{below}") };
        parts.push(format!(
            "{size}: {:.1} lists, {:.0} draws, {:.0} calls, {:.2} ms ({:.2} us/call)",
            lists as f64 / frames,
            draws as f64 / frames,
            commands as f64 / frames,
            ns as f64 / 1e6 / frames,
            ns as f64 / 1e3 / commands.max(1) as f64,
        ));
    }
    let [calls, ns, longest, draws] = CALLS.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    tracing::info!(
        "recording per frame by host list draws: {}; ExecuteCommandLists {:.1} calls, {:.2} ms, \
         longest {:.2} ms (submission of {draws} draws)",
        parts.join("; "),
        calls as f64 / frames,
        ns as f64 / 1e6 / frames,
        longest as f64 / 1e6,
    );
}

#[cfg(test)]
mod tests {
    /// Lists go to the first bucket they have fewer draws than; the last
    /// takes the rest.
    #[test]
    fn lists_are_bucketed_by_draws() {
        use super::bucket;
        assert_eq!([0, 99, 100, 499, 500, 999, 1000, 1999, 2000, 50_000].map(bucket), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
        assert_eq!(bucket(u32::MAX), 4);
    }
}
