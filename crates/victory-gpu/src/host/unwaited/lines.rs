//! mac: E1 of `notes/overlap-stages-design.md`: lines only, for every run
//! with `VICTORY_ASYNC_SUBMIT=1` (flights or not), so that the arms of an
//! overlap A/B can be read side by side. Nothing counted here changes what
//! runs or when.
//!
//! - bottom-of-pipe writes made, and those into aliased memory (a later
//!   submission in the air could read them on the GPU before we write them);
//! - shader-write ranges finished, and those on aliased copies (stage 3's
//!   declarations that need no finish);
//! - per thread calling us: the time inside our queue calls, and the waits
//!   for the held-work lock and the submission lock in it
//!   (`crate::flights::threads`).

use super::*;

/// Since the last frame line: words made, into aliased memory; ranges
/// declared, on aliased copies.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// When the last line was made.
static SINCE: Mutex<Option<Instant>> = Mutex::new(None);

/// A bottom-of-pipe write of `bytes` at `at` was made.
pub(in crate::host) fn word(at: usize, bytes: usize) {
    COUNTS[0].fetch_add(1, Relaxed);
    if crate::shadow::find_aliased(at as u64, (at + bytes) as u64).is_some() {
        COUNTS[1].fetch_add(1, Relaxed);
    }
}

/// A finished submission declared that shaders write a range of `copy`.
pub(in crate::host) fn declared(copy: &crate::shadow::Shadow) {
    COUNTS[2].fetch_add(1, Relaxed);
    if copy.is_aliased() {
        COUNTS[3].fetch_add(1, Relaxed);
    }
}

/// The submission lock, its wait counted for the thread.
pub(super) fn lock_submitted(lock: &Mutex<u64>) -> std::sync::MutexGuard<'_, u64> {
    let at = Instant::now();
    let held = lock.lock().unwrap_or_else(|e| e.into_inner());
    crate::flights::threads::add(crate::flights::threads::Spent::Lock, at.elapsed().as_micros() as u64);
    held
}

/// mac: the waits for a submission left (`Host::wait_left`) since the last
/// line: waits, their microseconds, wakes by the fence event, wakes by the
/// 2 ms timeout with the fence still short, and wakes by the timeout that
/// found the fence reached (its event never came: up to 2 ms the GPU sat
/// idle before we saw it was done).
static WAITS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A wait for a submission left ended after `waited`.
pub(super) fn waited(waited: std::time::Duration) {
    WAITS[0].fetch_add(1, Relaxed);
    WAITS[1].fetch_add(waited.as_micros() as u64, Relaxed);
}

/// A wake in such a wait: by the event (`event`), and whether the fence
/// had reached the value then (`ran`).
pub(super) fn woke(event: bool, ran: bool) {
    let k = if event { 2 } else if ran { 4 } else { 3 };
    WAITS[k].fetch_add(1, Relaxed);
}

/// The line (`frames` since the last one).
pub(super) fn log(frames: f64) {
    let [words, aliased_words, ranges, aliased_ranges] = COUNTS.each_ref().map(|n| n.swap(0, Relaxed) as f64 / frames);
    let now = Instant::now();
    let since = SINCE.lock().unwrap_or_else(|e| e.into_inner()).replace(now);
    let frame = since.map_or(0.0, |s| (now - s).as_secs_f64() * 1000.0 / frames);
    let [waits, waited_us, by_event, short, late] = WAITS.each_ref().map(|n| n.swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "waits for a submission left: {waits:.1}/frame, {:.1} ms/frame; wakes: {by_event:.1} by the fence event, \
         {short:.1} by the 2 ms timeout with the fence short, {late:.1} by the timeout with the fence reached (event missed)",
        waited_us / 1000.0
    );
    let mut threads = crate::flights::threads::take();
    threads.sort_by_key(|t| std::cmp::Reverse(t.1[0]));
    let threads: Vec<String> = threads
        .iter()
        .map(|(id, us)| {
            let [inside, held, lock] = us.map(|u| u as f64 / 1000.0 / frames);
            format!("{id}: {inside:.1}/{held:.1}/{lock:.1}")
        })
        .collect();
    tracing::info!(
        "overlap lines: {words:.1}/frame bottom-of-pipe writes made, {aliased_words:.1} of them into aliased memory; \
         {ranges:.1}/frame shader-write ranges finished, {aliased_ranges:.1} of them on aliased copies; threads \
         calling us by Windows thread id, ms/frame inside our queue calls (ExecuteCommandLists, Signal, Wait, \
         PresentX) / waiting for the held-work lock / waiting for the submission lock: [{}]; {frame:.1} ms/frame by \
         the clock (outside our calls, a thread spent that less its first figure)",
        threads.join(", ")
    );
}
