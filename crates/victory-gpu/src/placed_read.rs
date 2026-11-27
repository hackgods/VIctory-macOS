//! mac: `VICTORY_PLACED_AT_ONCE=read` (off by default; notes/flicker.md,
//! "Placed views for read slots"): which heap slots a heap sync makes at
//! once when it finds them rewritten with a view of a placed texture whose
//! host texture is not made yet.
//!
//! `=all` makes every such slot at once, and with it a host texture for
//! every texture the title places, read or not: that costs uploads and
//! memory, and past the card's budget the textures nobody read push out
//! ones that shaders do read (runs 27 Nov 13:17, 13:21: the slots read
//! while null were those given up). `=read` makes at once only the slots a
//! shader read through the heap in the last
//! `VICTORY_PLACED_READ_FRAMES` presented frames (default
//! [`DEFAULT_FRAMES`]), and leaves the others as `keep` does.
//!
//! The reads are the shaders' read marks (`victory_shader::emit`, "Read
//! marks": every texture read through the heap itself marks its index),
//! taken once a frame by the eviction pass (`crate::shadow::reads_since`)
//! and after each submission (`crate::shadow::take_reads`). A slot index
//! keeps its last read across the views written there: a slot the title
//! empties (releasing its texture, which a heap sync nulls) and fills again
//! counts as read. A slot whose index no shader read in the window is not
//! made at once: the first read of a slot new to the shaders still reads
//! zeros for one submission, as with `keep`.
//!
//! Plain std, so its tests run on any host.

#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// The default window: presented frames since a slot's last read within
/// which a rewrite of it is made at once.
pub(crate) const DEFAULT_FRAMES: u32 = 60;

/// `VICTORY_PLACED_READ_FRAMES`'s rule: a whole number above 0, else
/// [`DEFAULT_FRAMES`].
fn frames_rule(value: Option<&str>) -> u32 {
    value.and_then(|v| v.trim().parse::<u32>().ok()).filter(|&n| n > 0).unwrap_or(DEFAULT_FRAMES)
}

/// The window, `VICTORY_PLACED_READ_FRAMES`, read once.
pub(crate) fn frames() -> u32 {
    static FRAMES: OnceLock<u32> = OnceLock::new();
    *FRAMES.get_or_init(|| frames_rule(std::env::var("VICTORY_PLACED_READ_FRAMES").ok().as_deref()))
}

/// The selection rule: a slot last read at presented frame `last` (none if
/// never) is made at once at frame `now` if that read lies within `window`
/// frames. A read noted at a frame after `now` (another thread's) counts.
pub(crate) fn chosen(last: Option<u64>, now: u64, window: u32) -> bool {
    last.is_some_and(|read| now.saturating_sub(read) <= u64::from(window))
}

/// Per heap slot (an index into the read marks, below
/// `victory_shader::emit::HEAP_VIEWS`): the presented frame it was last
/// read at, plus one (0: never), and the frame plus one it was last made at
/// once at, until a read after it (0: none pending).
pub(crate) struct History {
    read: Box<[AtomicU32]>,
    made: Box<[AtomicU32]>,
}

/// A frame as kept in a [`History`]: plus one, so 0 is "none"; frames
/// past `u32::MAX - 1` stay there (years at any frame rate).
fn stored(frame: u64) -> u32 {
    frame.saturating_add(1).min(u64::from(u32::MAX)) as u32
}

impl History {
    /// A history of `slots` slots, none read.
    pub(crate) fn new(slots: usize) -> Self {
        let none = || (0..slots).map(|_| AtomicU32::new(0)).collect();
        History { read: none(), made: none() }
    }

    /// Shaders read `slots` by `frame`. Returns how many of them had been
    /// made at once since their last read (the rule's hits).
    pub(crate) fn note(&self, slots: &[u32], frame: u64) -> u64 {
        let at = stored(frame);
        let mut hits = 0;
        for &s in slots {
            let Some(read) = self.read.get(s as usize) else { continue };
            read.fetch_max(at, Relaxed);
            if self.made[s as usize].swap(0, Relaxed) != 0 {
                hits += 1;
            }
        }
        hits
    }

    /// The frame slot `slot` was last read at, if any.
    pub(crate) fn last_read(&self, slot: u32) -> Option<u64> {
        let at = self.read.get(slot as usize)?.load(Relaxed);
        (at != 0).then(|| u64::from(at) - 1)
    }

    /// Whether a rewrite of slot `slot` at frame `now` is made at once
    /// ([`chosen`] within `window`); if it is, it is noted as made at once
    /// (a later read of it is a hit).
    pub(crate) fn choose(&self, slot: u32, now: u64, window: u32) -> bool {
        let hot = chosen(self.last_read(slot), now, window);
        if hot {
            self.made[slot as usize].store(stored(now), Relaxed);
        }
        hot
    }
}

/// The counts for the frame line since the last line: slots made at once
/// by the rule, slots it left (as `keep` leaves them), reads of slots made
/// at once since, and placed-texture slots shaders read while null.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const MADE: usize = 0;
const SKIPPED: usize = 1;
const HITS: usize = 2;
const NULL: usize = 3;

/// Whether the history is kept: `VICTORY_PLACED_AT_ONCE=read` on side A,
/// or named for side B in `VICTORY_AB`. Read once.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let a = std::env::var("VICTORY_PLACED_AT_ONCE").is_ok_and(|v| v.trim() == "read");
        let b = std::env::var("VICTORY_AB").is_ok_and(|v| v.contains("VICTORY_PLACED_AT_ONCE=read"));
        a || b
    })
}

/// The history of the slots below `victory_shader::emit::HEAP_VIEWS` (the
/// read marks' slots), made on first use; none unless [`wanted`].
fn history() -> Option<&'static History> {
    static HISTORY: OnceLock<Option<History>> = OnceLock::new();
    HISTORY
        .get_or_init(|| {
            wanted().then(|| {
                let window = frames();
                tracing::info!(
                    window,
                    "VICTORY_PLACED_AT_ONCE=read: a heap slot rewritten with a view of a placed texture not made yet is made at once \
                     if a shader read the slot through the heap in the last VICTORY_PLACED_READ_FRAMES presented frames, else as keep"
                );
                History::new(victory_shader::emit::HEAP_VIEWS as usize)
            })
        })
        .as_ref()
}

/// Shaders read `slots` through the heap (their read marks), by the frame
/// presented now.
#[cfg(windows)]
pub(crate) fn note_reads(slots: &[u32]) {
    if let Some(h) = history() {
        let hits = h.note(slots, crate::queue::presented());
        COUNTS[HITS].fetch_add(hits, Relaxed);
    }
}

/// Whether a heap sync makes slot `index`, found rewritten with a view of a
/// placed texture not made yet, at once (`VICTORY_PLACED_AT_ONCE=read`).
/// Counted as made at once when it is.
#[cfg(windows)]
pub(crate) fn at_once(index: u32) -> bool {
    let Some(h) = history() else { return false };
    let hot = h.choose(index, crate::queue::presented(), frames());
    if hot {
        COUNTS[MADE].fetch_add(1, Relaxed);
    }
    hot
}

/// A slot the rule did not make at once was left as `keep` leaves it.
pub(crate) fn skipped() {
    COUNTS[SKIPPED].fetch_add(1, Relaxed);
}

/// A submission's shaders read `n` placed-texture slots while they were
/// null (`crate::descriptor::make_read`).
pub(crate) fn null_reads(n: usize) {
    if n > 0 && wanted() {
        COUNTS[NULL].fetch_add(n as u64, Relaxed);
    }
}

/// The frame line's part, per frame over `frames` frames; none unless the
/// history is kept. The counts are cleared.
pub(crate) fn line(frames: f64) -> Option<String> {
    if !wanted() {
        return None;
    }
    let [made, skipped, hits, null] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
    Some(text(frames, self::frames(), made, skipped, hits, null))
}

fn text(frames: f64, window: u32, made: u64, skipped: u64, hits: u64, null: u64) -> String {
    let per = |n: u64| n as f64 / frames.max(1.0);
    format!(
        "placed views for read slots (VICTORY_PLACED_AT_ONCE=read, window {window} frames): {:.1} slots/frame made at once \
         (read in the window), {:.1} left as keep leaves them; {:.1}/frame reads of slots made at once; \
         {:.1}/frame placed slots shaders read while null; totals {made} made, {skipped} left, {hits} read, {null} null",
        per(made),
        per(skipped),
        per(hits),
        per(null),
    )
}

/// Logs the frame line's part ([`line`]).
pub(crate) fn log(frames: f64) {
    if let Some(line) = line(frames) {
        tracing::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_a_whole_number_above_zero_else_the_default() {
        for other in [None, Some(""), Some("0"), Some("-3"), Some("x"), Some("1.5")] {
            assert_eq!(frames_rule(other), DEFAULT_FRAMES, "{other:?}");
        }
        assert_eq!(frames_rule(Some(" 12 ")), 12);
        assert_eq!(frames_rule(Some("1")), 1);
    }

    #[test]
    fn a_slot_is_chosen_only_when_read_within_the_window() {
        assert!(!chosen(None, 500, 60), "never read: left as keep leaves it");
        assert!(chosen(Some(500), 500, 60), "read this frame");
        assert!(chosen(Some(440), 500, 60), "read at the window's edge");
        assert!(!chosen(Some(439), 500, 60), "read one frame before it");
        assert!(chosen(Some(501), 500, 60), "read by another thread's later frame");
        assert!(chosen(Some(0), 0, 1), "frame 0 counts as a read");
    }

    #[test]
    fn a_read_is_kept_per_slot_index_across_the_views_written_there() {
        let h = History::new(64);
        assert_eq!(h.last_read(9), None);
        h.note(&[9, 12], 100);
        assert_eq!(h.last_read(9), Some(100));
        // The title empties slot 9 and fills it again 30 frames on: its
        // index was read, so the new view is made at once.
        assert!(h.choose(9, 130, 60));
        // Slot 10 was never read: left (a slot new to the shaders).
        assert!(!h.choose(10, 130, 60));
        // Slot 12, read long ago: left.
        assert!(!h.choose(12, 200, 60));
    }

    #[test]
    fn the_last_read_wins_and_an_older_note_does_not_set_it_back() {
        let h = History::new(8);
        h.note(&[3], 50);
        h.note(&[3], 40);
        assert_eq!(h.last_read(3), Some(50));
        h.note(&[3], 70);
        assert_eq!(h.last_read(3), Some(70));
    }

    #[test]
    fn slots_past_the_marks_are_never_read_and_never_chosen() {
        let h = History::new(16);
        assert_eq!(h.note(&[16, 1000], 5), 0);
        assert_eq!(h.last_read(16), None);
        assert!(!h.choose(16, 5, 60));
    }

    #[test]
    fn a_slot_made_at_once_and_then_read_is_one_hit() {
        let h = History::new(16);
        h.note(&[1, 2], 10);
        assert!(h.choose(1, 20, 60));
        assert!(h.choose(2, 20, 60));
        assert_eq!(h.note(&[1, 5], 21), 1, "slot 1 made at once and read");
        assert_eq!(h.note(&[1], 22), 0, "counted once");
        assert_eq!(h.note(&[2], 23), 1);
        assert!(!h.choose(5, 200, 60));
        assert_eq!(h.note(&[5], 201), 0, "not made at once: no hit");
    }

    #[test]
    fn many_threads_noting_and_choosing_keep_the_latest_read() {
        let h = std::sync::Arc::new(History::new(1024));
        std::thread::scope(|s| {
            for t in 0..4u64 {
                let h = h.clone();
                s.spawn(move || {
                    for frame in 0..200u64 {
                        let slots: Vec<u32> = (0..1024).filter(|i| i % 4 == t as u32).collect();
                        h.note(&slots, frame * 4 + t);
                        let _ = h.choose((frame % 1024) as u32, frame * 4, 8);
                    }
                });
            }
        });
        for i in 0..1024u32 {
            assert_eq!(h.last_read(i), Some(199 * 4 + u64::from(i % 4)), "slot {i}");
        }
    }

    #[test]
    fn the_line_gives_per_frame_counts_and_totals() {
        let line = text(4.0, 60, 40, 400, 30, 8);
        assert!(line.contains("window 60 frames"), "{line}");
        assert!(line.contains("10.0 slots/frame made at once"), "{line}");
        assert!(line.contains("100.0 left as keep leaves them"), "{line}");
        assert!(line.contains("7.5/frame reads of slots made at once"), "{line}");
        assert!(line.contains("2.0/frame placed slots shaders read while null"), "{line}");
        assert!(line.ends_with("totals 40 made, 400 left, 30 read, 8 null"), "{line}");
        assert!(text(0.0, 60, 3, 0, 0, 0).contains("3.0 slots/frame"), "no frames: counted as one");
    }

    #[test]
    fn the_history_is_kept_only_when_asked_for() {
        // The tests never set the switches.
        if std::env::var_os("VICTORY_PLACED_AT_ONCE").is_none() && std::env::var_os("VICTORY_AB").is_none() {
            assert!(!wanted());
            assert!(history().is_none());
            assert_eq!(line(1.0), None);
        }
    }
}
