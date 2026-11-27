//! mac: `VICTORY_FLICKER_WATCH=1` (off by default): log lines for objects
//! that go black for one frame while their shape is right (notes/flicker.md).
//! Two ways a draw reads nothing for one submission are told:
//!
//! - **Heap slots read while null.** A view heap slot of a texture the title
//!   placed itself is left null when a heap sync finds it rewritten
//!   (`HostHeap::mirror`, the views "made when a draw reads them"), and a
//!   slot whose texture is given up is left null too (`HostHeap::evict`).
//!   Such a slot is made only after the submission whose shaders read it
//!   through the heap (`crate::descriptor::make_read`), so those shaders
//!   read zeros for that one submission. Each slot left null is noted here
//!   with why ([`Why`]); after a submission whose shaders read some, one line
//!   says how many, why, and how many viewed a placed texture whose host
//!   texture was made already (those needed no null frame at all), with the
//!   first few slots.
//! - **Lists past the indirect budget.** The clamps of indirect records
//!   (`indirect-clamp.hlsl`) give each host list 2^27 vertex instances (mesh
//!   groups cost 128): the first record past it and every one after it run
//!   empty. After a submission in which a list's budget word is full, one
//!   line says how many lists.
//!
//! The lines carry the presented frame, which the per-frame `copy order this
//! frame` lines also carry, so a frame in a screen recording can be matched
//! to them by counting frames. Nothing changes in what is drawn.

// The callers are the Windows build's; the native build runs the tests.
#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Whether the watch is on: `VICTORY_FLICKER_WATCH=1`, read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = rule(std::env::var("VICTORY_FLICKER_WATCH").ok().as_deref());
        if on {
            tracing::info!(
                "VICTORY_FLICKER_WATCH=1: a line after each submission whose shaders read heap slots left null \
                 (placed textures rewritten or given up), and after each in which a list ran past the indirect budget"
            );
        }
        on
    })
}

/// `VICTORY_FLICKER_WATCH`'s rule: on only for `1`.
fn rule(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// Why a heap slot was left null until a shader reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// A heap sync found the slot rewritten with a view of a placed texture
    /// while it held a host view (it showed something until then).
    Shown,
    /// The same, for a slot that held no host view.
    Fresh,
    /// Its texture was given up (`crate::resource::evict`).
    GivenUp,
}

impl Why {
    fn name(self) -> &'static str {
        match self {
            Why::Shown => "rewritten while shown",
            Why::Fresh => "first view",
            Why::GivenUp => "given up",
        }
    }
}

/// Per slot, why it was last left null (taken when a shader reads it).
static WHY: Mutex<Option<HashMap<u32, Why>>> = Mutex::new(None);

/// Slot `slot` was left null, until a shader reads it, for `why`.
pub(crate) fn nulled(slot: u32, why: Why) {
    if on() {
        note(&WHY, slot, why);
    }
}

fn note(map: &Mutex<Option<HashMap<u32, Why>>>, slot: u32, why: Why) {
    map.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(slot, why);
}

/// Why slot `slot` was left null, if it was since it was last taken.
pub(crate) fn take(slot: u32) -> Option<Why> {
    take_from(&WHY, slot)
}

fn take_from(map: &Mutex<Option<HashMap<u32, Why>>>, slot: u32) -> Option<Why> {
    map.lock().unwrap_or_else(|e| e.into_inner()).as_mut()?.remove(&slot)
}

/// What a slot read while null viewed, when it is a placed texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    pub(crate) address: u64,
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) mips: u16,
    pub(crate) format: u32,
    /// Its host texture was made before the shaders read the slot: the
    /// slot could have been made when the heap sync found it.
    pub(crate) made_already: bool,
}

/// A slot that a submission's shaders read while it was null.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub(crate) slot: u32,
    /// Why it was null; none if it was not told (null since before the
    /// watch, or left null another way).
    pub(crate) why: Option<Why>,
    pub(crate) placed: Option<Placed>,
}

/// The counts of one submission's slots read while null.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    pub(crate) null: usize,
    pub(crate) shown: usize,
    pub(crate) fresh: usize,
    pub(crate) given_up: usize,
    pub(crate) untold: usize,
    pub(crate) made_already: usize,
}

pub(crate) fn tally(reads: &[Read]) -> Tally {
    let mut t = Tally { null: reads.len(), ..Tally::default() };
    for r in reads {
        match r.why {
            Some(Why::Shown) => t.shown += 1,
            Some(Why::Fresh) => t.fresh += 1,
            Some(Why::GivenUp) => t.given_up += 1,
            None => t.untold += 1,
        }
        t.made_already += usize::from(r.placed.is_some_and(|p| p.made_already));
    }
    t
}

/// Slots named one by one in a line; the rest are only counted.
pub(crate) const NAMED: usize = 6;

/// The line for one submission: `first` slots its shaders read for the
/// first time since they were last made or left null, of which `reads` were
/// null; none when no slot was null.
pub(crate) fn reads_line(frame: u64, first: usize, reads: &[Read]) -> Option<String> {
    if reads.is_empty() {
        return None;
    }
    let t = tally(reads);
    let mut line = format!(
        "flicker watch: heap slots shaders read while null in the submission just run (they read zeros there): \
         frame={frame} null={} of {first} first reads; rewritten while shown {}, first view {}, given up {}, untold {}; \
         host texture made already {}",
        t.null, t.shown, t.fresh, t.given_up, t.untold, t.made_already
    );
    // Slots that showed something first: those are the ones seen to flicker.
    let mut named: Vec<&Read> = reads.iter().collect();
    named.sort_by_key(|r| (r.why != Some(Why::Shown), r.slot));
    for r in named.iter().take(NAMED) {
        line.push_str(&format!("; slot {} {}", r.slot, r.why.map_or("untold", Why::name)));
        if let Some(p) = r.placed {
            line.push_str(&format!(
                " {:#x} {}x{} mips {} format {}{}",
                p.address,
                p.width,
                p.height,
                p.mips,
                p.format,
                if p.made_already { " made already" } else { "" }
            ));
        }
    }
    if reads.len() > NAMED {
        line.push_str(&format!("; {} more", reads.len() - NAMED));
    }
    Some(line)
}

/// After a submission: logs its slots read while null, if any.
pub(crate) fn log_reads(frame: u64, first: usize, reads: &[Read]) {
    if let Some(line) = reads_line(frame, first, reads) {
        tracing::warn!("{line}");
    }
}

/// A host list's indirect-work budget (`indirect-clamp.hlsl`'s `BUDGET`).
pub(crate) const BUDGET: u32 = 1 << 27;

/// Whether a budget word read after its list ran says that records ran
/// empty: the clamp writes the whole budget once a record does not fit.
pub(crate) fn past_budget(word: u32) -> bool {
    word >= BUDGET
}

/// The line for one submission whose lists left budget words `words`
/// (one a list with a budget of its own); none when no list ran past it.
pub(crate) fn budget_line(frame: u64, words: &[u32]) -> Option<String> {
    let past = words.iter().filter(|&&w| past_budget(w)).count();
    if past == 0 {
        return None;
    }
    let most_under = words.iter().copied().filter(|&w| !past_budget(w)).max().unwrap_or(0);
    Some(format!(
        "flicker watch: host lists whose indirect records ran past the budget of {BUDGET} vertex instances \
         in the submission just run (their records from the first that did not fit ran empty): \
         frame={frame} lists={past} of {} with a budget; most spent by another {most_under}",
        words.len()
    ))
}

/// After a submission: logs its lists past the indirect budget, if any.
pub(crate) fn log_budgets(frame: u64, words: &[u32]) {
    if let Some(line) = budget_line(frame, words) {
        tracing::warn!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_watch_is_off_unless_it_is_1() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("true"), Some("2")] {
            assert!(!rule(off), "{off:?}");
        }
        assert!(rule(Some("1")));
        assert!(rule(Some(" 1 ")));
    }

    #[test]
    fn a_slot_left_null_is_told_once_with_its_last_reason() {
        let map = Mutex::new(None);
        assert_eq!(take_from(&map, 7), None, "nothing noted yet");
        note(&map, 7, Why::Fresh);
        note(&map, 7, Why::Shown);
        note(&map, 9, Why::GivenUp);
        assert_eq!(take_from(&map, 7), Some(Why::Shown), "the last reason");
        assert_eq!(take_from(&map, 7), None, "taken once");
        assert_eq!(take_from(&map, 9), Some(Why::GivenUp));
    }

    fn placed(made_already: bool) -> Option<Placed> {
        Some(Placed { address: 0x7059_0e00_00, width: 512, height: 256, mips: 10, format: 71, made_already })
    }

    #[test]
    fn a_submission_with_no_null_reads_has_no_line() {
        assert_eq!(reads_line(3, 40, &[]), None);
    }

    #[test]
    fn null_reads_are_counted_by_reason_and_the_shown_ones_named_first() {
        let reads = [
            Read { slot: 30, why: Some(Why::Fresh), placed: placed(false) },
            Read { slot: 50, why: Some(Why::Shown), placed: placed(true) },
            Read { slot: 10, why: Some(Why::GivenUp), placed: None },
            Read { slot: 20, why: None, placed: placed(true) },
            Read { slot: 40, why: Some(Why::Shown), placed: placed(false) },
        ];
        assert_eq!(
            tally(&reads),
            Tally { null: 5, shown: 2, fresh: 1, given_up: 1, untold: 1, made_already: 2 }
        );
        let line = reads_line(1413, 77, &reads).unwrap();
        assert!(line.contains("frame=1413 null=5 of 77 first reads"), "{line}");
        assert!(line.contains("rewritten while shown 2, first view 1, given up 1, untold 1"), "{line}");
        assert!(line.contains("host texture made already 2"), "{line}");
        let first = line.find("; slot 40 rewritten while shown").expect("slot 40 named");
        let second = line.find("; slot 50 rewritten while shown 0x70590e0000 512x256 mips 10 format 71 made already");
        let fresh = line.find("; slot 30 first view").expect("slot 30 named");
        assert!(second.is_some_and(|s| first < s && s < fresh), "shown slots first, by slot: {line}");
        assert!(line.contains("; slot 10 given up"), "{line}");
        assert!(!line.contains("more"), "all five named: {line}");
    }

    #[test]
    fn past_the_named_slots_the_rest_are_counted() {
        let reads: Vec<Read> = (0..NAMED as u32 + 3).map(|slot| Read { slot, why: None, placed: None }).collect();
        let line = reads_line(0, reads.len(), &reads).unwrap();
        assert_eq!(line.matches("; slot ").count(), NAMED);
        assert!(line.ends_with("; 3 more"), "{line}");
    }

    #[test]
    fn a_full_budget_word_is_a_list_past_the_budget() {
        assert!(!past_budget(0));
        assert!(!past_budget(BUDGET - 1), "spent all but one: every record fitted");
        assert!(past_budget(BUDGET), "what the clamp writes once a record does not fit");
        assert_eq!(budget_line(5, &[0, 12, BUDGET - 1]), None);
        let line = budget_line(5, &[BUDGET, 12, BUDGET, 90]).unwrap();
        assert!(line.contains("frame=5 lists=2 of 4 with a budget; most spent by another 90"), "{line}");
    }
}
