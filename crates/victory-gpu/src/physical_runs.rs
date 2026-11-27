//! mac: how much of the refreshed title memory lies in runs of consecutive
//! pages of one physical section (`VICTORY_PHYSICAL_RUNS=1`, a count only,
//! off by default): what aliasing the sections' pages (a `VICTORY_ALIAS_PHYSICAL`
//! not built yet, `notes/candidates-3.md`, "Physical-page mappings") could
//! take off the copies. `victory_hle`'s `xmem` tells each view of physical
//! pages it maps and unmaps; a refreshed host copy is counted by whether its
//! title buffer lies in one run of consecutive pages of one section (one view,
//! or views side by side that continue each other), in physical pages that are
//! not one run, or elsewhere.
//!
//! mac: also, always and whatever the switch, the counts that size stage 2
//! (`notes/alias-physical-design.md`, I0): the title's placeholder
//! reservations and their sizes, its physical sections, the pages its calls
//! map fresh, map over another page or unmap, and the pages shown at two or
//! more addresses ([`stage2_log`], a line of its own after the frame line).
//!
//! Plain std, so its tests run everywhere.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The 64 KiB page physical pages come in.
const PAGE: usize = 64 << 10;

/// Whether the count is on (`VICTORY_PHYSICAL_RUNS=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_RUNS"))
}

/// Views of physical pages by title address: their length, section and the
/// index of their first page in it.
static VIEWS: Mutex<BTreeMap<usize, (usize, usize, usize)>> = Mutex::new(BTreeMap::new());

/// `xmem` mapped `len` bytes of `section` from page `index` at `at`.
pub fn mapped(at: usize, len: usize, section: usize, index: usize) {
    VIEWS_NOW.fetch_add(1, Relaxed);
    if on() {
        VIEWS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(at, (len, section, index));
    }
}

/// `xmem` unmapped the view at `at`.
pub fn unmapped(at: usize) {
    VIEWS_NOW.fetch_sub(1, Relaxed);
    if on() {
        VIEWS.lock().unwrap_or_else(|e| e.into_inner()).remove(&at);
    }
}

/// Where `[from, to)` lies: 0 in one run of consecutive pages of one section,
/// 1 in physical pages that are not one run, 2 elsewhere.
fn class(views: &BTreeMap<usize, (usize, usize, usize)>, from: usize, to: usize) -> usize {
    let Some((&base, &(len, section, index))) = views
        .range(..=from)
        .next_back()
        .filter(|(b, v)| **b + v.0 > from)
    else {
        return 2;
    };
    let (mut end, mut next) = (base + len, index + len / PAGE);
    while end < to {
        match views.get(&end) {
            Some(&(len, s, i)) if s == section && i == next => {
                (end, next) = (end + len, next + len / PAGE)
            }
            _ => return 1,
        }
    }
    0
}

/// Per class: copies refreshed, bytes copied, microseconds.
static COUNTS: [[AtomicU64; 3]; 3] = [const { [const { AtomicU64::new(0) }; 3] }; 3];

/// A refresh copied `bytes` into the host copy of `[start, start + len)` in
/// `us` microseconds.
pub(crate) fn refreshed(start: usize, len: usize, bytes: usize, us: u64) {
    let k = class(
        &VIEWS.lock().unwrap_or_else(|e| e.into_inner()),
        start,
        start + len,
    );
    for (i, n) in [1, bytes as u64, us].into_iter().enumerate() {
        COUNTS[k][i].fetch_add(n, Relaxed);
    }
}

/// The line logged with the frame line, per frame over `frames` frames; then
/// stage 2's counts ([`stage2_log`]), always.
pub(crate) fn log(frames: f64) {
    if on() {
        runs_log(frames);
    }
    stage2_log(frames);
}

/// The `physical runs` line (`VICTORY_PHYSICAL_RUNS=1`).
fn runs_log(frames: f64) {
    let [run, scattered, other] = COUNTS.each_ref().map(|c| {
        c.each_ref()
            .map(|n| n.swap(0, Relaxed) as f64 / frames.max(1.0))
    });
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "physical runs (VICTORY_PHYSICAL_RUNS), per frame: refreshes of copies in one run of one section's pages \
         {:.0} copies, {:.1} MB in {:.1} ms; in physical pages not one run {:.0} copies, {:.1} MB in {:.1} ms; \
         elsewhere {:.0} copies, {:.1} MB in {:.1} ms; {views} views mapped",
        run[0],
        run[1] / 1e6,
        run[2] / 1e3,
        scattered[0],
        scattered[1] / 1e6,
        scattered[2] / 1e3,
        other[0],
        other[1] / 1e6,
        other[2] / 1e3,
    );
}

// mac: stage 2's counts (`notes/alias-physical-design.md`, I0), always on:
// atomics and, per reservation, one entry. `xmem` tells them under its own
// lock, from what it already looks up.

/// 4 GiB: the largest piece of a reservation one reserved buffer covers in
/// stage 2 (root views and clamp batches take 32-bit offsets).
pub(crate) const PIECE: usize = 1 << 32;

/// What [`STAGE2`] holds.
#[derive(Clone, Copy)]
enum S {
    /// Since the last line: pages of the title's calls (`XMemMapPhysicalPages`,
    /// its scatter form, `CopyPageMappingsBatchX`) mapped where no page was,
    /// mapped over another page, unmapped, and mapped again over themselves.
    Fresh,
    Remapped,
    Unmapped,
    Again,
}

static STAGE2: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The same since the start.
static STAGE2_TOTAL: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Now: views of physical pages, pages shown at two or more addresses, the
/// addresses they are shown at past the first, sections and their bytes.
static VIEWS_NOW: AtomicI64 = AtomicI64::new(0);
static TWICE: [AtomicI64; 2] = [const { AtomicI64::new(0) }; 2];
static SECTIONS: [AtomicI64; 2] = [const { AtomicI64::new(0) }; 2];

/// Placeholder reservations of 64 KiB pages: their start and length, in the
/// order made.
static RESERVATIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// What one title call did to the pages of its range, from the page (any key
/// that tells pages apart) each page address showed before the call and shows
/// after it: `[fresh, remapped, unmapped, again]`.
pub fn tally<K: PartialEq>(pairs: impl IntoIterator<Item = (Option<K>, Option<K>)>) -> [usize; 4] {
    let mut n = [0; 4];
    for (before, after) in pairs {
        let k = match (before, after) {
            (None, Some(_)) => S::Fresh,
            (Some(b), Some(a)) if b == a => S::Again,
            (Some(_), Some(_)) => S::Remapped,
            (Some(_), None) => S::Unmapped,
            (None, None) => continue,
        };
        n[k as usize] += 1;
    }
    n
}

/// Adds one call's [`tally`].
pub fn pages_changed(n: [usize; 4]) {
    for (i, n) in n.into_iter().enumerate() {
        STAGE2[i].fetch_add(n as u64, Relaxed);
        STAGE2_TOTAL[i].fetch_add(n as u64, Relaxed);
    }
}

/// How a page's addresses going from `before` to `after` in number changes
/// [pages shown at two or more addresses, addresses past the first].
pub fn shown_delta(before: usize, after: usize) -> [i64; 2] {
    let twice = |n: usize| i64::from(n >= 2);
    let extra = |n: usize| n.saturating_sub(1) as i64;
    [twice(after) - twice(before), extra(after) - extra(before)]
}

/// Adds the [`shown_delta`]s of one map or unmap.
pub fn shown_changed(delta: [i64; 2]) {
    for (i, d) in delta.into_iter().enumerate() {
        if d != 0 {
            TWICE[i].fetch_add(d, Relaxed);
        }
    }
}

/// A placeholder reservation of 64 KiB pages at `[at, at + len)`, the memory
/// physical pages are mapped into. The first ones are logged, each.
pub fn reserved(at: usize, len: usize) {
    let mut all = RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
    all.push((at, len));
    let n = all.len();
    drop(all);
    if n <= 64 {
        tracing::info!(
            n,
            at = format_args!("{at:#x}"),
            len = format_args!("{len:#x}"),
            mb = len >> 20,
            below_4gib = at.saturating_add(len) <= 1 << 32,
            pieces = len.div_ceil(PIECE),
            "XMem placeholder reservation of 64 KiB pages"
        );
    }
}

/// A physical section of `len` bytes was made (`sign` 1) or freed (-1).
pub fn section(len: usize, sign: i64) {
    SECTIONS[0].fetch_add(sign, Relaxed);
    SECTIONS[1].fetch_add(sign * len as i64, Relaxed);
}

/// What the reservations add up to.
#[derive(Debug, Default, PartialEq, Eq)]
struct Reservations {
    n: usize,
    bytes: usize,
    largest: usize,
    /// Over 4 GiB: more than one piece.
    over: usize,
    /// Wholly below 4 GiB (32-bit memory).
    low: usize,
    /// Pieces of at most 4 GiB, from each reservation's start.
    pieces: usize,
}

fn reservations(all: &[(usize, usize)]) -> Reservations {
    let mut r = Reservations::default();
    for &(at, len) in all {
        r.n += 1;
        r.bytes += len;
        r.largest = r.largest.max(len);
        r.over += usize::from(len > PIECE);
        r.low += usize::from(at.saturating_add(len) <= 1 << 32);
        r.pieces += len.div_ceil(PIECE);
    }
    r
}

/// Stage 2's line, per frame over `frames` frames: `window` the page counts
/// since the last line, `total` since the start, `now` views, pages at two
/// or more addresses, addresses past the first, sections and their bytes.
fn stage2_line(
    window: [u64; 4],
    total: [u64; 4],
    now: [i64; 5],
    r: &Reservations,
    frames: f64,
) -> String {
    let [fresh, remapped, unmapped, again] = window.map(|n| n as f64 / frames.max(1.0));
    let [views, twice, extra, sections, section_bytes] = now;
    let mapped = total[S::Fresh as usize] as i64 - total[S::Unmapped as usize] as i64;
    format!(
        "physical pages, per frame: {fresh:.1} pages mapped where none was, {remapped:.1} mapped over another page, \
         {unmapped:.1} unmapped, {again:.1} mapped again over themselves; since the start {} fresh, {} remapped, {} \
         unmapped; now {mapped} page addresses mapped in {views} views, {twice} pages at two or more addresses ({extra} \
         addresses past the first); {} placeholder reservations {:.2} GB (largest {} MB, {} over 4 GiB, {} below 4 GiB, \
         {} pieces of at most 4 GiB); {sections} sections {:.2} GB",
        total[S::Fresh as usize],
        total[S::Remapped as usize],
        total[S::Unmapped as usize],
        r.n,
        r.bytes as f64 / 1e9,
        r.largest >> 20,
        r.over,
        r.low,
        r.pieces,
        section_bytes as f64 / 1e9,
    )
}

/// Logs [`stage2_line`] once the title has reserved or allocated anything.
fn stage2_log(frames: f64) {
    let r = reservations(&RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()));
    let sections = SECTIONS.each_ref().map(|n| n.load(Relaxed));
    let window = STAGE2.each_ref().map(|n| n.swap(0, Relaxed));
    if r.n == 0 && sections[0] == 0 && window == [0; 4] {
        return;
    }
    let total = STAGE2_TOTAL.each_ref().map(|n| n.load(Relaxed));
    let [twice, extra] = TWICE.each_ref().map(|n| n.load(Relaxed));
    let now = [
        VIEWS_NOW.load(Relaxed),
        twice,
        extra,
        sections[0],
        sections[1],
    ];
    tracing::info!("{}", stage2_line(window, total, now, &r, frames));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_is_counted_by_where_its_pages_are() {
        let a = 0x1_0000_0000;
        let mut views = BTreeMap::new();
        // Section 7, pages 10 and 11 at `a`, page 12 at `a + 2 pages`: one run.
        views.insert(a, (2 * PAGE, 7, 10));
        views.insert(a + 2 * PAGE, (PAGE, 7, 12));
        // Then page 40 of it, and a page of section 8.
        views.insert(a + 3 * PAGE, (PAGE, 7, 40));
        views.insert(a + 4 * PAGE, (PAGE, 8, 41));
        assert_eq!(
            class(&views, a + 100, a + 3 * PAGE),
            0,
            "three pages in order"
        );
        assert_eq!(class(&views, a + PAGE, a + PAGE + 8), 0, "inside one view");
        assert_eq!(
            class(&views, a, a + 4 * PAGE),
            1,
            "the run breaks at page 40"
        );
        assert_eq!(class(&views, a + 3 * PAGE, a + 5 * PAGE), 1, "two sections");
        assert_eq!(
            class(&views, a + 4 * PAGE, a + 6 * PAGE),
            1,
            "runs off the views"
        );
        assert_eq!(class(&views, a - PAGE, a + PAGE), 2, "starts outside");
        assert_eq!(class(&views, a + 9 * PAGE, a + 10 * PAGE), 2);
    }

    #[test]
    fn a_call_is_tallied_by_what_each_page_address_showed() {
        // Pages as (section, index): a map over two empty addresses, one
        // that showed another page and one that showed this page already.
        let map = [
            (None, Some((1, 0))),
            (None, Some((1, 1))),
            (Some((1, 9)), Some((1, 2))),
            (Some((1, 3)), Some((1, 3))),
        ];
        assert_eq!(tally(map), [2, 1, 0, 1]);
        // An unmap: of two mapped addresses and one that showed nothing.
        assert_eq!(
            tally([(Some((1, 0)), None), (Some((2, 0)), None), (None, None)]),
            [0, 0, 2, 0]
        );
        // The same page at a second address is fresh there.
        assert_eq!(
            tally([(None, Some((1, 0))), (Some((1, 0)), Some((1, 0)))]),
            [1, 0, 0, 1]
        );
        assert_eq!(
            tally(std::iter::empty::<(Option<u8>, Option<u8>)>()),
            [0; 4]
        );
    }

    #[test]
    fn a_page_shown_twice_is_counted_once_with_its_extra_addresses() {
        assert_eq!(shown_delta(0, 1), [0, 0], "first address");
        assert_eq!(shown_delta(1, 2), [1, 1], "second address");
        assert_eq!(shown_delta(2, 3), [0, 1], "third");
        assert_eq!(shown_delta(3, 2), [0, -1]);
        assert_eq!(shown_delta(2, 1), [-1, -1], "back to one");
        assert_eq!(shown_delta(1, 0), [0, 0]);
        assert_eq!(shown_delta(3, 0), [-1, -2], "every address at once");
        // Any sequence comes back to nothing.
        let path = [0, 1, 2, 3, 1, 4, 0];
        let sum = path
            .windows(2)
            .map(|w| shown_delta(w[0], w[1]))
            .fold([0, 0], |s, d| [s[0] + d[0], s[1] + d[1]]);
        assert_eq!(sum, [0, 0]);
    }

    #[test]
    fn reservations_add_up_to_pieces_of_at_most_4_gib() {
        const GIB: usize = 1 << 30;
        let all = [
            (0x1084_0000, 0xae0_0000),  // the 174 MiB heap below 4 GiB
            (0x1_0000_0000, 4 * GIB),   // exactly one piece
            (0x2_0000_0000, 9 * GIB),   // three: 4, 4 and 1 GiB
            (0xff_0000_0000, 64 << 10), // one page
        ];
        assert_eq!(
            reservations(&all),
            Reservations {
                n: 4,
                bytes: 0xae0_0000 + 13 * GIB + (64 << 10),
                largest: 9 * GIB,
                over: 1,
                low: 1,
                pieces: 6,
            }
        );
        // A reservation ending at 4 GiB is 32-bit memory; one past it is not.
        assert_eq!(reservations(&[((1 << 32) - GIB, GIB)]).low, 1);
        assert_eq!(reservations(&[((1 << 32) - GIB, GIB + (64 << 10))]).low, 0);
        assert_eq!(reservations(&[]), Reservations::default());
    }

    #[test]
    fn the_stage2_line_gives_per_frame_and_standing_counts() {
        let r = reservations(&[(0x1_0000_0000, 5 << 30)]);
        let line = stage2_line(
            [200, 20, 10, 4],
            [1000, 30, 100, 8],
            [12, 3, 4, 1, 6_550_000_000],
            &r,
            10.0,
        );
        assert!(line.starts_with("physical pages, per frame: 20.0 pages mapped where none was, 2.0 mapped over another page, 1.0 unmapped, 0.4 mapped again"), "{line}");
        assert!(
            line.contains("since the start 1000 fresh, 30 remapped, 100 unmapped"),
            "{line}"
        );
        assert!(line.contains("now 900 page addresses mapped in 12 views, 3 pages at two or more addresses (4 addresses past the first)"), "{line}");
        assert!(line.contains("1 placeholder reservations 5.37 GB (largest 5120 MB, 1 over 4 GiB, 0 below 4 GiB, 2 pieces of at most 4 GiB)"), "{line}");
        assert!(line.ends_with("; 1 sections 6.55 GB"), "{line}");
        // No frames (a flip of `VICTORY_AB` can end a window early): per frame as one.
        assert!(
            stage2_line([3, 0, 0, 0], [3, 0, 0, 0], [0; 5], &r, 0.0).contains(": 3.0 pages mapped")
        );
    }
}
