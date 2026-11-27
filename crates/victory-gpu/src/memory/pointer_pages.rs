//! Title memory shaders reach only through pointers gets copies
//! (`VICTORY_POINTER_PAGES=1`, off by default; `notes/rt-builder.md`,
//! section 7).
//!
//! The page table names the host copies that draws and dispatches made
//! through views (`crate::shadow`). Memory reached only by an address a
//! shader computes, such as the nodes of the ray-tracing structures, which
//! the builder and the tracers follow by pointer, has no copy: it reads 0
//! and takes no writes. Every ray then finds nothing. And as each such
//! read is a miss, every frame withholds what shaders wrote
//! (`super::withholding`). In the runs of 27 November with the builder on,
//! 5 to 11 pages and one or two whole 256 MiB regions a frame missed.
//!
//! Here each missed page that is committed title memory gets a copy
//! ([`crate::shadow::find`]: the title buffer holding it, else a loose
//! window), so the next submission's page table names it. A region with
//! no block is known by its number only (the shader marks the region
//! where it has no page entries): the first committed memory in it gets a
//! copy, so the region has a block, and later misses in it name their
//! pages. A page that is not committed (a stale or garbage pointer) stays
//! missing, as before.
//!
//! The idea is fxceless/victory-fork 578d459c and 9a9091d1 ("copies shaders
//! reached through pointers", from fault notes) and DexNom/VIctory
//! bc3cd4b0; their title memory has another layout, so the code is this
//! line's. As avatarmanz/VIctory-7900XTX b0ba5faa found, a bound on such
//! pages (2,048 there) is met within a minute of ray tracing: there is
//! none here.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// `VICTORY_POINTER_PAGES=1`: missed pages get copies. Unset or anything
/// else: off.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_POINTER_PAGES").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!(
                "VICTORY_POINTER_PAGES=1: title memory shaders reach by address with no copy gets one from the next submission on"
            );
        }
        on
    })
}

/// Tests: [`on`] whatever the variable says.
#[cfg(test)]
pub(crate) static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes of a page of the page table.
const PAGE: u64 = 1 << 16;
/// Bytes of a region of the page table's level 1.
const REGION: u64 = 1 << 28;
/// Below this nothing is mapped (the null page).
const LOWEST: u64 = 0x10000;

/// Copies made for missed pages and regions, and misses left (not
/// committed), since the last [`take`].
static MADE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// What a missed address is to [`learn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Missed {
    /// A page with no copy: its address.
    Page(u64),
    /// A 256 MiB region with no block: its first address.
    Region(u64),
}

/// The address to copy for a missed page or region, if any: the page
/// itself, or the first committed byte of the region (found by
/// `first_committed` over `[start, end)`); none below the null page's end.
pub(crate) fn target(missed: Missed, first_committed: impl Fn(u64, u64) -> Option<u64>) -> Option<u64> {
    match missed {
        Missed::Page(at) if at >= LOWEST => Some(at & !(PAGE - 1)),
        Missed::Page(_) => None,
        Missed::Region(at) => {
            let start = (at & !(REGION - 1)).max(LOWEST);
            first_committed(start, (at & !(REGION - 1)) + REGION)
        }
    }
}

/// After a list ran: copies for the pages and regions it missed (each once
/// per list; the copies' own lookups keep repeats cheap).
pub(crate) fn learn(missed: &[Missed]) {
    for &m in missed {
        let page = matches!(m, Missed::Page(_));
        let made = target(m, first_committed).filter(|&at| crate::texture::committed(at as *const u8, 1));
        match made.and_then(|at| crate::shadow::find(at, 1)) {
            Some(_) => {
                MADE[usize::from(!page)].fetch_add(1, Ordering::Relaxed);
            }
            None => {
                MADE[2].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Copies asked for since the last call: (for pages, for regions, left
/// missing).
pub(crate) fn take() -> [u64; 3] {
    MADE.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}

/// The first committed, readable address in `[start, end)`, by the
/// host's memory map.
fn first_committed(start: u64, end: u64) -> Option<u64> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = start;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return None;
        }
        let readable = info.Protect != 0 && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
        if info.State == MEM_COMMIT && readable {
            return Some(p.max(info.BaseAddress as u64));
        }
        let next = info.BaseAddress as u64 + info.RegionSize as u64;
        if next <= p {
            return None;
        }
        p = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missed page is copied at its page; one in the null page is not; a
    /// region is copied at its first committed byte, looked for from the
    /// region's start (past the null page in region 0) to its end.
    #[test]
    fn missed_pages_and_regions_name_what_to_copy() {
        let none = |_: u64, _: u64| None;
        assert_eq!(target(Missed::Page(0x9b2c4_1234), none), Some(0x9b2c4_0000));
        assert_eq!(target(Missed::Page(0x8000), none), None, "the null page");
        assert_eq!(target(Missed::Page(0), none), None);
        let asked = std::cell::Cell::new((0, 0));
        let at = |s: u64, e: u64| {
            asked.set((s, e));
            Some(s + 0x40_0000)
        };
        assert_eq!(target(Missed::Region(0x71_5000_0000), at), Some(0x71_5040_0000));
        assert_eq!(asked.get(), (0x71_5000_0000, 0x71_6000_0000));
        assert_eq!(target(Missed::Region(0), at), Some(0x10000 + 0x40_0000));
        assert_eq!(asked.get(), (0x10000, 0x1000_0000));
        assert_eq!(target(Missed::Region(0x71_5000_0000), none), None, "nothing committed");
    }

    /// The host's memory map: a committed allocation is found from below
    /// it, and nothing past it.
    #[test]
    fn the_first_committed_byte_is_found_by_the_memory_map() {
        let held = vec![1u8; 3 << 16];
        let at = held.as_ptr() as u64;
        assert_eq!(first_committed(at, at + 1), Some(at));
        let page = at.next_multiple_of(1 << 16);
        assert_eq!(first_committed(page, page + (1 << 16)), Some(page));
    }
}
