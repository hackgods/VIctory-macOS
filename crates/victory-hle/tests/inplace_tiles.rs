//! Title memory in place through tile mappings (`VICTORY_INPLACE_TILES=1`,
//! with `VICTORY_INPLACE_SECTIONS=1`): a reservation of 64 KiB pages holds
//! title memory in place wherever its pages are shown from sections on
//! heaps, in any order, and follows the title's maps and unmaps. A test
//! binary of its own, so that the switches are set before anything reads
//! them. Under Wine with D3DMetal (Rosetta) only; elsewhere it says so and
//! passes.

#![cfg(windows)]

use std::ffi::c_void;

use victory_hle::modules::xmem::{
    XMemAllocatePhysicalPages, XMemFreePhysicalPages, XMemMapPhysicalPages, XMemVirtualAlloc,
};

const PAGE: usize = 64 << 10;
const MEM_RESERVE: u32 = 0x2000;
/// The console's 64 KiB pages (2 MiB pages with the 64 KiB bit).
const PAGES_64K: u32 = 0x2000_0000 | 0x40_0000;
const PAGE_READWRITE: u32 = 4;

fn tiled(at: *mut c_void, len: usize) -> bool {
    victory_gpu::title_memory_tiled(at as u64, len as u64)
}

#[test]
fn reservations_hold_title_memory_in_place_through_tiles() {
    // SAFETY: the test's only thread so far; nothing has read them yet.
    unsafe {
        std::env::set_var("VICTORY_INPLACE_SECTIONS", "1");
        std::env::set_var("VICTORY_INPLACE_TILES", "1");
    }
    if !victory_gpu::in_place_tiles() {
        eprintln!("no title memory in place here (not Rosetta with D3DMetal): skipped");
        return;
    }
    let (mut pages, mut n) = ([0usize; 4], 4);
    assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
    let a = XMemVirtualAlloc(
        std::ptr::null_mut(),
        4 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!a.is_null());

    // Out of order: no run of one section, in place through tiles all the
    // same.
    let shuffled = [pages[2], pages[0], pages[3], pages[1]];
    assert_eq!(XMemMapPhysicalPages(a, 4, shuffled.as_ptr()), a);
    assert!(tiled(a, 4 * PAGE));
    // SAFETY: mapped read-write.
    unsafe {
        a.cast::<u32>().write_volatile(0x1234_5678);
    }

    // A page unmapped: not in place over it, still elsewhere.
    let second = a.wrapping_byte_add(PAGE);
    assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
    assert!(!tiled(a, 2 * PAGE));
    assert!(tiled(a.wrapping_byte_add(2 * PAGE), 2 * PAGE));
    // Mapped again, with the page it showed: in place again (tiles follow).
    assert_eq!(XMemMapPhysicalPages(second, 1, [pages[0]].as_ptr()), second);
    assert!(tiled(a, 4 * PAGE), "tiles follow the page mapped again");
    // A page the title shows at `a + 3 * PAGE` too: shown twice, so left to
    // one GPU address per page (aliased tiles are not taken for coherent).
    assert_eq!(XMemMapPhysicalPages(second, 1, [pages[1]].as_ptr()), second);
    assert!(!tiled(a, 4 * PAGE), "a page shown twice");
    assert!(
        tiled(a.wrapping_byte_add(2 * PAGE), PAGE),
        "a page shown once"
    );
    // Shown once again: through tiles again.
    assert_eq!(XMemMapPhysicalPages(second, 1, [pages[0]].as_ptr()), second);
    assert!(tiled(a, 4 * PAGE), "every page shown once again");

    XMemMapPhysicalPages(a, 4, std::ptr::null());
    assert!(!tiled(a, PAGE));
    XMemFreePhysicalPages(pages.len(), pages.as_ptr());
}
