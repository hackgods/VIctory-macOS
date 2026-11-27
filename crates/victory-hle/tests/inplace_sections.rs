//! Title memory in place on XMem physical pages (`VICTORY_INPLACE_SECTIONS=1`):
//! a section on a D3DMetal heap shows the heap's pages at every address the
//! title maps it at, and title memory is found in place where views next to
//! each other show its pages in order. A test binary of its own, so that the
//! switch is set before anything reads it. Under Wine with D3DMetal (Rosetta)
//! only; elsewhere it says so and passes.

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

fn in_place(at: *mut c_void, len: usize) -> bool {
    victory_gpu::title_memory_in_place(at as u64, len as u64)
}

#[test]
fn views_of_a_section_on_a_heap_hold_title_memory_in_place() {
    // SAFETY: the test's only thread so far; nothing has read it yet.
    unsafe { std::env::set_var("VICTORY_INPLACE_SECTIONS", "1") };
    if !victory_gpu::in_place_sections() {
        eprintln!("no title memory in place here (not Rosetta with D3DMetal): skipped");
        return;
    }
    let (mut pages, mut n) = ([0usize; 4], 4);
    assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
    let reserve = || {
        XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        )
    };
    let (a, b) = (reserve(), reserve());
    assert!(!a.is_null() && !b.is_null());

    assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
    assert!(in_place(a.wrapping_byte_add(100), 3 * PAGE));
    assert!(
        !in_place(a.wrapping_byte_add(100), 4 * PAGE),
        "past the last view"
    );

    // The same pages at another address show the same memory, which the
    // heap holds: a write through one view is read through the other.
    assert_eq!(XMemMapPhysicalPages(b, 4, pages.as_ptr()), b);
    assert!(in_place(b, 4 * PAGE));
    // SAFETY: both mapped read-write, 4 pages each.
    unsafe {
        a.cast::<u32>()
            .add(PAGE / 4 + 5)
            .write_volatile(0xfeed_f00d);
        assert_eq!(
            b.cast::<u32>().add(PAGE / 4 + 5).read_volatile(),
            0xfeed_f00d
        );
    }

    // Pages out of order break a run; each page alone is still in place.
    let swapped = [pages[0], pages[2], pages[1], pages[3]];
    assert_eq!(XMemMapPhysicalPages(b, 4, swapped.as_ptr()), b);
    assert!(!in_place(b, 4 * PAGE));
    assert!(in_place(b, PAGE));
    assert!(in_place(b.wrapping_byte_add(PAGE), PAGE));
    // SAFETY: as above; b's second page now shows the third.
    unsafe {
        a.cast::<u32>()
            .add(2 * PAGE / 4)
            .write_volatile(0x0bad_cafe);
        assert_eq!(b.cast::<u32>().add(PAGE / 4).read_volatile(), 0x0bad_cafe);
    }

    // An unmapped page breaks the run around it.
    assert_eq!(
        XMemMapPhysicalPages(a.wrapping_byte_add(PAGE), 1, std::ptr::null()),
        a.wrapping_byte_add(PAGE)
    );
    assert!(!in_place(a, 2 * PAGE));
    assert!(in_place(a.wrapping_byte_add(2 * PAGE), 2 * PAGE));

    for at in [a, b] {
        XMemMapPhysicalPages(at, 4, std::ptr::null());
    }
    XMemFreePhysicalPages(pages.len(), pages.as_ptr());
}
