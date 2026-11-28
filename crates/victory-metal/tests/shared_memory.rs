//! Title memory as Metal sees it through the bridge, with no copy (native
//! Metal host, milestone 1). The memory is made the way `victory-hle`'s
//! `xmem` makes the title's: a pagefile-backed section of 64 KiB physical
//! pages (`XMemAllocatePhysicalPages`), one whole view of it for our own
//! use, and the title's addresses as placeholders that single pages of the
//! section replace, one physical page at several addresses. One no-copy
//! buffer over the whole view is the GPU's only way to the pages (the
//! "canonical" buffer); the tests check every word both ways.
//!
//! tools/macos/test-wine.sh builds the bridge and sets it up; elsewhere
//! these tests say so and pass. The ignored ones are measurements:
//!
//! `tools/macos/test-wine.sh -p victory-metal --release --test shared_memory -- --ignored --nocapture`
#![cfg(windows)]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use victory_metal::{Bridge, Op, OpenError};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT,
    UnmapViewOfFile, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The title's physical page.
const PAGE: usize = 64 << 10;
const PAGE_WORDS: u32 = (PAGE / 4) as u32;

/// The bridge, or `None` (with the reason printed) where it cannot be.
fn bridge() -> Option<Bridge> {
    match Bridge::open() {
        Ok(b) => Some(b),
        Err(e @ (OpenError::NotWine | OpenError::NotAsked)) => {
            eprintln!("no Metal bridge here ({e}): skipped");
            None
        }
        Err(e) => panic!("the Metal bridge was asked for but did not open: {e}"),
    }
}

/// Title physical memory: a pagefile-backed section committed whole (as
/// `xmem` makes it under Wine), and one view of all of it.
struct Section {
    handle: HANDLE,
    view: *mut u8,
    size: usize,
}

impl Section {
    fn new(pages: usize) -> Self {
        let size = pages * PAGE;
        // SAFETY: plain section creation, checked.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        assert!(!handle.is_null(), "a section of {size} bytes");
        // SAFETY: our section, mapped whole.
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, size) }
            .Value
            .cast::<u8>();
        assert!(!view.is_null(), "the whole view of {size} bytes");
        Self { handle, view, size }
    }

    /// The words of physical page `page` through the whole view.
    // Mapped memory, not Rust-owned: the tests are single-threaded and never
    // hold two borrows of one page across a call that writes it.
    #[allow(clippy::mut_from_ref)]
    fn page(&self, page: usize) -> &mut [u32] {
        assert!((page + 1) * PAGE <= self.size);
        // SAFETY: inside the view, committed and read-write.
        unsafe { words(self.view.add(page * PAGE), PAGE) }
    }

    #[allow(clippy::mut_from_ref)]
    fn all(&self) -> &mut [u32] {
        // SAFETY: the whole view.
        unsafe { words(self.view, self.size) }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: the view and handle made in `new`.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.view.cast(),
            });
            CloseHandle(self.handle);
        }
    }
}

/// # Safety
/// `[at, at + bytes)` is committed, readable and writable memory nothing
/// else borrows meanwhile.
unsafe fn words<'a>(at: *mut u8, bytes: usize) -> &'a mut [u32] {
    // SAFETY: as the caller vouches.
    unsafe { std::slice::from_raw_parts_mut(at.cast(), bytes / 4) }
}

/// The title's addresses: `slots` placeholders of one physical page each,
/// which pages of a [`Section`] replace (as `xmem`'s `MapViewOfFile3` with
/// `MEM_REPLACE_PLACEHOLDER` does).
struct TitleRange {
    base: *mut u8,
    mapped: Vec<bool>,
}

impl TitleRange {
    fn new(slots: usize) -> Self {
        // SAFETY: a fresh placeholder reservation, checked.
        let base = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                slots * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        }
        .cast::<u8>();
        assert!(!base.is_null(), "a placeholder of {slots} pages");
        // One placeholder a page, so that each can be replaced alone.
        for i in 0..slots - 1 {
            // SAFETY: splits our own placeholder at a page boundary.
            let ok = unsafe {
                VirtualFree(
                    base.add(i * PAGE).cast(),
                    PAGE,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            };
            assert!(ok != 0, "splitting the placeholder at slot {i}");
        }
        Self {
            base,
            mapped: vec![false; slots],
        }
    }

    /// Shows physical page `page` of `section` at slot `slot`.
    fn map(&mut self, slot: usize, section: &Section, page: usize) {
        let at = self.slot(slot);
        // SAFETY: replaces our own placeholder with a page of our section.
        let view = unsafe {
            MapViewOfFile3(
                section.handle,
                GetCurrentProcess(),
                at.cast(),
                (page * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value.cast::<u8>(), at, "page {page} at slot {slot}");
        self.mapped[slot] = true;
    }

    fn slot(&self, slot: usize) -> *mut u8 {
        assert!(slot < self.mapped.len());
        // SAFETY: inside the reservation.
        unsafe { self.base.add(slot * PAGE) }
    }

    /// The words at slot `slot`, which has to be mapped.
    #[allow(clippy::mut_from_ref)]
    fn words(&self, slot: usize) -> &mut [u32] {
        assert!(self.mapped[slot], "slot {slot} is not mapped");
        // SAFETY: a mapped read-write view of one page.
        unsafe { words(self.slot(slot), PAGE) }
    }
}

impl Drop for TitleRange {
    fn drop(&mut self) {
        for (slot, &mapped) in self.mapped.iter().enumerate() {
            let at = self.slot(slot).cast::<c_void>();
            // SAFETY: our own views and placeholders, each let go once.
            unsafe {
                if mapped {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at });
                } else {
                    VirtualFree(at, 0, MEM_RELEASE);
                }
            }
        }
    }
}

fn fill(words: &mut [u32], value: u32) {
    for (k, w) in words.iter_mut().enumerate() {
        *w = value.wrapping_add(k as u32);
    }
}

/// The first word that is not `value + k`, if any.
fn first_wrong(words: &[u32], value: u32) -> Option<(usize, u32)> {
    words
        .iter()
        .enumerate()
        .find(|&(k, &w)| w != value.wrapping_add(k as u32))
        .map(|(k, &w)| (k, w))
}

fn run_checked(bridge: &Bridge, ops: &mut [Op]) {
    bridge.run(ops).expect("run");
    for (i, op) in ops.iter().enumerate() {
        assert_eq!(
            op.mismatches(),
            0,
            "op {i}: {} words wrong, the first at {:?}",
            op.mismatches(),
            op.first_bad()
        );
    }
}

#[test]
fn title_pages_reach_the_gpu_through_one_canonical_buffer() {
    let Some(bridge) = bridge() else { return };
    let section = Section::new(16);
    section.all().fill(0);
    let mut title = TitleRange::new(8);
    // Scattered as the title's pools are, with physical page 3 at two
    // addresses (slots 1 and 6).
    for (slot, page) in [(0, 5), (1, 3), (2, 12), (3, 0), (6, 3), (7, 15)] {
        title.map(slot, &section, page);
    }
    let canonical = bridge
        .wrap(section.view, section.size, false)
        .expect("one no-copy buffer over the whole section");
    assert_ne!(canonical.gpu_address, 0);
    let b = canonical.handle;
    let at = |page: usize| (page * PAGE) as u64;

    // The title's CPU stores at one of a page's addresses are what the GPU
    // reads, with no copy.
    fill(title.words(1), 0x1000_0000);
    assert_eq!(
        first_wrong(title.words(6), 0x1000_0000),
        None,
        "the other address of page 3"
    );
    run_checked(&bridge, &mut [Op::check(b, at(3), PAGE_WORDS, 0x1000_0000)]);

    // The GPU's stores are what the title reads at every address of the
    // page, and through the whole view.
    bridge
        .run(&mut [
            Op::write(b, at(5), PAGE_WORDS, 0x2000_0000),
            Op::write(b, at(3), PAGE_WORDS, 0x3000_0000),
        ])
        .expect("GPU writes");
    assert_eq!(
        first_wrong(title.words(0), 0x2000_0000),
        None,
        "page 5 at slot 0"
    );
    assert_eq!(
        first_wrong(section.page(5), 0x2000_0000),
        None,
        "page 5 in the whole view"
    );
    assert_eq!(
        first_wrong(title.words(1), 0x3000_0000),
        None,
        "page 3 at slot 1"
    );
    assert_eq!(
        first_wrong(title.words(6), 0x3000_0000),
        None,
        "page 3 at slot 6"
    );
    // Nothing next to what the GPU wrote changed.
    assert!(section.page(4).iter().all(|&w| w == 0), "page 4 changed");
    assert!(section.page(6).iter().all(|&w| w == 0), "page 6 changed");

    // Every word of the section, both ways.
    bridge
        .run(&mut [Op::write(b, 0, (section.size / 4) as u32, 0x4000_0000)])
        .expect("GPU writes all");
    assert_eq!(
        first_wrong(section.all(), 0x4000_0000),
        None,
        "the whole view"
    );
    for (slot, page) in [(0, 5), (1, 3), (2, 12), (3, 0), (6, 3), (7, 15)] {
        let first = 0x4000_0000 + page as u32 * PAGE_WORDS;
        assert_eq!(
            first_wrong(title.words(slot), first),
            None,
            "slot {slot} (page {page})"
        );
    }
    fill(section.all(), 0x5000_0000);
    run_checked(
        &bridge,
        &mut [Op::check(b, 0, (section.size / 4) as u32, 0x5000_0000)],
    );
    // A title store through a slot after that, then the GPU again.
    fill(title.words(2), 0x6000_0000);
    run_checked(
        &bridge,
        &mut [Op::check(b, at(12), PAGE_WORDS, 0x6000_0000)],
    );

    bridge.release(b).expect("release");
}

#[test]
fn guarded_title_pages_stay_reachable_by_the_gpu() {
    // VIctory guards title pages read-only to learn what the title writes
    // (`watch.rs`); the GPU's stores through the canonical buffer go past
    // the guard of the title's address, and the guard keeps working.
    let Some(bridge) = bridge() else { return };
    let section = Section::new(2);
    let mut title = TitleRange::new(1);
    title.map(0, &section, 1);
    let canonical = bridge
        .wrap(section.view, section.size, false)
        .expect("wrap");
    let mut old = 0;
    // SAFETY: our own mapped view.
    let ok = unsafe { VirtualProtect(title.slot(0).cast(), PAGE, PAGE_READONLY, &mut old) };
    assert!(ok != 0, "guarding the title's page");
    bridge
        .run(&mut [Op::write(
            canonical.handle,
            PAGE as u64,
            PAGE_WORDS,
            0x7000_0000,
        )])
        .expect("GPU write");
    assert_eq!(
        first_wrong(title.words(0), 0x7000_0000),
        None,
        "read through the guarded address"
    );
    // SAFETY: our own mapped view, back to read-write.
    unsafe { VirtualProtect(title.slot(0).cast(), PAGE, PAGE_READWRITE, &mut old) };
    bridge.release(canonical.handle).expect("release");
}

#[test]
fn cpu_and_gpu_meet_inside_one_command_buffer_through_a_shared_event() {
    // Ordering without waiting for the whole command buffer: the GPU waits
    // for the CPU's stores, the CPU for the GPU's, through one event.
    let Some(bridge) = bridge() else { return };
    let section = Section::new(4);
    section.all().fill(0);
    let mut title = TitleRange::new(1);
    title.map(0, &section, 1);
    let b = bridge
        .wrap(section.view, section.size, false)
        .expect("wrap")
        .handle;
    let event = bridge.event().expect("event");
    let at = |page: usize| (page * PAGE) as u64;
    let mut ops = [
        Op::wait_event(event, 1),
        Op::check(b, at(1), PAGE_WORDS, 0x1100_0000),
        Op::write(b, at(2), PAGE_WORDS, 0x2200_0000),
        Op::signal_event(event, 2),
        Op::wait_event(event, 3),
        Op::check(b, at(3), PAGE_WORDS, 0x3300_0000),
        Op::signal_event(event, 4),
    ];
    let run = bridge.start(&mut ops).expect("start");
    // The GPU is held at the first wait: its check has not run yet.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(bridge.value(event).expect("value"), 0);
    fill(title.words(0), 0x1100_0000);
    bridge.signal(event, 1).expect("signal 1");
    let waited = bridge
        .wait_for(event, 2, Duration::from_secs(5))
        .expect("the GPU's signal 2");
    // The command buffer is still on the GPU (held at the wait for 3), and
    // its write is already what the CPU reads.
    assert_eq!(
        first_wrong(section.page(2), 0x2200_0000),
        None,
        "page 2 before the run ended"
    );
    fill(section.page(3), 0x3300_0000);
    bridge.signal(event, 3).expect("signal 3");
    let times = run.wait().expect("the run");
    assert_eq!(bridge.value(event).expect("value"), 4);
    for (i, op) in ops.iter().enumerate() {
        assert_eq!(op.mismatches(), 0, "op {i} found {:?}", op.first_bad());
    }
    eprintln!("CPU signal -> GPU check + write -> GPU signal -> CPU: {waited:?}; run {times:?}");
    bridge.release(event).expect("release");
    bridge.release(b).expect("release");
}

#[test]
fn no_copy_buffers_start_on_any_4_kib_page() {
    // Metal's pages are 16 KiB, Rosetta's 4 KiB: buffers over 4 KiB
    // pages that are not on 16 KiB still see the right bytes.
    let Some(bridge) = bridge() else { return };
    const BYTES: usize = 128 << 10;
    // SAFETY: a fresh reservation, checked.
    let base = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            BYTES,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    }
    .cast::<u8>();
    assert!(!base.is_null());
    // SAFETY: committed and ours.
    let all = unsafe { words(base, BYTES) };
    all.fill(0xdead_beef);
    for (start, len) in [
        (4 << 10, 4 << 10),
        (12 << 10, 20 << 10),
        (36 << 10, 64 << 10),
    ] {
        // SAFETY: inside the reservation.
        let at = unsafe { base.add(start) };
        let w = bridge
            .wrap(at, len, false)
            .unwrap_or_else(|e| panic!("wrap at +{start:#x}: {e}"));
        bridge
            .run(&mut [Op::write(w.handle, 0, (len / 4) as u32, start as u32)])
            .expect("write");
        let (before, rest) = all.split_at(start / 4);
        let (inside, after) = rest.split_at(len / 4);
        assert_eq!(
            first_wrong(inside, start as u32),
            None,
            "+{start:#x}, {len} bytes"
        );
        assert_eq!(
            before.last().copied(),
            Some(0xdead_beef),
            "before +{start:#x}"
        );
        assert_eq!(
            after.first().copied(),
            Some(0xdead_beef),
            "past +{start:#x}"
        );
        bridge.release(w.handle).expect("release");
        all.fill(0xdead_beef);
        eprintln!("+{start:#x}, {len} bytes: gpu address {:#x}", w.gpu_address);
    }
    // Off a page boundary, refused before Metal sees it.
    // SAFETY: inside the reservation.
    let e = bridge
        .wrap(unsafe { base.add(4) }, 4 << 10, false)
        .expect_err("an unaligned start");
    assert!(e.message.contains("whole pages"), "{e}");
    // SAFETY: the reservation made above.
    unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
}

#[test]
fn reserved_memory_without_access_is_refused() {
    // Wine leaves reserved, uncommitted memory without access (PROT_NONE):
    // a buffer has to cover committed memory.
    let Some(bridge) = bridge() else { return };
    // SAFETY: a fresh reservation, checked.
    let base =
        unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) }.cast::<u8>();
    assert!(!base.is_null());
    match bridge.wrap(base, PAGE, false) {
        Err(e) => eprintln!("reserved memory: refused ({e})"),
        Ok(w) => {
            eprintln!(
                "reserved memory: Metal made a buffer over it ({:#x}); never used",
                w.gpu_address
            );
            bridge.release(w.handle).expect("release");
        }
    }
    // SAFETY: the reservation made above.
    unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
}

#[test]
fn two_buffers_over_one_physical_page() {
    // nowak-kowalski and tebieshi (#221) found that the GPU reads stale data
    // when it writes a page through one buffer and reads it through another
    // in the same command buffer; across command buffers that wait for each
    // other it is right. That is why one canonical buffer per physical page.
    let Some(bridge) = bridge() else { return };
    let section = Section::new(4);
    section.all().fill(0);
    let mut title = TitleRange::new(1);
    title.map(0, &section, 3);
    let canonical = bridge
        .wrap(section.view, section.size, false)
        .expect("canonical")
        .handle;
    let alias = bridge
        .wrap(title.slot(0), PAGE, false)
        .expect("a buffer over the title's address")
        .handle;
    let at3 = (3 * PAGE) as u64;
    // Across command buffers: right.
    bridge
        .run(&mut [Op::write(canonical, at3, PAGE_WORDS, 0x0a00_0000)])
        .expect("write");
    run_checked(&bridge, &mut [Op::check(alias, 0, PAGE_WORDS, 0x0a00_0000)]);
    bridge
        .run(&mut [Op::write(alias, 0, PAGE_WORDS, 0x0b00_0000)])
        .expect("write");
    run_checked(
        &bridge,
        &mut [Op::check(canonical, at3, PAGE_WORDS, 0x0b00_0000)],
    );
    // In one command buffer, after a barrier and after an encoder split:
    // measured, not asserted.
    let mut same = [
        Op::write(canonical, at3, PAGE_WORDS, 0x0c00_0000),
        Op::check(alias, 0, PAGE_WORDS, 0x0c00_0000),
        Op::write(canonical, at3, PAGE_WORDS, 0x0d00_0000),
        Op::split(),
        Op::check(alias, 0, PAGE_WORDS, 0x0d00_0000),
    ];
    bridge.run(&mut same).expect("same command buffer");
    eprintln!(
        "two buffers over one page, one command buffer: {} of {PAGE_WORDS} words stale after a barrier, {} after an encoder split",
        same[1].mismatches(),
        same[4].mismatches()
    );
    bridge.release(alias).expect("release");
    bridge.release(canonical).expect("release");
}

/// The size of [`large_section_past_4_gib`]'s section: `VICTORY_METAL_LARGE_GIB`
/// (default 6.25, a little over the 6.09 GB section Grand Theft Auto VI asks
/// for).
fn large_pages() -> usize {
    let gib: f64 = std::env::var("VICTORY_METAL_LARGE_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6.25);
    ((gib * (1u64 << 30) as f64) as usize).div_ceil(PAGE)
}

#[test]
#[ignore = "maps and wires several GiB: run alone, with no game running"]
fn large_section_past_4_gib() {
    // D3DMetal reads zeros from buffers of 8 GiB or more and bounds raw views
    // at 2 GiB - 4 (pr-drafts/memory-architecture.md, section 4b); a no-copy
    // buffer over the title's whole physical section has to reach every page.
    let Some(bridge) = bridge() else { return };
    let device = bridge.device().expect("device");
    let pages = large_pages();
    let size = pages * PAGE;
    eprintln!(
        "section {:.2} GiB; maxBufferLength {:.2} GiB, recommended working set {:.2} GiB",
        size as f64 / (1u64 << 30) as f64,
        device.max_buffer_length as f64 / (1u64 << 30) as f64,
        device.recommended_working_set as f64 / (1u64 << 30) as f64
    );
    let started = Instant::now();
    let section = Section::new(pages);
    eprintln!("section and view: {:?}", started.elapsed());
    let canonical = bridge
        .wrap(section.view, section.size, false)
        .expect("wrap the whole section");
    eprintln!(
        "wrap: {:?}, gpu address {:#x}",
        canonical.wrap, canonical.gpu_address
    );
    let b = canonical.handle;
    let probes = [
        0,
        (2 << 30) / PAGE - 1,
        (2 << 30) / PAGE,
        (4 << 30) / PAGE,
        (6 << 30) / PAGE - 1,
        pages - 1,
    ];
    let probes: Vec<usize> = probes.into_iter().filter(|&p| p < pages).collect();
    let mut ops: Vec<Op> = probes
        .iter()
        .map(|&p| Op::write(b, (p * PAGE) as u64, PAGE_WORDS, p as u32 * 7))
        .collect();
    let first = bridge.run(&mut ops).expect("GPU writes");
    for &p in &probes {
        assert_eq!(
            first_wrong(section.page(p), p as u32 * 7),
            None,
            "page {p} ({:.2} GiB in)",
            (p * PAGE) as f64 / (1u64 << 30) as f64
        );
    }
    for &p in &probes {
        fill(section.page(p), p as u32 * 11);
    }
    let mut ops: Vec<Op> = probes
        .iter()
        .map(|&p| Op::check(b, (p * PAGE) as u64, PAGE_WORDS, p as u32 * 11))
        .collect();
    let second = bridge.run(&mut ops).expect("GPU checks");
    for (op, &p) in ops.iter().zip(&probes) {
        assert_eq!(
            op.mismatches(),
            0,
            "page {p}: first wrong word {:?}",
            op.first_bad()
        );
    }
    eprintln!("first run (first use of the buffer) {first:?}; second {second:?}");
    // The whole section read once by the GPU (what wiring all of it costs).
    let mut all = [Op::check(b, 0, (size / 4).min(1 << 31) as u32, 0)];
    let whole = bridge.run(&mut all).expect("read all");
    eprintln!(
        "one GPU read pass over all {:.2} GiB: {whole:?}",
        size as f64 / (1u64 << 30) as f64
    );
    bridge.release(b).expect("release");
}

#[test]
#[ignore = "a measurement: run with --nocapture, no game running"]
fn shared_memory_costs() {
    let Some(bridge) = bridge() else { return };
    // Wrapping, by size, over committed VirtualAlloc memory.
    for bytes in [64usize << 10, 1 << 20, 64 << 20, 1 << 30] {
        // SAFETY: a fresh reservation, checked.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!base.is_null());
        let mut wraps = Vec::new();
        for _ in 0..5 {
            let w = bridge.wrap(base, bytes, false).expect("wrap");
            wraps.push(w.wrap);
            bridge.release(w.handle).expect("release");
        }
        wraps.sort();
        let w = bridge.wrap(base, bytes, false).expect("wrap");
        let words = (bytes / 4) as u32;
        let mut first = [Op::check(w.handle, 0, words, 0)];
        let t1 = bridge.run(&mut first).expect("first pass");
        let mut again = [Op::check(w.handle, 0, words, 0)];
        let t2 = bridge.run(&mut again).expect("second pass");
        let gbs = |d: Duration| bytes as f64 / d.as_secs_f64() / 1e9;
        eprintln!(
            "{:>8} KiB: wrap median {:?}; GPU read, first use {:?} ({:.1} GB/s), again {:?} ({:.1} GB/s)",
            bytes >> 10,
            wraps[2],
            t1.gpu,
            gbs(t1.gpu),
            t2.gpu,
            gbs(t2.gpu)
        );
        bridge.release(w.handle).expect("release");
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }
    // CPU -> GPU -> CPU through one event, the GPU doing nothing between.
    let event = bridge.event().expect("event");
    let mut latencies = Vec::new();
    for round in 0..20u64 {
        let mut ops = [
            Op::wait_event(event, 2 * round + 1),
            Op::signal_event(event, 2 * round + 2),
        ];
        let run = bridge.start(&mut ops).expect("start");
        std::thread::sleep(Duration::from_millis(2));
        let t = Instant::now();
        bridge.signal(event, 2 * round + 1).expect("signal");
        bridge
            .wait_for(event, 2 * round + 2, Duration::from_secs(5))
            .expect("wait");
        latencies.push(t.elapsed());
        run.wait().expect("run");
    }
    latencies.sort();
    eprintln!(
        "event round trip CPU -> GPU -> CPU (GPU waiting already): median {:?}, min {:?}, max {:?}",
        latencies[10], latencies[0], latencies[19]
    );
    bridge.release(event).expect("release");
}
