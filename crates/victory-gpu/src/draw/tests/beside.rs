//! `VICTORY_PREP_BESIDE` (`crate::host::flight`): the next submission's
//! buffer copies are refreshed beside the flight in the air, but for what
//! the flight may write, and nothing the title writes meanwhile is lost.

use std::sync::Arc;

use super::*;
use crate::host::{BesideGuard, Gate, ModeGuard, beside_counts};

/// A raw view of a buffer's 256 bytes.
const RAW_256: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];

const PAGE: usize = crate::watch::PAGE;

/// A flight's shader writes a dword of a buffer's first page through a
/// heap's unordered view; the title writes that page and the next one while
/// the flight is in the air, and a second list asks for the whole buffer.
/// Beside the flight only the second page is refreshed: the first, which
/// the flight may write, keeps the copy's bytes until the flight has landed
/// and its write has gone back. The title then writes the second page
/// again: the submission after the flight reads that write too, as the
/// refresh guarded the page again when it took it.
#[test]
fn beside_a_flight_only_what_it_cannot_write_is_refreshed_and_later_title_writes_follow() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const LIVE_MARKED_WRITE: &str = r#"
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer writes : register(u1);
cbuffer c : register(b0) { uint byte_offset; uint value; uint mark_word_bytes; uint mark_mask; }
[numthreads(1, 1, 1)]
void main() {
    dst.Store(byte_offset, value);
    writes.Store(mark_word_bytes, mark_mask);
}
"#;
    let (root, pso) = compile_cs(
        LIVE_MARKED_WRITE,
        &[
            Param::Table(&[uav(0, 1)]),
            Param::Uav { register: 1 },
            Param::Constants {
                register: 0,
                count: 4,
            },
        ],
    );
    let (_, title_heap, heap_host) = view_heap(8);
    let p = crate::watch::tests::pool(2);
    let word = |at: usize| {
        // SAFETY: a dword of our pages.
        unsafe { (at as *const u32).read_volatile() }
    };
    let set = |at: usize, value: u32| {
        // SAFETY: a dword of our pages.
        unsafe { (at as *mut u32).write_volatile(value) }
    };
    set(p, 0x1111_1111);
    set(p + 16, 0x2222_2222);
    set(p + PAGE, 0x3333_3333);
    let buf = buffer_at_address(p as u64, 2 * PAGE as u64);
    let copy = crate::shadow::find(p as u64, 2 * PAGE as u64).expect("the pages' copy");
    let Some((bit, _)) = copy.mark_entry(p as u64) else {
        crate::shadow::unregister(buf as usize, p as u64);
        return; // no write marks on this host
    };
    crate::descriptor::create_uav(
        std::ptr::null(),
        buf,
        std::ptr::null(),
        &RAW_256,
        title_heap.start,
    );
    let _ = heap_host.mirror(0, title_heap.start, true);
    let _mode = ModeGuard::new(true, false);
    let _beside = BesideGuard::new(true);
    // The copy holds the title's bytes before the flight.
    copy.update(p, 2 * PAGE);
    let mut gate = Gate::hold(host);
    let list = Arc::new(host.new_list().unwrap());
    list.defer_heap(&heap_host.viewed);
    let writes = crate::shadow::view_writes(false).expect("the write marks bitmap");
    let constants = [0u32, 0xDEAD_BEEF, (bit >> 5) * 4, 1 << (bit & 31)];
    // SAFETY: an open list, live objects and descriptors in our heap.
    unsafe {
        list.list
            .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
        list.list.SetComputeRootUnorderedAccessView(1, writes);
        list.list
            .SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
        list.list.SetPipelineState(&pso);
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    host.execute_flight1(vec![list], vec![]);
    // The title goes on and writes both pages while the flight is in the air.
    set(p + 16, 0xCAFE_BABE);
    set(p + PAGE, 0xAAAA_0001);
    // The next list asks for the whole buffer.
    let next = Arc::new(host.new_list().unwrap());
    next.defer_refresh(&copy, p, 2 * PAGE);
    next.close();
    let before = beside_counts();
    host.prepare_beside(std::slice::from_ref(&next));
    let after = beside_counts();
    let held = gate.held();
    let (first, second) = (copy.read_host((p + 16) as u64, 1)[0], copy.read_host((p + PAGE) as u64, 1)[0]);
    // The title writes the page refreshed beside the flight once more.
    set(p + PAGE, 0xAAAA_0002);
    gate.open();
    host.execute_flight1(vec![next], vec![]);
    host.finish_flight_now();
    if held {
        assert_eq!(after[0], before[0] + 1, "one go beside the flight");
        assert_eq!(
            second, 0xAAAA_0001,
            "the page the flight cannot write was refreshed beside it"
        );
        assert_eq!(
            first, 0x2222_2222,
            "the page the flight may write was left alone until it landed"
        );
        assert!(after[2] > before[2], "bytes copied beside the flight are counted");
        assert!(after[3] > before[3], "bytes left for after the landing are counted");
    }
    assert_eq!(word(p), 0xDEAD_BEEF, "the shader's write went back");
    assert_eq!(word(p + 16), 0xCAFE_BABE, "the title's write on the shader's page stays");
    assert_eq!(
        copy.read_host(p as u64, 1)[0],
        0xDEAD_BEEF,
        "the copy keeps the shader's write"
    );
    assert_eq!(
        copy.read_host((p + 16) as u64, 1)[0],
        0xCAFE_BABE,
        "the copy took the title's write on the shader's page after the landing"
    );
    assert_eq!(
        copy.read_host((p + PAGE) as u64, 1)[0],
        0xAAAA_0002,
        "a title write after the refresh beside the flight is copied by the next preparation"
    );
    crate::shadow::unregister(buf as usize, p as u64);
}

/// With the switch off nothing is refreshed beside a flight.
#[test]
fn beside_a_flight_nothing_is_refreshed_without_the_switch() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let p = crate::watch::tests::pool(1);
    // SAFETY: a dword of our page.
    unsafe { (p as *mut u32).write_volatile(0x1111_1111) };
    let buf = buffer_at_address(p as u64, PAGE as u64);
    let copy = crate::shadow::find(p as u64, PAGE as u64).expect("the page's copy");
    let _mode = ModeGuard::new(true, false);
    let _beside = BesideGuard::new(false);
    copy.update(p, PAGE);
    let mut gate = Gate::hold(host);
    let flight = Arc::new(host.new_list().unwrap());
    flight.close();
    host.execute_flight1(vec![flight], vec![]);
    // SAFETY: a dword of our page.
    unsafe { (p as *mut u32).write_volatile(0x5555_5555) };
    let next = Arc::new(host.new_list().unwrap());
    next.defer_refresh(&copy, p, PAGE);
    next.close();
    let before = beside_counts();
    host.prepare_beside(std::slice::from_ref(&next));
    assert_eq!(beside_counts()[..6], before[..6], "no go beside the flight");
    assert_eq!(
        copy.read_host(p as u64, 1)[0],
        0x1111_1111,
        "the copy is as it was while the flight is in the air"
    );
    gate.open();
    host.execute_flight1(vec![next], vec![]);
    host.finish_flight_now();
    assert_eq!(copy.read_host(p as u64, 1)[0], 0x5555_5555, "refreshed by the preparation");
    crate::shadow::unregister(buf as usize, p as u64);
}

/// A heap holds two unordered views of one buffer, on its first and its
/// third page, and a flight's shader writes through the first. What the
/// flight may write is what the views reach, not everything from the first
/// view to the last: the page between them, written by the title during
/// the flight, is refreshed beside it, and the two pages under the views
/// are left until it has landed.
#[test]
fn beside_a_flight_the_pages_between_its_heaps_unordered_views_are_refreshed() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const LIVE_MARKED_WRITE: &str = r#"
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer writes : register(u1);
cbuffer c : register(b0) { uint byte_offset; uint value; uint mark_word_bytes; uint mark_mask; }
[numthreads(1, 1, 1)]
void main() {
    dst.Store(byte_offset, value);
    writes.Store(mark_word_bytes, mark_mask);
}
"#;
    let (root, pso) = compile_cs(
        LIVE_MARKED_WRITE,
        &[
            Param::Table(&[uav(0, 1)]),
            Param::Uav { register: 1 },
            Param::Constants {
                register: 0,
                count: 4,
            },
        ],
    );
    let (_, title_heap, heap_host) = view_heap(8);
    let p = crate::watch::tests::pool(3);
    let set = |at: usize, value: u32| {
        // SAFETY: a dword of our pages.
        unsafe { (at as *mut u32).write_volatile(value) }
    };
    for (page, value) in [0x1111_1111u32, 0x2222_2222, 0x3333_3333].into_iter().enumerate() {
        set(p + page * PAGE + 16, value);
    }
    let buf = buffer_at_address(p as u64, 3 * PAGE as u64);
    let copy = crate::shadow::find(p as u64, 3 * PAGE as u64).expect("the pages' copy");
    let Some((bit, _)) = copy.mark_entry(p as u64) else {
        crate::shadow::unregister(buf as usize, p as u64);
        return; // no write marks on this host
    };
    // Raw views of 256 bytes: of the first page, and of the third.
    let third: [u32; 10] = [39, 1, (2 * PAGE / 4) as u32, 0, 64, 0, 0, 0, 1, 0];
    let slot1 = title_heap.start + 32;
    crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &RAW_256, title_heap.start);
    crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &third, slot1);
    let _ = heap_host.mirror(0, title_heap.start, true);
    let _ = heap_host.mirror(1, slot1, true);
    let _mode = ModeGuard::new(true, false);
    let _beside = BesideGuard::new(true);
    copy.update(p, 3 * PAGE);
    let mut gate = Gate::hold(host);
    let list = Arc::new(host.new_list().unwrap());
    list.defer_heap(&heap_host.viewed);
    let writes = crate::shadow::view_writes(false).expect("the write marks bitmap");
    let constants = [0u32, 0xDEAD_BEEF, (bit >> 5) * 4, 1 << (bit & 31)];
    // SAFETY: an open list, live objects and descriptors in our heap.
    unsafe {
        list.list
            .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
        list.list.SetComputeRootUnorderedAccessView(1, writes);
        list.list
            .SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
        list.list.SetPipelineState(&pso);
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    host.execute_flight1(vec![list], vec![]);
    // The title writes all three pages while the flight is in the air.
    for (page, value) in [0xAAAA_0000u32, 0xAAAA_0001, 0xAAAA_0002].into_iter().enumerate() {
        set(p + page * PAGE + 16, value);
    }
    let next = Arc::new(host.new_list().unwrap());
    next.defer_refresh(&copy, p, 3 * PAGE);
    next.close();
    host.prepare_beside(std::slice::from_ref(&next));
    let held = gate.held();
    let seen: Vec<u32> = (0..3)
        .map(|page| copy.read_host((p + page * PAGE + 16) as u64, 1)[0])
        .collect();
    gate.open();
    host.execute_flight1(vec![next], vec![]);
    host.finish_flight_now();
    if held {
        assert_eq!(
            seen,
            [0x1111_1111, 0xAAAA_0001, 0x3333_3333],
            "beside the flight: the pages under its heap's unordered views left, the page between refreshed"
        );
    }
    let after: Vec<u32> = (0..3)
        .map(|page| copy.read_host((p + page * PAGE + 16) as u64, 1)[0])
        .collect();
    assert_eq!(
        after,
        [0xAAAA_0000, 0xAAAA_0001, 0xAAAA_0002],
        "after the landing the copy holds the title's writes on all three"
    );
    assert_eq!(
        copy.read_host(p as u64, 1)[0],
        0xDEAD_BEEF,
        "and the shader's write on the first"
    );
    crate::shadow::unregister(buf as usize, p as u64);
}

/// While a flight lands beside a heap sync, a write of ours into a
/// shader-visible heap's memory is noticed (the sync is made again after
/// the landing); a write elsewhere is not.
#[test]
fn a_landing_beside_a_heap_sync_notices_writes_into_heap_memory() {
    if crate::host::host().is_none() {
        return;
    }
    let (_, title_heap, _heap_host) = view_heap(8);
    let _mode = ModeGuard::new(true, false);
    let p = crate::watch::tests::pool(1);
    assert!(!crate::descriptor::landing_beside(true));
    drop(crate::watch::writing(p, 16));
    assert!(
        !crate::descriptor::landing_beside(false),
        "a write outside the heaps is not noticed"
    );
    crate::descriptor::landing_beside(true);
    drop(crate::watch::writing(title_heap.start as usize, 32));
    assert!(
        crate::descriptor::landing_beside(false),
        "a write into a heap's memory is"
    );
    drop(crate::watch::writing(title_heap.start as usize, 32));
    assert!(
        !crate::descriptor::landing_beside(false),
        "and nothing is noted while no landing runs beside a sync"
    );
    crate::watch::tests::free(p);
}
