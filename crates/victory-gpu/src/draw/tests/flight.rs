//! Title submissions as Flight-1 flights (`VICTORY_FLIGHT1`, with and
//! without `VICTORY_FLIGHT1_PREWARM`, `crate::host::flight`): what their
//! shaders write reaches title memory before anything the title waits on
//! says they ran, in order, with nothing lost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::*;
use crate::host::flight::{Gate, ModeGuard, TEST_PREWARM};
use windows::Win32::Foundation::WAIT_OBJECT_0;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

/// A raw view of a buffer's 256 bytes.
const RAW_256: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];

/// The slot the spinner's (never made) store goes through.
const SINK: u32 = 15;

/// Title buffers of 256 bytes, each viewed by a slot of a title heap (as an
/// unordered view), and shaders that store a value through one or copy a
/// dword from one into another.
struct Rig {
    title_heap: &'static crate::descriptor::Heap,
    heap_host: &'static crate::descriptor::host::HostHeap,
    buffers: Vec<This>,
    writer: (ID3D12RootSignature, ID3D12PipelineState),
    copier: (ID3D12RootSignature, ID3D12PipelineState),
    /// Keeps the GPU busy a while (a flight still runs when the next
    /// submission is prepared), writing nothing in practice.
    spinner: (ID3D12RootSignature, ID3D12PipelineState),
}

impl Rig {
    fn new(buffers: u32) -> Rig {
        let (_, title_heap, heap_host) = view_heap(16);
        let mut rig = Rig {
            title_heap,
            heap_host,
            buffers: Vec::new(),
            writer: compile_cs(
                r#"
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint offset; uint value; }
[numthreads(1, 1, 1)]
void main() { dst.Store(offset, value); }
"#,
                &[
                    Param::Table(&[uav(0, 1)]),
                    Param::Constants {
                        register: 0,
                        count: 2,
                    },
                ],
            ),
            copier: compile_cs(
                r#"
RWByteAddressBuffer dst : register(u0);
ByteAddressBuffer src : register(t0);
cbuffer c : register(b0) { uint offset; }
[numthreads(1, 1, 1)]
void main() { dst.Store(offset, src.Load(offset)); }
"#,
                &[
                    Param::Table(&[uav(0, 1)]),
                    Param::Table(&[srv(0, 1)]),
                    Param::Constants {
                        register: 0,
                        count: 1,
                    },
                ],
            ),
            spinner: compile_cs(
                r#"
RWByteAddressBuffer sink : register(u0);
cbuffer c : register(b0) { uint n; }
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    for (uint i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; }
    if (x == 0x9e3779b9u && n == 0xffffffffu) { sink.Store(0, x); }
}
"#,
                &[
                    Param::Table(&[uav(0, 1)]),
                    Param::Constants {
                        register: 0,
                        count: 1,
                    },
                ],
            ),
        };
        for slot in 0..buffers {
            rig.view(slot, buffer(256));
        }
        rig
    }

    /// The title makes slot `slot` a view of `buffer` (its call mirrors it
    /// into the host heap at once).
    fn view(&mut self, slot: u32, buffer: This) {
        crate::descriptor::create_uav(
            std::ptr::null(),
            buffer,
            std::ptr::null(),
            &RAW_256,
            self.handle(slot),
        );
        let _ = self.heap_host.mirror(slot, self.handle(slot), true);
        self.buffers.push(buffer);
    }

    /// The title makes slot `slot` a raw shader view of `buffer`.
    fn view_srv(&mut self, slot: u32, buffer: This) {
        const RAW_SRV_256: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        crate::descriptor::create_srv(std::ptr::null(), buffer, &RAW_SRV_256, self.handle(slot));
        let _ = self.heap_host.mirror(slot, self.handle(slot), true);
        self.buffers.push(buffer);
    }

    fn handle(&self, slot: u32) -> u64 {
        self.title_heap.start + 32 * u64::from(slot)
    }

    /// The title memory of slot `slot`'s buffer at `offset`.
    fn at(&self, slot: u32, offset: u32) -> *mut u32 {
        let (bytes, _) = crate::resource::buffer_bytes(self.buffers[slot as usize]).unwrap();
        // SAFETY: inside the buffer's 256 bytes.
        unsafe { bytes.add(offset as usize).cast() }
    }

    fn title(&self, slot: u32, offset: u32) -> u32 {
        // SAFETY: a dword of the buffer's title memory.
        unsafe { self.at(slot, offset).read_volatile() }
    }

    /// The title writes a dword of slot `slot`'s buffer from the CPU.
    fn write(&self, slot: u32, offset: u32, value: u32) {
        // SAFETY: a dword of the buffer's title memory.
        unsafe { self.at(slot, offset).write_volatile(value) };
    }

    fn view_of(&self, slot: u32) -> crate::descriptor::ViewPin {
        crate::descriptor::view(self.handle(slot)).expect("the slot's view")
    }

    /// A host list whose shaders store each `(slot, offset, value)`, the
    /// writes declared as they are for a title draw (copied back after).
    /// Left open.
    fn list(&self, writes: &[(u32, u32, u32)]) -> Arc<HostList> {
        let list = Arc::new(crate::host::host().unwrap().new_list().unwrap());
        for &(slot, offset, value) in writes {
            readable(&list, &used(ResourceType::Uav), &self.view_of(slot), false);
            let constants = [offset, value];
            // SAFETY: an open list, live objects and descriptors in our heap.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(self.heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&self.writer.0);
                list.list
                    .SetComputeRootDescriptorTable(0, self.heap_host.gpu(slot));
                list.list
                    .SetComputeRoot32BitConstants(1, 2, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&self.writer.1);
                list.list.Dispatch(1, 1, 1);
                list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
            }
        }
        list
    }

    /// A host list whose shader copies the dword at `offset` of slot
    /// `from`'s buffer (a shader view, read as the title wrote it:
    /// refreshed) into slot `to`'s (copied back after). Left open.
    fn copying(&self, from: u32, to: u32, offset: u32) -> Arc<HostList> {
        let list = Arc::new(crate::host::host().unwrap().new_list().unwrap());
        readable(&list, &used(ResourceType::Srv), &self.view_of(from), false);
        readable(&list, &used(ResourceType::Uav), &self.view_of(to), false);
        // SAFETY: an open list, live objects and descriptors in our heap.
        unsafe {
            list.list
                .SetDescriptorHeaps(&[Some(self.heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&self.copier.0);
            list.list
                .SetComputeRootDescriptorTable(0, self.heap_host.gpu(to));
            list.list
                .SetComputeRootDescriptorTable(1, self.heap_host.gpu(from));
            list.list
                .SetComputeRoot32BitConstants(2, 1, [offset].as_ptr().cast(), 0);
            list.list.SetPipelineState(&self.copier.1);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        // Then a while of GPU work through a slot no list declares (its
        // store never happens): the flight is still in the air, its write
        // made, when the title submits the next one.
        // SAFETY: as above.
        unsafe {
            list.list.SetComputeRootSignature(&self.spinner.0);
            list.list
                .SetComputeRootDescriptorTable(0, self.heap_host.gpu(SINK));
            list.list
                .SetComputeRoot32BitConstants(1, 1, [20_000u32].as_ptr().cast(), 0);
            list.list.SetPipelineState(&self.spinner.1);
            list.list.Dispatch(64, 1, 1);
        }
        list
    }
}

fn title_queue() -> This {
    crate::queue::create_direct_for_test()
}

fn title_fence() -> This {
    let mut fence = std::ptr::null();
    assert_eq!(
        crate::fence::create(
            std::ptr::null(),
            0,
            0,
            &crate::fence::FENCE_IIDS[0],
            &mut fence
        ),
        S_OK
    );
    fence
}

/// The fence's value as the title polls it (`GetCompletedValue`).
fn completed(fence: This) -> u64 {
    crate::object::slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence)
}

/// `SetEventOnCompletion` as the title calls it.
fn on_completion(fence: This, value: u64, event: *mut std::ffi::c_void) -> HRESULT {
    crate::object::slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> HRESULT>(
        fence, 0x48,
    )(fence, value, event)
}

fn release(object: This) {
    crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
}

/// Runs a list of the title's as a flight, as the queues' pump does.
fn submit(list: &Arc<HostList>) {
    list.close();
    crate::host::host()
        .unwrap()
        .execute_flight1(vec![list.clone()], vec![]);
}

/// Waits (at most 10 s) for `done`, calling nothing of ours meanwhile.
fn poll(what: &str, done: impl Fn() -> bool) {
    let since = Instant::now();
    while !done() {
        assert!(
            since.elapsed() < Duration::from_secs(10),
            "{what}: timed out"
        );
        std::thread::sleep(Duration::from_micros(200));
    }
}

/// How the title learns its fence moved.
#[derive(Clone, Copy, Debug)]
enum Learns {
    /// An event it asked for before the Signal.
    ByEvent,
    /// Reading the fence's memory, with no call of ours.
    ByMemory,
    /// Polling `GetCompletedValue`.
    ByPolling,
    /// A blocking wait (`SetEventOnCompletion` with no event), which lands
    /// the flight first.
    Blocking,
}

/// A Signal behind a flight moves its fence only once what the flight's
/// shaders wrote is in title memory, however the title learns of it.
#[test]
fn a_title_fence_moves_once_its_flights_writes_are_in_title_memory() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let rig = Rig::new(1);
    let _mode = ModeGuard::new(true, false);
    let rounds = [
        Learns::ByEvent,
        Learns::ByMemory,
        Learns::ByPolling,
        Learns::Blocking,
    ];
    for (round, learns) in rounds.into_iter().enumerate() {
        let (queue, fence) = (title_queue(), title_fence());
        let value = 0xA500 + round as u32;
        let offset = 4 * round as u32;
        // SAFETY: a plain auto-reset event.
        let event = unsafe { CreateEventW(None, false, false, None) }.unwrap();
        if matches!(learns, Learns::ByEvent) {
            assert_eq!(on_completion(fence, 1, event.0), S_OK);
        }
        // The blocking wait comes while the flight is surely in the air.
        let mut gate = matches!(learns, Learns::Blocking).then(|| Gate::hold(host));
        submit(&rig.list(&[(0, offset, value)]));
        assert_eq!(crate::queue::signal_for_test(queue, fence, 1), S_OK);
        if crate::fence::value(fence) == 1 {
            assert_eq!(
                rig.title(0, offset),
                value,
                "{learns:?}: the fence moved before the copy-back"
            );
        }
        match learns {
            Learns::ByEvent => {
                // SAFETY: our event.
                let woke = unsafe { WaitForSingleObject(event, 10_000) };
                assert_eq!(woke, WAIT_OBJECT_0);
            }
            Learns::ByMemory => poll("the fence's memory", || crate::fence::value(fence) == 1),
            Learns::ByPolling => poll("the polled fence", || completed(fence) == 1),
            Learns::Blocking => {
                assert_eq!(
                    crate::fence::value(fence),
                    0,
                    "the Signal stands behind the held flight"
                );
                let opener = gate.take().map(|mut gate| {
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(50));
                        gate.open();
                    })
                });
                assert_eq!(
                    on_completion(fence, 1, std::ptr::null_mut()),
                    S_OK,
                    "the blocking wait"
                );
                assert_eq!(crate::fence::value(fence), 1);
                if let Some(opener) = opener {
                    opener.join().unwrap();
                }
            }
        }
        assert_eq!(
            rig.title(0, offset),
            value,
            "{learns:?}: in title memory when the fence says so"
        );
        release(queue);
        release(fence);
    }
}

/// A title thread may read what the GPU wrote with no call of ours (a
/// bottom-of-pipe word it polls, as on the console): the flight lands on
/// our thread once the GPU has run it, the shaders' writes first.
#[test]
fn the_title_sees_a_flights_writes_without_calling_us() {
    let Some(_) = crate::host::host() else { return };
    let rig = Rig::new(1);
    let _mode = ModeGuard::new(true, false);
    let word = Box::new(AtomicU64::new(0));
    let list = rig.list(&[(0, 16, 0xB0B)]);
    list.defer_write(word.as_ptr().cast(), crate::queue::Written::Value(1), 8);
    submit(&list);
    poll("the bottom-of-pipe word", || {
        word.load(Ordering::Acquire) == 1
    });
    assert_eq!(
        rig.title(0, 16),
        0xB0B,
        "the shader's write landed before the bottom-of-pipe write"
    );
}

/// Signals behind a flight move their fences when it lands, in their
/// order, after its copy-back, as a title thread polling memory sees: a
/// fence never goes back, and another queue's Signal of the same fence
/// stays behind them.
#[test]
fn signals_behind_a_flight_move_in_order() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let rig = Rig::new(1);
    let _mode = ModeGuard::new(true, false);
    let (queue, other) = (title_queue(), title_queue());
    let fence = title_fence();
    // A long-ish flight, so the Signals are attached while it runs.
    let writes: Vec<(u32, u32, u32)> = (0..32).map(|i| (0, 32 + 4 * (i % 8), 0xC00 + i)).collect();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watched = {
        let (stop, at) = (stop.clone(), fence as usize);
        let written = rig.at(0, 60) as usize;
        std::thread::spawn(move || {
            // SAFETY: a dword of the buffer's title memory, alive until the end.
            let dword = || unsafe { (written as *const u32).read_volatile() };
            let mut seen = vec![(0, 0)];
            while !stop.load(Ordering::Acquire) {
                let v = crate::fence::value(at as This);
                if seen.last().map(|s| s.0) != Some(v) {
                    seen.push((v, dword()));
                }
            }
            seen.push((crate::fence::value(at as This), dword()));
            seen
        })
    };
    submit(&rig.list(&writes));
    for value in [1, 2] {
        assert_eq!(crate::queue::signal_for_test(queue, fence, value), S_OK);
    }
    assert_eq!(crate::queue::signal_for_test(other, fence, 3), S_OK);
    host.finish_flight_now();
    poll("the last Signal", || crate::fence::value(fence) == 3);
    stop.store(true, Ordering::Release);
    let seen = watched.join().unwrap();
    assert!(
        seen.windows(2).all(|w| w[0].0 <= w[1].0),
        "the fence went back: {seen:?}"
    );
    assert_eq!(seen.last().map(|s| s.0), Some(3), "{seen:?}");
    // The last store to dword 60 is i = 31 (offset 32 + 4 * 7).
    assert!(
        seen.iter().all(|&(v, dword)| v == 0 || dword == 0xC00 + 31),
        "the fence moved before the copy-back: {seen:?}"
    );
    for object in [queue, other, fence] {
        release(object);
    }
}

/// The title writes a view into a heap slot while a flight that binds the
/// heap runs on the host GPU (its call makes the host descriptor at once);
/// the next flight writes through it, and both land.
#[test]
fn a_heap_slot_written_during_a_flight_serves_the_next() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let mut rig = Rig::new(1);
    let _ = host.log_messages();
    let _mode = ModeGuard::new(true, false);
    submit(&rig.list(&[(0, 0, 0xD01)]));
    rig.view(1, buffer(256));
    submit(&rig.list(&[(1, 0, 0xD02)]));
    host.finish_flight_now();
    assert_eq!(rig.title(0, 0), 0xD01, "the first flight's write");
    assert_eq!(
        rig.title(1, 0),
        0xD02,
        "the second's, through the slot written meanwhile"
    );
    assert_eq!(host.log_messages(), None);
    assert!(!crate::host::device_removed());
}

/// Many flights back to back, with and without the prewarm: each copies a
/// dword the title wrote from the CPU just before submitting it (from a
/// buffer the flight before only reads, which the prewarm brings up to
/// date while that flight runs) into another buffer every flight writes
/// (its copy-back merges into title memory; the prewarm must leave it
/// alone), and stores a bottom-of-pipe count; their work for after runs in
/// order, and nothing is lost.
#[test]
fn back_to_back_flights_lose_no_writes() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const N: u32 = 64;
    let mut rig = Rig::new(0);
    for slot in 0..4 {
        if slot % 2 == 0 {
            // Write-watched memory, as the title's pools are: its copy
            // takes only the pages the title wrote (the prewarm's kind).
            let pool = crate::watch::tests::pool(1);
            rig.view_srv(
                slot,
                buffer_at_address(pool as u64, crate::watch::PAGE as u64),
            );
        } else {
            let pool = crate::watch::tests::pool(1);
            rig.view(
                slot,
                buffer_at_address(pool as u64, crate::watch::PAGE as u64),
            );
        }
    }
    let sink = buffer(256);
    crate::descriptor::create_uav(
        std::ptr::null(),
        sink,
        std::ptr::null(),
        &RAW_256,
        rig.handle(SINK),
    );
    let _ = rig.heap_host.mirror(SINK, rig.handle(SINK), true);
    for (round, prewarm) in [false, true].into_iter().enumerate() {
        let (from, to) = (2 * round as u32, 2 * round as u32 + 1);
        let _mode = ModeGuard::new(true, false);
        TEST_PREWARM.with(|p| p.set(prewarm));
        let word = Box::new(AtomicU64::new(0));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let early = crate::host::flight::early_counts();
        for i in 0..N {
            rig.write(from, 4 * (i % 32), 0x2000 * (round as u32 + 1) + i);
            // The title also writes the other half of the page the flights
            // write, while the one before runs.
            rig.write(to, 128 + 4 * (i % 32), 0x3000 + i);
            let list = rig.copying(from, to, 4 * (i % 32));
            list.defer_write(
                word.as_ptr().cast(),
                crate::queue::Written::Value(u64::from(i) + 1),
                8,
            );
            let log = order.clone();
            list.defer_after(move || log.lock().unwrap().push(i));
            submit(&list);
        }
        host.finish_flight_now();
        TEST_PREWARM.with(|p| p.set(false));
        let now = crate::host::flight::early_counts();
        let early: Vec<u64> = now.iter().zip(early).map(|(a, b)| a - b).collect();
        if prewarm {
            assert!(
                early[3] > 0,
                "the prewarm took copies while flights ran: {early:?}"
            );
        }
        let lost: Vec<u32> = (0..32)
            .filter(|&k| rig.title(to, 4 * k) != 0x2000 * (round as u32 + 1) + 32 + k)
            .collect();
        assert!(
            lost.is_empty(),
            "prewarm {prewarm}: dwords lost or stale {lost:?}"
        );
        let clobbered: Vec<u32> = (0..32)
            .filter(|&k| rig.title(to, 128 + 4 * k) != 0x3000 + 32 + k)
            .collect();
        assert!(
            clobbered.is_empty(),
            "prewarm {prewarm}: the title's own writes lost {clobbered:?}"
        );
        assert_eq!(
            word.load(Ordering::Acquire),
            u64::from(N),
            "prewarm {prewarm}: the last bottom-of-pipe write"
        );
        assert_eq!(
            *order.lock().unwrap(),
            (0..N).collect::<Vec<_>>(),
            "prewarm {prewarm}: work for after in order"
        );
    }
}

/// A flight keeps its native calls split before a deferred UAV-to-SRV
/// prelude (VIctory-macOS #242), and every mark, the split ones too, shows
/// it ran.
#[test]
fn a_flight_keeps_its_split_native_calls() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let words = texture_words(16, 16, 41, 0, 4);
    let texture = texture_at_address(title_memory(1 << 16), &words);
    let lists: Vec<Arc<HostList>> = (0..2).map(|_| Arc::new(host.new_list().unwrap())).collect();
    let _ = lists[0].transition(texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
    let _ = lists[1].transition(
        texture,
        (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0) as u32,
    );
    lists.iter().for_each(|l| l.close());
    let mode = ModeGuard::new(true, false);
    crate::host::tests::SPLIT_PRELUDES.with(|s| s.set(true));
    let lost = crate::host::tests::lost_so_far();
    crate::host::tests::start_native_call_trace();
    host.execute_flight1(lists.clone(), vec![false, false]);
    host.finish_flight_now();
    let groups = crate::host::tests::take_native_call_trace();
    crate::host::tests::SPLIT_PRELUDES.with(|s| s.set(false));
    drop(mode);
    assert_eq!(groups.len(), 1, "one submission: {groups:?}");
    assert!(groups[0].1, "its fence signalled");
    assert_eq!(groups[0].0.len(), 2, "two native calls: {groups:?}");
    assert_eq!(
        crate::host::tests::lost_so_far(),
        lost,
        "every mark saw the flight"
    );
}
