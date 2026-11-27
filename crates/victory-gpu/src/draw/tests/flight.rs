//! Title submissions as Flight-1 flights (`VICTORY_FLIGHT1`, with and
//! without `VICTORY_FLIGHT1_PREWARM`, `crate::host::flight`): what their
//! shaders write reaches title memory before anything the title waits on
//! says they ran, in order, with nothing lost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::*;
use crate::host::flight::{Gate, LanderHeld, ModeGuard, TEST_FLIGHTS, TEST_PREWARM};
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
                let value = crate::fence::value(fence);
                if gate.as_ref().is_some_and(Gate::held) {
                    assert_eq!(value, 0, "the Signal stands behind the held flight");
                }
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
    // Landed here (the native calls are told to this thread's trace when
    // the flight lands), unless another test's submission lands it first.
    let held = LanderHeld::on();
    host.execute_flight1(lists.clone(), vec![false, false]);
    host.finish_flight_now();
    drop(held);
    let groups = crate::host::tests::take_native_call_trace();
    crate::host::tests::SPLIT_PRELUDES.with(|s| s.set(false));
    drop(mode);
    if !groups.is_empty() {
        assert_eq!(groups.len(), 1, "one submission: {groups:?}");
        assert!(groups[0].1, "its fence signalled");
        assert_eq!(groups[0].0.len(), 2, "two native calls: {groups:?}");
    }
    assert_eq!(
        crate::host::tests::lost_so_far(),
        lost,
        "every mark saw the flight"
    );
}

/// `VICTORY_FLIGHTS=2` on for this thread until dropped.
struct TwoFlights;

impl TwoFlights {
    fn on() -> TwoFlights {
        TEST_FLIGHTS.with(|n| n.set(2));
        TwoFlights
    }
}

impl Drop for TwoFlights {
    fn drop(&mut self) {
        TEST_FLIGHTS.with(|n| n.set(0));
    }
}

/// Two flights in the air at once (the queue held by a gate): they land in
/// order, each one's copy-back before its own Signal and before the next
/// one's, as a title thread polling memory sees.
#[test]
fn two_flights_go_up_together_and_land_in_order() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let rig = Rig::new(2);
    let _mode = ModeGuard::new(true, false);
    let _two = TwoFlights::on();
    let (queue, fence) = (title_queue(), title_fence());
    let word = Box::new(AtomicU64::new(0));
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watched = {
        let (stop, at) = (stop.clone(), fence as usize);
        let (a, b) = (rig.at(0, 8) as usize, rig.at(1, 8) as usize);
        std::thread::spawn(move || {
            // SAFETY: dwords of the buffers' title memory, alive until the end.
            let dword = |at: usize| unsafe { (at as *const u32).read_volatile() };
            let mut seen = vec![(0, 0, 0)];
            while !stop.load(Ordering::Acquire) {
                let v = crate::fence::value(at as This);
                if seen.last().map(|s| s.0) != Some(v) {
                    seen.push((v, dword(a), dword(b)));
                }
            }
            seen.push((crate::fence::value(at as This), dword(a), dword(b)));
            seen
        })
    };
    let mut gate = Gate::hold(host);
    for (k, slot) in [0u32, 1].into_iter().enumerate() {
        let list = rig.list(&[(slot, 8, 0xE00 + slot)]);
        list.defer_write(
            word.as_ptr().cast(),
            crate::queue::Written::Value(k as u64 + 1),
            8,
        );
        let log = order.clone();
        list.defer_after(move || log.lock().unwrap().push(k));
        submit(&list);
        assert_eq!(
            crate::queue::signal_for_test(queue, fence, k as u64 + 1),
            S_OK
        );
    }
    let (in_air, value) = (host.in_air(), crate::fence::value(fence));
    if gate.held() {
        assert_eq!(in_air, 2, "two flights in the air");
        assert_eq!(value, 0, "both Signals stand behind them");
    }
    gate.open();
    host.finish_flight_now();
    poll("the second Signal", || crate::fence::value(fence) == 2);
    stop.store(true, Ordering::Release);
    let seen = watched.join().unwrap();
    assert_eq!(*order.lock().unwrap(), [0, 1], "work for after in order");
    assert_eq!(
        word.load(Ordering::Acquire),
        2,
        "the second's bottom-of-pipe write is the last"
    );
    assert!(
        seen.windows(2).all(|w| w[0].0 <= w[1].0),
        "the fence went back: {seen:?}"
    );
    assert!(
        seen.iter()
            .all(|&(v, a, b)| (v < 1 || a == 0xE00) && (v < 2 || b == 0xE01)),
        "a fence moved before its flight's copy-back: {seen:?}"
    );
    release(queue);
    release(fence);
}

/// A flight that writes a copy the flight in the air writes too does not
/// go beside it: that one lands first (the gate opens meanwhile), and both
/// writes land.
#[test]
fn a_flight_writing_what_the_one_in_the_air_writes_waits_for_it() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let rig = Rig::new(1);
    let _mode = ModeGuard::new(true, false);
    let _two = TwoFlights::on();
    let mut gate = Gate::hold(host);
    let beside = crate::host::flight::beside_counts()[0];
    submit(&rig.list(&[(0, 0, 0xF01)]));
    let opener = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        gate.open();
    });
    submit(&rig.list(&[(0, 4, 0xF02)]));
    opener.join().unwrap();
    // (Whether the first was still in the air, or another test's
    // submission landed it first, the second never goes beside it.)
    assert_eq!(
        crate::host::flight::beside_counts()[0],
        beside,
        "it went up alone"
    );
    host.finish_flight_now();
    assert_eq!((rig.title(0, 0), rig.title(0, 4)), (0xF01, 0xF02));
}

/// A flight in the air has run on the GPU but is not landed yet; the title
/// writes another dword of a copy it wrote, and the next submission reads
/// both: its refresh of that copy waits until the first is landed (else the
/// title's older bytes would go over what the first's shader wrote, and
/// that write would be lost), then reads the title's write and the
/// shader's.
#[test]
fn a_refresh_of_what_the_flight_in_the_air_writes_waits_for_its_landing() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let mut rig = Rig::new(0);
    // Slots 0 and 1 view the same write-watched buffer (for writing, for
    // reading); slot 2 the buffer the next one copies into.
    let pool = crate::watch::tests::pool(1);
    let shared = buffer_at_address(pool as u64, crate::watch::PAGE as u64);
    rig.view(0, shared);
    rig.view_srv(1, shared);
    rig.view(2, buffer(256));
    let _mode = ModeGuard::new(true, false);
    let _two = TwoFlights::on();
    let held = LanderHeld::on();
    submit(&rig.list(&[(0, 4, 0x1234)]));
    // Its shader has run (a tiny dispatch), and nothing of ours lands it
    // (another test's submission may: then this checks only the results).
    std::thread::sleep(Duration::from_millis(50));
    rig.write(1, 0, 0x5678);
    let first = rig.copying(1, 2, 0);
    let second = rig.copying(1, 2, 4);
    first.close();
    second.close();
    host.execute_flight1(vec![first, second], vec![false, false]);
    drop(held);
    host.finish_flight_now();
    assert_eq!(
        rig.title(0, 4),
        0x1234,
        "the first flight's write is in title memory"
    );
    assert_eq!(rig.title(0, 0), 0x5678, "and the title's own");
    assert_eq!(
        (rig.title(2, 0), rig.title(2, 4)),
        (0x5678, 0x1234),
        "the next one read both"
    );
}

/// The title writes a view into a heap slot while two flights that bind
/// the heap are in the air; the next flight writes through it, and every
/// write lands.
#[test]
fn a_heap_slot_written_while_two_flights_are_in_the_air_serves_the_next() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let mut rig = Rig::new(2);
    let _ = host.log_messages();
    let _mode = ModeGuard::new(true, false);
    let _two = TwoFlights::on();
    let mut gate = Gate::hold(host);
    submit(&rig.list(&[(0, 0, 0xD11)]));
    submit(&rig.list(&[(1, 0, 0xD12)]));
    let in_air = host.in_air();
    if gate.held() {
        assert_eq!(in_air, 2, "two flights in the air");
    }
    rig.view(2, buffer(256));
    gate.open();
    submit(&rig.list(&[(2, 0, 0xD13)]));
    host.finish_flight_now();
    assert_eq!(
        (rig.title(0, 0), rig.title(1, 0), rig.title(2, 0)),
        (0xD11, 0xD12, 0xD13)
    );
    assert_eq!(host.log_messages(), None);
    assert!(!crate::host::device_removed());
}

/// Many submissions back to back with two flights allowed: they alternate
/// between two buffers (each goes beside the one before) and now and then
/// write the same one twice in a row (then the one in the air lands
/// first), with bottom-of-pipe counts and Signals between them: nothing is
/// lost, and their work for after and Signals come in order.
#[test]
fn back_to_back_with_two_flights_lose_no_writes() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const N: u32 = 64;
    let rig = Rig::new(2);
    let _mode = ModeGuard::new(true, false);
    let _two = TwoFlights::on();
    let (queue, fence) = (title_queue(), title_fence());
    let word = Box::new(AtomicU64::new(0));
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let [beside, _, _] = crate::host::flight::beside_counts();
    for i in 0..N {
        let slot = if i % 8 == 7 { (i + 1) % 2 } else { i % 2 };
        let list = rig.list(&[(slot, 4 * (i / 2 % 32), 0x4000 + i)]);
        list.defer_write(
            word.as_ptr().cast(),
            crate::queue::Written::Value(u64::from(i) + 1),
            8,
        );
        let log = order.clone();
        list.defer_after(move || log.lock().unwrap().push(i));
        submit(&list);
        if i % 4 == 3 {
            assert_eq!(
                crate::queue::signal_for_test(queue, fence, u64::from(i / 4) + 1),
                S_OK
            );
        }
    }
    host.finish_flight_now();
    let went = crate::host::flight::beside_counts()[0] - beside;
    let mut want = std::collections::HashMap::new();
    for i in 0..N {
        let slot = if i % 8 == 7 { (i + 1) % 2 } else { i % 2 };
        want.insert((slot, 4 * (i / 2 % 32)), 0x4000 + i);
    }
    let lost: Vec<_> = want
        .iter()
        .filter(|&(&(slot, at), &v)| rig.title(slot, at) != v)
        .collect();
    assert!(
        lost.is_empty(),
        "writes lost: {lost:?} ({went} went beside)"
    );
    assert_eq!(word.load(Ordering::Acquire), u64::from(N));
    assert_eq!(*order.lock().unwrap(), (0..N).collect::<Vec<_>>());
    assert_eq!(crate::fence::value(fence), u64::from(N / 4));
    release(queue);
    release(fence);
}

/// A shader writes a dword through a heap's unordered view (marked, and
/// declared by no list: the bindless path), and while its flight is in the
/// air the title writes another dword of the same 64 KiB page from the CPU,
/// as its render thread goes on writing a frame's data with Flight-1: both
/// writes are in title memory once the flight has landed, as on the
/// console, where memory is written byte by byte.
#[test]
fn a_heap_view_write_and_a_title_write_on_one_page_during_a_flight_both_land() {
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
    let p = crate::watch::tests::pool(1);
    // SAFETY: a committed write-watched page of ours.
    unsafe {
        (p as *mut u32).write_volatile(0x1111_1111);
        ((p + 16) as *mut u32).write_volatile(0x2222_2222);
    }
    let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
    let copy = crate::shadow::find(p as u64, crate::watch::PAGE as u64).expect("the page's copy");
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
    let mut gate = Gate::hold(host);
    let list = Arc::new(host.new_list().unwrap());
    // The heap's views brought up to date (its unordered views included),
    // as for a list whose shaders index the heap: no write declared.
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
    // The title's render thread goes on, and writes the page meanwhile.
    // SAFETY: a dword of our page.
    unsafe { ((p + 16) as *mut u32).write_volatile(0xCAFE_BABE) };
    gate.open();
    host.finish_flight_now();
    // SAFETY: dwords of our page.
    let (gpu, cpu) = unsafe {
        (
            (p as *const u32).read_volatile(),
            ((p + 16) as *const u32).read_volatile(),
        )
    };
    assert_eq!(cpu, 0xCAFE_BABE, "the title's write");
    assert_eq!(
        gpu, 0xDEAD_BEEF,
        "the shader's write, on the page the title wrote during the flight"
    );
    crate::shadow::unregister(buf as usize, p as u64);
}

/// Bench (`--ignored --nocapture`): a title that submits and then reads its
/// fence from memory until the Signal behind the submission moved it, as
/// Grand Theft Auto VI's render thread does, with the submission waited
/// for (`VICTORY_FLIGHT1` off) and as a flight `victory-flight` lands: the
/// time from submitting to the fence moved, per submission. And what a host
/// list costs made new and reset after it ran (a title list reset while its
/// host list is in the air takes a spare one, or one made new).
#[test]
#[ignore = "bench: prints what a flight costs the title"]
fn what_a_flight_costs_the_title() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const N: u32 = 200;
    let mut rig = Rig::new(0);
    for slot in 0..2 {
        let pool = crate::watch::tests::pool(1);
        let buffer = buffer_at_address(pool as u64, crate::watch::PAGE as u64);
        if slot == 0 {
            rig.view_srv(slot, buffer);
        } else {
            rig.view(slot, buffer);
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
    let (queue, fence) = (title_queue(), title_fence());
    let median = |mut t: Vec<Duration>| {
        t.sort();
        t[t.len() / 2]
    };
    let mut value = 0;
    for _ in 0..2 {
        for flight in [false, true] {
            let _mode = ModeGuard::new(flight, false);
            let (mut turn, mut gpu) = (Vec::new(), Vec::new());
            for i in 0..N {
                rig.write(0, 4 * (i % 32), i);
                let list = rig.copying(0, 1, 4 * (i % 32));
                value += 1;
                let since = Instant::now();
                submit(&list);
                let back = since.elapsed();
                assert_eq!(crate::queue::signal_for_test(queue, fence, value), S_OK);
                while crate::fence::value(fence) < value {
                    std::hint::spin_loop();
                }
                turn.push(since.elapsed());
                gpu.push(back);
            }
            host.finish_flight_now();
            println!(
                "flight {flight}: submit-to-fence median {:?} (submission returned after {:?})",
                median(turn),
                median(gpu)
            );
        }
    }
    let t = Instant::now();
    let lists: Vec<Arc<HostList>> = (0..64)
        .map(|_| Arc::new(host.new_list().expect("host list")))
        .collect();
    let made = t.elapsed() / 64;
    for list in &lists {
        list.close();
    }
    let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
    host.execute_sync(&runs, &[]);
    let t = Instant::now();
    for list in &lists {
        list.reset();
    }
    println!(
        "a host list made new {made:?}, reset after a run {:?}",
        t.elapsed() / 64
    );
    release(fence);
}
