//! Native Metal work beside D3DMetal's (native Metal host, milestone 2):
//! one buffer both reach, D3D12's as VIctory makes its host copies (a
//! custom heap, write-back CPU pages in the GPU's pool, mapped) and the
//! bridge's a no-copy buffer over the same mapped pages; the work ordered
//! across the two queues, first by waits on the CPU, then without blocking
//! the submitting thread (helper threads that hand values from one side's
//! fence to the other's). Every word is checked; the ignored test measures
//! what each handoff costs.
//!
//! `tools/macos/test-wine.sh -p victory-metal --release --test beside_d3d12 -- --ignored --nocapture`
#![cfg(windows)]

use std::cell::Cell;
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use victory_metal::{Bridge, Op, OpenError};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

mod d3d12;

const BYTES: usize = 1 << 20;
const WORDS: u32 = (BYTES / 4) as u32;

// Word k = value + k, as the bridge's ops write them.
const FILL: &str = "cbuffer c : register(b0) { uint value; uint words; };\n\
    RWByteAddressBuffer o : register(u0);\n\
    [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
        if (id.x < words) o.Store(id.x * 4, value + id.x);\n\
    }\n";

fn bridge() -> Option<&'static Bridge> {
    match Bridge::open() {
        Ok(b) => Some(Box::leak(Box::new(b))),
        Err(e @ (OpenError::NotWine | OpenError::NotAsked)) => {
            eprintln!("no Metal bridge here ({e}): skipped");
            None
        }
        Err(e) => panic!("the Metal bridge was asked for but did not open: {e}"),
    }
}

type Pair = (ID3D12CommandAllocator, ID3D12GraphicsCommandList, Cell<u64>);

/// D3D12 (D3DMetal) with one shared buffer, its fill pipeline and a fence.
struct Side {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    /// Two allocators and lists in turn, each with the fence value of its
    /// last use: a list is reset only once that is done.
    lists: [Pair; 2],
    turn: Cell<usize>,
    fence: ID3D12Fence,
    event: HANDLE,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    shared: ID3D12Resource,
    /// The shared buffer's CPU address (mapped once, kept).
    data: *mut u32,
    back: ID3D12Resource,
    value: Cell<u64>,
}

impl Side {
    fn new() -> Self {
        // SAFETY: plain D3D12 setup; every out pointer is ours.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
            let device = device.unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let pair = || -> Pair {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                list.Close().unwrap();
                (allocator, list, Cell::new(0))
            };
            let lists = [pair(), pair()];
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 2,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let root = d3d12::root_signature(&device, &params, D3D12_ROOT_SIGNATURE_FLAG_NONE);
            let pso = d3d12::compute_pipeline(&device, &root, FILL);
            // As VIctory's host copies: write-back CPU pages in the GPU's
            // own pool (`alias.rs`, `host.rs`).
            let shared = buffer(
                &device,
                D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                },
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            let mut p = std::ptr::null_mut();
            shared.Map(0, None, Some(&mut p)).unwrap();
            let back = buffer(
                &device,
                D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            Self {
                device,
                queue,
                lists,
                turn: Cell::new(0),
                fence,
                event,
                root,
                pso,
                shared,
                data: p.cast(),
                back,
                value: Cell::new(0),
            }
        }
    }

    // Mapped memory, not Rust-owned; the tests use one borrow at a time.
    #[allow(clippy::mut_from_ref)]
    fn words(&self) -> &mut [u32] {
        // SAFETY: the mapped shared buffer, BYTES long.
        unsafe { std::slice::from_raw_parts_mut(self.data, BYTES / 4) }
    }

    /// Records a list and executes it; with `wait_first`, the queue waits
    /// for that fence value first (on the GPU, not the CPU). Signals and
    /// returns the next fence value, not waited for.
    fn execute(
        &self,
        wait_first: Option<(&ID3D12Fence, u64)>,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> u64 {
        let turn = self.turn.get();
        self.turn.set(1 - turn);
        let (allocator, list, last) = &self.lists[turn];
        // The list's last submission has to be done before its reset.
        self.wait_cpu(last.get());
        // SAFETY: that submission is done, so the allocator and list are
        // free; the resources outlive the submission.
        unsafe {
            allocator.Reset().unwrap();
            list.Reset(allocator, None).unwrap();
            record(list);
            list.Close().unwrap();
            if let Some((fence, value)) = wait_first {
                self.queue.Wait(fence, value).unwrap();
            }
            let executable: ID3D12CommandList = list.cast().unwrap();
            self.queue.ExecuteCommandLists(&[Some(executable)]);
            let value = self.value.get() + 1;
            self.value.set(value);
            self.queue.Signal(&self.fence, value).unwrap();
            last.set(value);
            value
        }
    }

    fn wait_cpu(&self, value: u64) {
        // SAFETY: our fence and event.
        unsafe {
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
            }
        }
    }

    fn fill(&self, list: &ID3D12GraphicsCommandList, value: u32) {
        // SAFETY: our pipeline, root signature and buffer.
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstants(0, 2, [value, WORDS].as_ptr().cast(), 0);
            list.SetComputeRootUnorderedAccessView(1, self.shared.GetGPUVirtualAddress());
            list.Dispatch(WORDS.div_ceil(64), 1, 1);
        }
    }

    fn copy_back(&self, list: &ID3D12GraphicsCommandList) {
        let to_copy = transition(
            &self.shared,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let to_uav = transition(
            &self.shared,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: both buffers are BYTES long; the barriers name ours.
        unsafe {
            list.ResourceBarrier(&[to_copy]);
            list.CopyBufferRegion(&self.back, 0, &self.shared, 0, BYTES as u64);
            list.ResourceBarrier(&[to_uav]);
        }
    }

    /// The readback buffer's first wrong word, if any.
    fn back_wrong(&self, value: u32) -> Option<(usize, u32)> {
        // SAFETY: a readback buffer, mapped after its copy was waited for.
        unsafe {
            let mut p = std::ptr::null_mut();
            self.back.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p.cast::<u32>(), BYTES / 4);
            let wrong = first_wrong(got, value);
            self.back.Unmap(0, None);
            wrong
        }
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer for the barrier's call; the
                // barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: a plain buffer description; the out pointer is ours.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: BYTES as u64,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut resource,
            )
            .unwrap();
    }
    resource.unwrap()
}

fn first_wrong(words: &[u32], value: u32) -> Option<(usize, u32)> {
    words
        .iter()
        .enumerate()
        .find(|&(k, &w)| w != value.wrapping_add(k as u32))
        .map(|(k, &w)| (k, w))
}

/// The bridge's buffer over D3D12's mapped one.
fn wrap_shared(bridge: &Bridge, side: &Side) -> u64 {
    let at = side.data as usize;
    eprintln!(
        "D3D12's mapped buffer starts at {at:#x} (16 KiB aligned: {})",
        at.is_multiple_of(16 << 10)
    );
    bridge
        .wrap(side.data.cast(), BYTES, false)
        .expect("a no-copy buffer over D3D12's mapped buffer")
        .handle
}

/// A thread that hands values from a D3D12 fence to the bridge's event:
/// for each `(fence value, event value)` it gets, it waits for the fence
/// and sets the event. Ends when the sender is dropped.
fn fence_to_event(
    bridge: &'static Bridge,
    fence: ID3D12Fence,
    event: u64,
) -> (Sender<(u64, u64)>, JoinHandle<()>) {
    let (tx, rx) = channel::<(u64, u64)>();
    let fence_addr = fence.into_raw() as usize;
    let thread = std::thread::spawn(move || {
        // SAFETY: the pointer `into_raw` gave, taken back once.
        let fence = unsafe { ID3D12Fence::from_raw(fence_addr as *mut _) };
        // SAFETY: a plain event.
        let wake = unsafe { CreateEventW(None, false, false, None) }.unwrap();
        for (from, to) in rx {
            // SAFETY: our fence and event.
            unsafe {
                if fence.GetCompletedValue() < from {
                    fence.SetEventOnCompletion(from, wake).unwrap();
                    WaitForSingleObject(wake, INFINITE);
                }
            }
            bridge.signal(event, to).expect("signal");
        }
    });
    (tx, thread)
}

/// A thread that hands values from the bridge's event to a D3D12 fence:
/// for each `(event value, fence value)`, it waits for the event (spinning
/// in the unixlib) and signals the fence from the CPU.
fn event_to_fence(
    bridge: &'static Bridge,
    event: u64,
    fence: ID3D12Fence,
) -> (Sender<(u64, u64)>, JoinHandle<()>) {
    let (tx, rx) = channel::<(u64, u64)>();
    let fence_addr = fence.into_raw() as usize;
    let thread = std::thread::spawn(move || {
        // SAFETY: the pointer `into_raw` gave, taken back once.
        let fence = unsafe { ID3D12Fence::from_raw(fence_addr as *mut _) };
        for (from, to) in rx {
            bridge
                .wait_for(event, from, Duration::from_secs(10))
                .expect("the GPU's signal");
            // SAFETY: our fence.
            unsafe { fence.Signal(to) }.unwrap();
        }
    });
    (tx, thread)
}

#[test]
fn native_and_d3dmetal_work_on_one_buffer_ordered_by_cpu_waits() {
    let Some(bridge) = bridge() else { return };
    let side = Side::new();
    let b = wrap_shared(bridge, &side);
    // D3D12 writes, the CPU waits for its fence, Metal reads.
    let done = side.execute(None, |l| side.fill(l, 0x1000_0000));
    side.wait_cpu(done);
    let mut check = [Op::check(b, 0, WORDS, 0x1000_0000)];
    bridge.run(&mut check).expect("native check");
    assert_eq!(
        check[0].mismatches(),
        0,
        "Metal read D3D12's write: first wrong {:?}",
        check[0].first_bad()
    );
    // Metal writes and is waited for, D3D12 reads (a copy to a readback
    // buffer).
    bridge
        .run(&mut [Op::write(b, 0, WORDS, 0x2000_0000)])
        .expect("native write");
    let done = side.execute(None, |l| side.copy_back(l));
    side.wait_cpu(done);
    assert_eq!(
        side.back_wrong(0x2000_0000),
        None,
        "D3D12 read Metal's write"
    );
    // And the CPU sees the last write in the mapped pages.
    assert_eq!(first_wrong(side.words(), 0x2000_0000), None);
    bridge.release(b).expect("release");
}

#[test]
fn native_and_d3dmetal_work_ordered_on_the_gpu_without_waiting_on_the_submitting_thread() {
    let Some(bridge) = bridge() else { return };
    let side = Side::new();
    let b = wrap_shared(bridge, &side);
    let event = bridge.event().expect("event");
    // SAFETY: plain fence creation.
    let handoff: ID3D12Fence =
        unsafe { side.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    let (to_event, to_event_thread) = fence_to_event(bridge, side.fence.clone(), event);
    let (to_fence, to_fence_thread) = event_to_fence(bridge, event, handoff.clone());
    // D3D12 -> Metal: Metal's run is committed first and waits on the GPU
    // for the event the helper sets once D3D12's fill is done. Metal ->
    // D3D12: D3D12's copy waits on the GPU for the fence the other helper
    // signals once Metal's run is.
    let mut ops = [
        Op::wait_event(event, 1),
        Op::check(b, 0, WORDS, 0x3000_0000),
        Op::write(b, 0, WORDS, 0x4000_0000),
        Op::signal_event(event, 2),
    ];
    let run = bridge.start(&mut ops).expect("start the native run");
    let filled = side.execute(None, |l| side.fill(l, 0x3000_0000));
    to_event.send((filled, 1)).unwrap();
    to_fence.send((2, 1)).unwrap();
    let read = side.execute(Some((&handoff, 1)), |l| side.copy_back(l));
    // Only now does the submitting thread wait, for the last of it.
    side.wait_cpu(read);
    run.wait().expect("the native run");
    drop((to_event, to_fence));
    to_event_thread.join().unwrap();
    to_fence_thread.join().unwrap();
    assert_eq!(
        ops[1].mismatches(),
        0,
        "Metal read D3D12's fill: first wrong {:?}",
        ops[1].first_bad()
    );
    assert_eq!(
        side.back_wrong(0x4000_0000),
        None,
        "D3D12 read Metal's write"
    );
    bridge.release(event).expect("release");
    bridge.release(b).expect("release");
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[test]
#[ignore = "a measurement: run alone with --nocapture"]
fn handoff_costs() {
    let Some(bridge) = bridge() else { return };
    let side = Side::new();
    let b = wrap_shared(bridge, &side);
    const N: usize = 40;
    // The parts alone: an empty D3D12 list waited for; a tiny native run.
    let mut d3d = Vec::new();
    let mut native = Vec::new();
    let (mut encode, mut wait, mut gpu) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..N {
        let t = Instant::now();
        let v = side.execute(None, |_| {});
        side.wait_cpu(v);
        d3d.push(t.elapsed());
        let t = Instant::now();
        let times = bridge.run(&mut [Op::write(b, 0, 64, 1)]).expect("tiny run");
        native.push(t.elapsed());
        encode.push(times.encode);
        wait.push(times.wait);
        gpu.push(times.gpu);
    }
    let empty = bridge.submit_empty(N as u32).expect("empty") / N as u32;
    eprintln!(
        "  bridge: empty command buffer {empty:?}; tiny run encode {:?}, wait {:?}, GPU {:?}",
        median(encode),
        median(wait),
        median(gpu)
    );
    // D3D12 then Metal, ordered by a CPU wait on the submitting thread.
    let mut cpu_d2m = Vec::new();
    for i in 0..N {
        let t = Instant::now();
        let v = side.execute(None, |l| side.fill(l, i as u32));
        side.wait_cpu(v);
        let mut c = [Op::check(b, 0, WORDS, i as u32)];
        bridge.run(&mut c).expect("check");
        cpu_d2m.push(t.elapsed());
        assert_eq!(c[0].mismatches(), 0);
    }
    // Metal then D3D12, ordered by a CPU wait.
    let mut cpu_m2d = Vec::new();
    for i in 0..N {
        let t = Instant::now();
        bridge
            .run(&mut [Op::write(b, 0, WORDS, 0x100 + i as u32)])
            .expect("write");
        let v = side.execute(None, |l| side.copy_back(l));
        side.wait_cpu(v);
        cpu_m2d.push(t.elapsed());
        assert_eq!(side.back_wrong(0x100 + i as u32), None);
    }
    // D3D12 -> Metal -> D3D12 through helper threads; the submitting
    // thread only commits and executes, then waits for the end.
    let event = bridge.event().expect("event");
    // SAFETY: plain fence creation.
    let handoff: ID3D12Fence =
        unsafe { side.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    let (to_event, to_event_thread) = fence_to_event(bridge, side.fence.clone(), event);
    let (to_fence, to_fence_thread) = event_to_fence(bridge, event, handoff.clone());
    let mut chain = Vec::new();
    let mut submitting = Vec::new();
    for i in 0..N as u64 {
        let mut ops = [
            Op::wait_event(event, 2 * i + 1),
            Op::check(b, 0, WORDS, 0x5000 + i as u32),
            Op::write(b, 0, WORDS, 0x6000 + i as u32),
            Op::signal_event(event, 2 * i + 2),
        ];
        let t = Instant::now();
        let run = bridge.start(&mut ops).expect("start");
        let filled = side.execute(None, |l| side.fill(l, 0x5000 + i as u32));
        to_event.send((filled, 2 * i + 1)).unwrap();
        to_fence.send((2 * i + 2, i + 1)).unwrap();
        let read = side.execute(Some((&handoff, i + 1)), |l| side.copy_back(l));
        submitting.push(t.elapsed());
        side.wait_cpu(read);
        run.wait().expect("run");
        chain.push(t.elapsed());
        assert_eq!(ops[1].mismatches(), 0, "round {i}");
        assert_eq!(side.back_wrong(0x6000 + i as u32), None, "round {i}");
    }
    drop((to_event, to_fence));
    to_event_thread.join().unwrap();
    to_fence_thread.join().unwrap();
    eprintln!("medians of {N}, 1 MiB shared buffer:");
    eprintln!(
        "  empty D3D12 list, waited:               {:?}",
        median(d3d)
    );
    eprintln!(
        "  tiny native run, waited:                {:?}",
        median(native)
    );
    eprintln!(
        "  D3D12 fill -> CPU wait -> native check: {:?}",
        median(cpu_d2m)
    );
    eprintln!(
        "  native write -> D3D12 copy, CPU waits:  {:?}",
        median(cpu_m2d)
    );
    eprintln!(
        "  D3D12 fill -> native check + write -> D3D12 copy through helper threads: \
         submitting thread busy {:?}, end to end {:?}",
        median(submitting),
        median(chain)
    );
    bridge.release(event).expect("release");
    bridge.release(b).expect("release");
}
