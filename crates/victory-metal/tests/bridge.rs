//! The bridge under Wine: Metal through the unixlib, on memory the Windows
//! side allocated, with no copy. tools/macos/test-wine.sh builds the bridge
//! and sets it up; elsewhere (Windows, or Wine without the bridge) these
//! tests say so and pass.
//!
//! `bridge_against_d3d12` (ignored) compares the bridge's costs with the
//! same work through D3D12, which is D3DMetal under Wine:
//!
//! `tools/macos/test-wine.sh -p victory-metal --release --test bridge -- --ignored --nocapture`
#![cfg(windows)]

use std::time::{Duration, Instant};

use victory_metal::{Bridge, OpenError};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

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

/// Committed read-write memory from `VirtualAlloc`, as the title's own
/// allocations are made, freed on drop.
struct Region {
    base: *mut u32,
    bytes: usize,
}

impl Region {
    fn new(bytes: usize) -> Self {
        // SAFETY: a fresh reservation; nothing else knows its address.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null(), "VirtualAlloc of {bytes} bytes failed");
        Self {
            base: base.cast(),
            bytes,
        }
    }

    fn words(&mut self) -> &mut [u32] {
        // SAFETY: committed, read-write, ours, and borrowed through `self`.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.bytes / 4) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the reservation made in `new`; no borrow outlives `self`.
        unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
    }
}

#[test]
fn device_is_apple_silicon_with_unified_memory() {
    let Some(bridge) = bridge() else { return };
    let device = bridge.device().expect("the unixlib made no Metal device");
    assert!(device.unified_memory, "{device:?}");
    assert!(device.registry_id != 0, "{device:?}");
    assert!(device.max_buffer_length >= 1 << 30, "{device:?}");
    // Rosetta's pages, which the no-copy buffers below start on.
    assert_eq!(device.page_size, 4096, "{device:?}");
}

#[test]
fn gpu_writes_virtualalloc_memory_in_place() {
    let Some(bridge) = bridge() else { return };
    // 1 MiB for the GPU, then 64 KiB it must not touch.
    const GPU: usize = 1 << 20;
    let mut region = Region::new(GPU + (64 << 10));
    let words = region.words();
    words.fill(0xdead_beef);
    let (gpu, rest) = words.split_at_mut(GPU / 4);
    let times = bridge.fill(gpu, 64, 0x1000).expect("fill");
    for (k, &w) in gpu.iter().enumerate() {
        assert_eq!(
            w,
            0x1000 + k as u32,
            "word {k} after the GPU wrote the memory in place"
        );
    }
    assert!(
        rest.iter().all(|&w| w == 0xdead_beef),
        "the GPU wrote past its range"
    );
    assert!(times.gpu > Duration::ZERO, "{times:?}");

    // The CPU's own stores in between are what the GPU sees next: no copy
    // in either direction, and nothing stale.
    gpu.fill(7);
    bridge
        .fill(&mut gpu[..1024], 1, 0x50_0000)
        .expect("second fill");
    assert!(
        gpu[..1024]
            .iter()
            .enumerate()
            .all(|(k, &w)| w == 0x50_0000 + k as u32)
    );
    assert!(gpu[1024..].iter().all(|&w| w == 7));
}

#[test]
fn fill_refuses_memory_off_page_boundaries() {
    let Some(bridge) = bridge() else { return };
    let mut region = Region::new(64 << 10);
    let words = region.words();
    let e = bridge
        .fill(&mut words[1..1025], 1, 0)
        .expect_err("an unaligned start");
    assert!(e.message.contains("whole pages"), "{e}");
    let e = bridge
        .fill(&mut words[..1024], 3, 0)
        .expect_err("an uneven split");
    assert!(e.message.contains("split evenly"), "{e}");
}

#[test]
fn empty_submissions_complete() {
    let Some(bridge) = bridge() else { return };
    let total = bridge.submit_empty(8).expect("empty submissions");
    assert!(total > Duration::ZERO);
}

/// A frame's shape on our M5 Pro: about 4,750 draws (here dispatches) and
/// 23 submissions, each waited for.
const DISPATCHES: u32 = 5000;
const SUBMISSIONS: u32 = 200;
const NOOPS: u32 = 1_000_000;

#[test]
#[ignore = "a benchmark: run it alone, with --nocapture"]
fn bridge_against_d3d12() {
    let Some(bridge) = bridge() else { return };
    let device = bridge.device().unwrap();
    println!("Metal device through the bridge: {device:?}");

    // The way in and back.
    for _ in 0..1000 {
        bridge.noop();
    }
    let t = Instant::now();
    for _ in 0..NOOPS {
        bridge.noop();
    }
    let noop = t.elapsed() / NOOPS;
    println!(
        "unix call (no-op):             {:>9.1} ns",
        noop.as_nanos() as f64
    );

    // Submissions that do nothing, each waited for.
    bridge.submit_empty(20).unwrap();
    let metal_empty = bridge.submit_empty(SUBMISSIONS).unwrap() / SUBMISSIONS;
    let d3d12 = d3d12::Bench::new();
    d3d12.submit_empty(20);
    let d3d12_empty = d3d12.submit_empty(SUBMISSIONS) / SUBMISSIONS;
    println!(
        "empty submission + wait:       Metal {:>7.1} us   D3D12/D3DMetal {:>7.1} us",
        us(metal_empty),
        us(d3d12_empty)
    );

    // DISPATCHES dispatches of 1,024 words each in one submission.
    let bytes = DISPATCHES as usize * 4096;
    let mut region = Region::new(bytes);
    let mut metal = Vec::new();
    for run in 0..6 {
        let t = Instant::now();
        let times = bridge.fill(region.words(), DISPATCHES, run).unwrap();
        let wall = t.elapsed();
        let words = region.words();
        assert!(words.iter().enumerate().all(|(k, &w)| w == run + k as u32));
        metal.push((wall, times));
    }
    let mut d3d = Vec::new();
    for run in 0..6 {
        d3d.push((
            d3d12.fill(bytes, DISPATCHES, run, false),
            d3d12.fill(bytes, DISPATCHES, run, true),
        ));
    }
    println!("{DISPATCHES} dispatches in one submission, run by run (the first compiles):");
    for (run, ((wall, m), (root, table))) in metal.iter().zip(&d3d).enumerate() {
        println!(
            "  run {run}: Metal wall {:>6.2} ms (wrap {:>5.3}, encode {:>5.2}, wait {:>5.2}, GPU {:>5.2})",
            ms(*wall),
            ms(m.wrap),
            ms(m.encode),
            ms(m.wait),
            ms(m.gpu),
        );
        for (how, d) in [("root UAV", root), ("view table", table)] {
            println!(
                "         D3D12/D3DMetal, {how:<10} wall {:>6.2} ms (record {:>5.2}, submit to fence {:>6.2})",
                ms(d.record + d.execute),
                ms(d.record),
                ms(d.execute),
            );
        }
    }
}

/// How a D3D12/D3DMetal submission's time grows with what its bound view
/// heap names: a title's frame binds one heap of millions of slots over
/// some 20,000 resources. A Metal host keeps its resources in a residency
/// set and its views in a buffer, so none of that is per submission.
#[test]
#[ignore = "a benchmark: run it alone, with --nocapture"]
fn d3d12_submission_against_heap_size() {
    if Bridge::open().is_err() {
        eprintln!("not under Wine with the bridge: skipped");
        return;
    }
    let d3d12 = d3d12::Bench::new();
    println!(
        "one submission of N dispatches through one view table, with the heap naming R buffers of 64 KiB:"
    );
    for (resources, slots) in [
        (0, 1),
        (1000, 1001),
        (10_000, 10_001),
        (20_000, 20_001),
        (0, 1_000_000),
    ] {
        let (one, many) = d3d12.heap_submission(resources, slots);
        println!(
            "  R {resources:>6}, slots {slots:>8}: N=1 {:>7.3} ms   N=5000 {:>7.3} ms (median submit to fence)",
            ms(one),
            ms(many)
        );
    }
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The same work through D3D12: a compute pipeline from HLSL (Wine's
/// compiler), root constants and a root UAV, as VIctory's own passes do.
mod d3d12 {
    use std::time::{Duration, Instant};

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    // A byte-address buffer: Wine's compiler has no stores through
    // structured buffers.
    const SOURCE: &str = "cbuffer c : register(b0) { uint value; uint first; uint words; };\n\
        RWByteAddressBuffer o : register(u0);\n\
        [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
            if (id.x < words) o.Store((first + id.x) * 4, value + first + id.x);\n\
        }\n";

    pub(super) struct Bench {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        event: HANDLE,
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        /// The same with the UAV in a view table instead of the root.
        root_table: ID3D12RootSignature,
        pso_table: ID3D12PipelineState,
        done: std::cell::Cell<u64>,
    }

    pub(super) struct Fill {
        /// Recording the list, on this thread.
        pub record: Duration,
        /// From `ExecuteCommandLists` to the fence's event.
        pub execute: Duration,
    }

    impl Bench {
        pub(super) fn new() -> Self {
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
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                list.Close().unwrap();
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                let event = CreateEventW(None, false, false, None).unwrap();
                let (root, pso) = pipeline(&device, false);
                let (root_table, pso_table) = pipeline(&device, true);
                Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    event,
                    root,
                    pso,
                    root_table,
                    pso_table,
                    done: std::cell::Cell::new(0),
                }
            }
        }

        fn wait(&self) {
            let value = self.done.get() + 1;
            self.done.set(value);
            // SAFETY: our queue, fence and event.
            unsafe {
                self.queue.Signal(&self.fence, value).unwrap();
                if self.fence.GetCompletedValue() < value {
                    self.fence.SetEventOnCompletion(value, self.event).unwrap();
                    WaitForSingleObject(self.event, INFINITE);
                }
            }
        }

        fn run(&self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> Fill {
            let t = Instant::now();
            // SAFETY: the previous submission was waited for, so the
            // allocator and the list are free to reset.
            unsafe {
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
                record(&self.list);
                self.list.Close().unwrap();
            }
            let recorded = t.elapsed();
            let t = Instant::now();
            let list: ID3D12CommandList = self.list.cast().unwrap();
            // SAFETY: a closed list of our queue's type.
            unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
            self.wait();
            Fill {
                record: recorded,
                execute: t.elapsed(),
            }
        }

        /// `count` empty lists, each submitted alone and waited for.
        pub(super) fn submit_empty(&self, count: u32) -> Duration {
            let t = Instant::now();
            for _ in 0..count {
                self.run(|_| {});
            }
            t.elapsed()
        }

        /// The bridge's fill, into a GPU buffer, checked by a copy back.
        /// With `tables`, every dispatch binds a view table of its own from
        /// a shader-visible heap (as VIctory's draws bind theirs) in place
        /// of one root UAV for all.
        pub(super) fn fill(&self, bytes: usize, dispatches: u32, value: u32, tables: bool) -> Fill {
            let buffer = committed(
                &self.device,
                D3D12_HEAP_TYPE_DEFAULT,
                bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let words = (bytes / 4) as u32 / dispatches;
            let heap = tables.then(|| self.views(&buffer, bytes, dispatches));
            let fill = self.run(|l| {
                // SAFETY: our pipelines, their root signatures, the heap and
                // the buffer, which outlive the submission.
                unsafe {
                    match &heap {
                        Some(heap) => {
                            l.SetDescriptorHeaps(&[Some(heap.clone())]);
                            l.SetComputeRootSignature(&self.root_table);
                            l.SetPipelineState(&self.pso_table);
                        }
                        None => {
                            l.SetComputeRootSignature(&self.root);
                            l.SetPipelineState(&self.pso);
                            l.SetComputeRootUnorderedAccessView(1, buffer.GetGPUVirtualAddress());
                        }
                    }
                    let (base, step) = match &heap {
                        Some(heap) => (
                            heap.GetGPUDescriptorHandleForHeapStart().ptr,
                            self.device.GetDescriptorHandleIncrementSize(
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            ) as u64,
                        ),
                        None => (0, 0),
                    };
                    for i in 0..dispatches {
                        let constants = [value, i * words, words];
                        l.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                        if heap.is_some() {
                            l.SetComputeRootDescriptorTable(
                                1,
                                D3D12_GPU_DESCRIPTOR_HANDLE {
                                    ptr: base + step * u64::from(i),
                                },
                            );
                        }
                        l.Dispatch(words.div_ceil(64), 1, 1);
                    }
                }
            });
            let back = committed(
                &self.device,
                D3D12_HEAP_TYPE_READBACK,
                bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            // SAFETY: both buffers are `bytes` long.
            self.run(|l| unsafe { l.CopyBufferRegion(&back, 0, &buffer, 0, bytes as u64) });
            // SAFETY: a readback buffer, mapped whole after the copy was
            // waited for.
            unsafe {
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                let got = std::slice::from_raw_parts(p.cast::<u32>(), bytes / 4);
                assert!(
                    got.iter().enumerate().all(|(k, &w)| w == value + k as u32),
                    "D3D12's fill is wrong"
                );
                back.Unmap(0, None);
            }
            fill
        }
    }

    impl Bench {
        /// Median submit-to-fence times of one dispatch and of 5,000, each
        /// through a table at slot 0 of a heap of `slots` whose next
        /// `resources` slots name buffers of 64 KiB each (null views after
        /// them), bound for the submission.
        pub(super) fn heap_submission(&self, resources: usize, slots: u32) -> (Duration, Duration) {
            const BYTES: usize = 5000 * 4096;
            let target = committed(
                &self.device,
                D3D12_HEAP_TYPE_DEFAULT,
                BYTES,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let others: Vec<ID3D12Resource> = (0..resources)
                .map(|_| {
                    committed(
                        &self.device,
                        D3D12_HEAP_TYPE_DEFAULT,
                        64 << 10,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )
                })
                .collect();
            // SAFETY: every slot is written once, with a view of a live
            // buffer or a null view.
            let heap = unsafe {
                let heap: ID3D12DescriptorHeap = self
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: slots,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .unwrap();
                let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
                let step = self
                    .device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                for i in 0..slots as usize {
                    let (resource, words) = match i {
                        0 => (Some(&target), BYTES / 4),
                        i if i <= resources => (Some(&others[i - 1]), (64 << 10) / 4),
                        _ => (None, 1),
                    };
                    self.device.CreateUnorderedAccessView(
                        resource,
                        None,
                        Some(&raw_view(words)),
                        D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: start + step * i,
                        },
                    );
                }
                heap
            };
            // SAFETY: a live shader-visible heap.
            let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
            let time = |dispatches: u32| {
                let words = (BYTES / 4) as u32 / dispatches;
                let mut times: Vec<Duration> = (0..9)
                    .map(|_| {
                        self.run(|l| {
                            // SAFETY: our pipeline, the heap and the buffers,
                            // which outlive the submission.
                            unsafe {
                                l.SetDescriptorHeaps(&[Some(heap.clone())]);
                                l.SetComputeRootSignature(&self.root_table);
                                l.SetPipelineState(&self.pso_table);
                                l.SetComputeRootDescriptorTable(1, gpu);
                                for i in 0..dispatches {
                                    let constants = [7, i * words, words];
                                    l.SetComputeRoot32BitConstants(
                                        0,
                                        3,
                                        constants.as_ptr().cast(),
                                        0,
                                    );
                                    l.Dispatch(words.div_ceil(64), 1, 1);
                                }
                            }
                        })
                        .execute
                    })
                    .collect();
                times.sort();
                times[times.len() / 2]
            };
            let result = (time(1), time(5000));
            drop(others);
            result
        }

        /// A shader-visible heap of `count` raw views of all of `buffer`.
        fn views(&self, buffer: &ID3D12Resource, bytes: usize, count: u32) -> ID3D12DescriptorHeap {
            // SAFETY: a heap of `count` slots, each written once with a
            // view of a live buffer.
            unsafe {
                let heap: ID3D12DescriptorHeap = self
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .unwrap();
                let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
                let step = self
                    .device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                let view = raw_view(bytes / 4);
                for i in 0..count as usize {
                    self.device.CreateUnorderedAccessView(
                        buffer,
                        None,
                        Some(&view),
                        D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: start + step * i,
                        },
                    );
                }
                heap
            }
        }
    }

    impl Drop for Bench {
        fn drop(&mut self) {
            // SAFETY: our event, closed once.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }

    /// A raw view of `words` words from the start of a buffer.
    fn raw_view(words: usize) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: words as u32,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        }
    }

    fn committed(
        device: &ID3D12Device,
        kind: D3D12_HEAP_TYPE,
        bytes: usize,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: a plain buffer description; the out pointer is ours.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: kind,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: bytes as u64,
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

    /// The fill pipeline, with its UAV in the root or, with `table`, in a
    /// view table of one.
    fn pipeline(device: &ID3D12Device, table: bool) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let uav = if table {
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            }
        } else {
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            }
        };
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 3,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            uav,
        ];
        // SAFETY: the description and the source live through the calls;
        // blobs are read in full before they are released.
        unsafe {
            let (mut blob, mut code): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    ..Default::default()
                },
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let root: ID3D12RootSignature = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    ),
                )
                .unwrap();
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("fill"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let code = code.unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }
}
