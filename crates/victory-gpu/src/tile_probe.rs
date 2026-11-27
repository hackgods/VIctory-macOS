//! mac: stage 1b of `notes/full-alias-study.md`. Can a reserved D3D12
//! buffer be tiled (`UpdateTileMappings`) in scrambled 64 KiB page order
//! from a heap whose memory the CPU maps (custom, write-back, L0: the form
//! of `crate::alias::PhysicalHeap`, #275, and Belobug cc4d18dd), so that
//! (a) what the CPU writes through the heap's mapping shaders read through
//! the reserved buffer, and (b) what shaders write through the reserved
//! buffer the CPU reads through the heap's mapping, with no copy call in
//! between; and (c) what changing the mappings of N tiles costs. Then the
//! two nearest alternatives: buffers placed over one heap at scattered
//! offsets, and `mach_vm_remap` of scattered 64 KiB title pages onto
//! consecutive pages of one buffer, page by page
//! (`PhysicalHeap::alias_view`), counted and timed for 1,000 and 10,000
//! pages.
//!
//! Shaders read and write through root views (GPU addresses) and through
//! descriptor-table views, as the title's root and heap views do. Ignored
//! by default: each probe prints `TILE` lines and one `TILE VERDICT` line
//! per question, and fails only where the host fails outright.
//!
//! ```sh
//! tools/macos/test-wine.sh -p victory-gpu --lib tile_probe -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

use crate::alias::PhysicalHeap;
use crate::compute::{Param, srv, uav};

/// A buffer's tile, and the title's page (both 64 KiB).
const TILE: usize = 64 << 10;
const WORDS: usize = TILE / 4;
/// The word the CPU and the shaders write at reserved word `r`:
/// `r * GOLDEN ^ seed`, so a wrong word tells whose it was.
const GOLDEN: u32 = 0x9e37_79b1;

const SHADER: &str = r"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Params : register(b0) { uint mode; uint words; uint seed; uint base; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= words) return;
    if (mode == 0) dst.Store(i * 4, src.Load(i * 4));
    else dst.Store(i * 4, ((base + i) * 2654435761u) ^ seed);
}
";

fn word(r: usize, seed: u32) -> u32 {
    (r as u32).wrapping_mul(GOLDEN) ^ seed
}

/// `GOLDEN`'s inverse modulo 2^32.
fn inverse() -> u32 {
    let mut x = GOLDEN;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(GOLDEN.wrapping_mul(x)));
    }
    x
}

/// The heap's untouched word `k` of heap tile `t`.
fn sentinel(t: usize, k: usize) -> u32 {
    0xa500_0000 | ((t as u32 & 0x3ff) << 14) | k as u32
}

/// What a wrong word `got` at reserved word `r` is, given the seeds written.
fn whose(got: u32, r: usize, seeds: &[(u32, &str)], tiles: usize) -> String {
    if got == 0 {
        return "zero (an unmapped tile, or bytes never written)".into();
    }
    if got & 0xff00_0000 == 0xa500_0000 && (got & 0x3fff) as usize == r % WORDS {
        return format!("heap tile {}'s untouched word (a tile mapped to the wrong heap tile)", (got >> 14) & 0x3ff);
    }
    for &(seed, what) in seeds {
        let at = (got ^ seed).wrapping_mul(inverse()) as usize;
        if at < tiles * WORDS && at % WORDS == r % WORDS {
            return format!("the {what} word of reserved tile {}", at / WORDS);
        }
    }
    "none of the probe's words".into()
}

fn code(e: &windows::core::Error) -> String {
    let hr = e.code().0 as u32;
    let name = match hr {
        0x8000_4001 => "E_NOTIMPL ",
        0x8007_0057 => "E_INVALIDARG ",
        0x8007_000e => "E_OUTOFMEMORY ",
        0x8000_4005 => "E_FAIL ",
        _ => "",
    };
    format!("{name}hr={hr:#010x} ({})", e.message().trim())
}

fn host_device() -> Option<ID3D12Device> {
    let device = crate::host::host().map(|h| h.device.clone());
    if device.is_none() {
        eprintln!("TILE skipped: no host GPU here");
    }
    device
}

/// Seeded xorshift: the scrambles are the same every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// `n` distinct values below `of`, in random order.
    fn pick(&mut self, of: usize, n: usize) -> Vec<u32> {
        let mut all: Vec<u32> = (0..of as u32).collect();
        for i in (1..of).rev() {
            all.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
        all.truncate(n);
        all
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn reserved(device: &ID3D12Device, size: usize) -> windows::core::Result<ID3D12Resource> {
    let mut out = None;
    // SAFETY: plain creation with an output slot of ours.
    unsafe {
        device.CreateReservedResource(
            &buffer_desc(size as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut out,
        )?;
    }
    out.ok_or_else(windows::core::Error::empty)
}

fn placed(device: &ID3D12Device, heap: &ID3D12Heap, offset: usize, size: usize) -> windows::core::Result<ID3D12Resource> {
    let mut out = None;
    // SAFETY: a buffer inside a live heap of ours.
    unsafe {
        device.CreatePlacedResource(
            heap,
            offset as u64,
            &buffer_desc(size as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut out,
        )?;
    }
    out.ok_or_else(windows::core::Error::empty)
}

/// Tiles `first..first + heap_tiles.len()` of `resource` onto those heap
/// tiles, in one call of one-tile ranges, or (`each`) one call per tile as
/// Belobug's `map_tiles` does.
fn map(queue: &ID3D12CommandQueue, resource: &ID3D12Resource, heap: &ID3D12Heap, first: u32, heap_tiles: &[u32], each: bool) {
    let n = heap_tiles.len() as u32;
    if each {
        for (i, offset) in heap_tiles.iter().enumerate() {
            let at = D3D12_TILED_RESOURCE_COORDINATE { X: first + i as u32, ..Default::default() };
            let size = D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() };
            // SAFETY: one tile of a live reserved buffer onto one of a live heap.
            unsafe {
                queue.UpdateTileMappings(
                    resource,
                    1,
                    Some(&at),
                    Some(&size),
                    heap,
                    1,
                    Some(&D3D12_TILE_RANGE_FLAG_NONE),
                    Some(offset),
                    Some(&1),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        return;
    }
    let at = D3D12_TILED_RESOURCE_COORDINATE { X: first, ..Default::default() };
    let size = D3D12_TILE_REGION_SIZE { NumTiles: n, ..Default::default() };
    let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; heap_tiles.len()];
    let counts = vec![1u32; heap_tiles.len()];
    // SAFETY: `n` tiles of a live reserved buffer onto `n` of a live heap.
    unsafe {
        queue.UpdateTileMappings(
            resource,
            1,
            Some(&at),
            Some(&size),
            heap,
            n,
            Some(flags.as_ptr()),
            Some(heap_tiles.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
}

/// A queue, list and fence of the probe's own, and its two compute
/// pipelines: views as root descriptors, and as a descriptor table.
struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
    root: (ID3D12RootSignature, ID3D12PipelineState),
    table: (ID3D12RootSignature, ID3D12PipelineState),
    views: ID3D12DescriptorHeap,
    step: u32,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

/// A view of a dispatch: a GPU address (root), or a buffer and its first
/// byte (table).
#[derive(Clone, Copy)]
enum View<'a> {
    Root(u64),
    Table(&'a ID3D12Resource, usize),
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        let root = crate::compute::pipeline(
            SHADER,
            s!("tile probe (root views)"),
            0,
            &[Param::Srv { register: 0, space: 0 }, Param::Uav { register: 0 }, Param::Constants { register: 0, count: 4 }],
        )?;
        let table = crate::compute::pipeline(
            SHADER,
            s!("tile probe (table views)"),
            0,
            &[Param::Table(&[srv(0, 1), uav(0, 1)]), Param::Constants { register: 0, count: 4 }],
        )?;
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let views: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4096,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            Ok(Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                done: 0,
                event: CreateEventW(None, false, false, None)?,
                root,
                table,
                views,
                step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            })
        }
    }

    /// Waits for everything sent to the queue so far (30 s at most); the
    /// milliseconds waited.
    fn wait(&mut self) -> windows::core::Result<f64> {
        let started = Instant::now();
        // SAFETY: our fence, queue and event.
        unsafe {
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            if self.fence.GetCompletedValue() < self.done {
                self.fence.SetEventOnCompletion(self.done, self.event)?;
                if WaitForSingleObject(self.event, 30_000).0 != 0 {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
            }
            self.device.GetDeviceRemovedReason()?;
        }
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    /// Records with `record`, runs it and waits.
    fn run(&mut self, record: impl FnOnce(&mut Rig)) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
        }
        record(self);
        // SAFETY: as above.
        unsafe {
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
        }
        self.wait().map(|_| ())
    }

    /// Records a dispatch: `mode` 0 copies `words` words from `src` to
    /// `dst`, 1 writes `word(base + i, seed)` at `dst`'s word `i`.
    fn dispatch(&mut self, src: View, dst: View, mode: u32, words: usize, seed: u32, base: usize) {
        let constants = [mode, words as u32, seed, base as u32];
        let list = &self.list;
        // SAFETY: live pipelines, views of live buffers, constants on this stack.
        unsafe {
            match (src, dst) {
                (View::Root(src), View::Root(dst)) => {
                    list.SetComputeRootSignature(&self.root.0);
                    list.SetPipelineState(&self.root.1);
                    list.SetComputeRootShaderResourceView(0, src);
                    list.SetComputeRootUnorderedAccessView(1, dst);
                    list.SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
                }
                (View::Table(src, src_at), View::Table(dst, dst_at)) => {
                    // Slots 0 and 1, written as the dispatch is recorded:
                    // one table dispatch a run.
                    let cpu = self.views.GetCPUDescriptorHandleForHeapStart();
                    let gpu = self.views.GetGPUDescriptorHandleForHeapStart();
                    let elements = words as u32;
                    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: (src_at / 4) as u64,
                                NumElements: elements,
                                StructureByteStride: 0,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            },
                        },
                    };
                    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                FirstElement: (dst_at / 4) as u64,
                                NumElements: elements,
                                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                ..Default::default()
                            },
                        },
                    };
                    let slot = |n: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + (n * self.step) as usize };
                    self.device.CreateShaderResourceView(src, Some(&srv_desc), slot(0));
                    self.device.CreateUnorderedAccessView(dst, None, Some(&uav_desc), slot(1));
                    list.SetDescriptorHeaps(&[Some(self.views.clone())]);
                    list.SetComputeRootSignature(&self.table.0);
                    list.SetPipelineState(&self.table.1);
                    list.SetComputeRootDescriptorTable(0, gpu);
                    list.SetComputeRoot32BitConstants(1, 4, constants.as_ptr().cast(), 0);
                }
                _ => unreachable!("both views of a dispatch are of one kind"),
            }
            list.Dispatch(words.div_ceil(64) as u32, 1, 1);
        }
    }

    fn committed(&self, kind: D3D12_HEAP_TYPE, size: usize, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let mut out = None;
        // SAFETY: plain creation.
        unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(size as u64, flags), state, None, &mut out)?;
        }
        out.ok_or_else(windows::core::Error::empty)
    }

    /// The words a GPU copy reads of `size` bytes of `resource` from `offset`.
    fn fetch(&mut self, resource: &ID3D12Resource, offset: usize, size: usize) -> windows::core::Result<Vec<u32>> {
        let back = self.committed(D3D12_HEAP_TYPE_READBACK, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        // SAFETY: two live buffers, the range inside both.
        self.run(|rig| unsafe { rig.list.CopyBufferRegion(&back, 0, resource, offset as u64, size as u64) })?;
        let mut data = std::ptr::null_mut();
        // SAFETY: the mapping holds `size` bytes until unmapped.
        unsafe {
            back.Map(0, None, Some(&mut data))?;
            let words = std::slice::from_raw_parts(data.cast::<u32>(), size / 4).to_vec();
            back.Unmap(0, None);
            Ok(words)
        }
    }

    /// A default-heap buffer shaders write, for what they read.
    fn target(&self, size: usize) -> windows::core::Result<ID3D12Resource> {
        self.committed(D3D12_HEAP_TYPE_DEFAULT, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)
    }
}

/// A check's outcome; the first that is not `Works` is the verdict.
enum Outcome {
    Works,
    Failed(String),
    Stale(String),
}

fn verdict(question: &str, outcomes: &[(String, Outcome)], works: &str) {
    for (what, outcome) in outcomes {
        match outcome {
            Outcome::Works => {}
            Outcome::Failed(why) => return eprintln!("TILE VERDICT {question}: {why} ({what})"),
            Outcome::Stale(why) => return eprintln!("TILE VERDICT {question}: stale bytes, {why} ({what})"),
        }
    }
    eprintln!("TILE VERDICT {question}: works ({works})");
}

/// The CPU's words at `at`.
///
/// # Safety
/// `at` must be readable for `words` words.
unsafe fn cpu_words<'a>(at: usize, words: usize) -> &'a [u32] {
    // SAFETY: the caller's.
    unsafe { std::slice::from_raw_parts(at as *const u32, words) }
}

/// # Safety
/// `at` must be writable for `WORDS` words.
unsafe fn fill(at: usize, f: impl Fn(usize) -> u32) {
    // SAFETY: the caller's.
    let page = unsafe { std::slice::from_raw_parts_mut(at as *mut u32, WORDS) };
    for (k, w) in page.iter_mut().enumerate() {
        *w = f(k);
    }
}

/// (a) and (b) on a reserved buffer of `tiles` tiles scrambled over a heap of
/// `2 * tiles`, two scrambles one after the other, root and table views.
fn reserved_checks(device: &ID3D12Device) {
    const TILES: usize = 64;
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: the right struct and size for the feature.
    let asked = unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32)
    };
    eprintln!(
        "TILE tiled resources tier {} ({}), resource binding tier {}",
        options.TiledResourcesTier.0,
        asked.as_ref().map_or_else(code, |_| "asked".into()),
        options.ResourceBindingTier.0
    );
    let refused = |what: &str, e: &windows::core::Error| {
        let line = format!("{what} {}", code(e));
        eprintln!("TILE {line}");
        for q in ["a", "b"] {
            eprintln!("TILE VERDICT {q}: {line}");
        }
    };
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return refused("a queue and pipelines of the probe's own:", &e),
    };
    let heap = match PhysicalHeap::new(device, 2 * TILES * TILE) {
        Ok(heap) => heap,
        Err(e) => return refused("CreateHeap (custom, write-back, L0, buffers only) + placed buffer + Map:", &e),
    };
    let buffer = match reserved(device, TILES * TILE) {
        Ok(buffer) => buffer,
        Err(e) => return refused("CreateReservedResource (UAV buffer):", &e),
    };
    let (mut count, mut packed, mut shape, mut subs, mut sub) =
        (0u32, D3D12_PACKED_MIP_INFO::default(), D3D12_TILE_SHAPE::default(), 1u32, D3D12_SUBRESOURCE_TILING::default());
    // SAFETY: every output is a slot on this stack.
    unsafe {
        device.GetResourceTiling(&buffer, Some(&mut count), Some(&mut packed), Some(&mut shape), Some(&mut subs), 0, &mut sub);
    }
    eprintln!("TILE reserved buffer of {TILES} tiles: {count} tiles, tile {} bytes wide", shape.WidthInTexels);
    let target = rig.target(TILES * TILE).expect("a default buffer");
    let cpu = heap.cpu_address();
    // SAFETY: a live reserved buffer.
    let (gpu, target_gpu) = unsafe { (buffer.GetGPUVirtualAddress(), target.GetGPUVirtualAddress()) };
    let mut rng = Rng(0x7a11_e5c0_ffee_0001);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut previous: u32 = 0;
    for scramble in 1..=2 {
        let order = rng.pick(2 * TILES, TILES);
        eprintln!("TILE scramble {scramble}: reserved tiles 0..8 on heap tiles {:?}", &order[..8]);
        // The heap: every word its own sentinel; then the tiles mapped.
        for t in 0..2 * TILES {
            // SAFETY: tile `t` of the heap's mapping; the GPU is idle.
            unsafe { fill(cpu + t * TILE, |k| sentinel(t, k)) };
        }
        map(&rig.queue, &buffer, &heap._heap, 0, &order, false);
        for (n, views) in ["root", "table"].into_iter().enumerate() {
            let seed = 0x1000_0000 * (2 * scramble + n as u32);
            // (a) The CPU writes through the heap's mapping, after the
            // mapping; a shader copies the reserved buffer out.
            for (i, &h) in order.iter().enumerate() {
                // SAFETY: heap tile `h` of the mapping; the GPU is idle.
                unsafe { fill(cpu + h as usize * TILE, |k| word(i * WORDS + k, seed)) };
            }
            let what = format!("scramble {scramble}, {views} views");
            let ran = rig.run(|rig| {
                if n == 0 {
                    rig.dispatch(View::Root(gpu), View::Root(target_gpu), 0, TILES * WORDS, 0, 0);
                } else {
                    rig.dispatch(View::Table(&buffer, 0), View::Table(&target, 0), 0, TILES * WORDS, 0, 0);
                }
            });
            let got = ran.and_then(|()| rig.fetch(&target, 0, TILES * TILE));
            a.push((
                what.clone(),
                match got {
                    Err(e) => Outcome::Failed(code(&e)),
                    Ok(got) => match (0..TILES * WORDS).find(|&r| got[r] != word(r, seed)) {
                        None => Outcome::Works,
                        Some(r) => {
                            let wrong = (0..TILES * WORDS).filter(|&r| got[r] != word(r, seed)).count();
                            Outcome::Stale(format!(
                                "first mismatch at reserved offset {:#x} (tile {} on heap tile {}): got {:#010x}, want {:#010x}, \
                                 {}; {wrong} of {} words wrong",
                                r * 4,
                                r / WORDS,
                                order[r / WORDS],
                                got[r],
                                word(r, seed),
                                whose(got[r], r, &[(seed, "CPU's"), (previous, "previous")], TILES),
                                TILES * WORDS
                            ))
                        }
                    },
                },
            ));
            previous = seed;
            // (b) A shader writes through the reserved buffer; the CPU
            // reads the heap's mapping once the GPU is done.
            let seed = seed | 0x0a5a_0000;
            let ran = rig.run(|rig| {
                if n == 0 {
                    rig.dispatch(View::Root(target_gpu), View::Root(gpu), 1, TILES * WORDS, seed, 0);
                } else {
                    rig.dispatch(View::Table(&target, 0), View::Table(&buffer, 0), 1, TILES * WORDS, seed, 0);
                }
            });
            b.push((
                what,
                match ran {
                    Err(e) => Outcome::Failed(code(&e)),
                    Ok(()) => {
                        let mut first = None;
                        let mut wrong = 0;
                        for (i, &h) in order.iter().enumerate() {
                            // SAFETY: heap tile `h` of the mapping; the GPU is done.
                            let seen = unsafe { cpu_words(cpu + h as usize * TILE, WORDS) };
                            for (k, &w) in seen.iter().enumerate() {
                                if w != word(i * WORDS + k, seed) {
                                    wrong += 1;
                                    first.get_or_insert((i, h as usize, k, w));
                                }
                            }
                        }
                        // The heap tiles no reserved tile is on keep their sentinels.
                        let mut strays = 0;
                        for t in (0..2 * TILES).filter(|t| !order.contains(&(*t as u32))) {
                            // SAFETY: heap tile `t` of the mapping.
                            let seen = unsafe { cpu_words(cpu + t * TILE, WORDS) };
                            strays += seen.iter().enumerate().filter(|&(k, &w)| w != sentinel(t, k)).count();
                        }
                        match first {
                            None if strays == 0 => Outcome::Works,
                            None => Outcome::Stale(format!("{strays} words written into heap tiles no reserved tile is on")),
                            Some((i, h, k, w)) => Outcome::Stale(format!(
                                "first mismatch at heap offset {:#x} (heap tile {h}, reserved offset {:#x}, tile {i}): got {w:#010x}, \
                                 want {:#010x}, {}; {wrong} of {} words wrong, {strays} stray",
                                h * TILE + k * 4,
                                i * TILE + k * 4,
                                word(i * WORDS + k, seed),
                                whose(w, i * WORDS + k, &[(seed, "shader's"), (previous, "CPU's")], TILES),
                                TILES * WORDS
                            )),
                        }
                    }
                },
            ));
            previous = seed;
        }
    }
    for (q, list) in [("a", &a), ("b", &b)] {
        for (what, outcome) in list.iter() {
            let line = match outcome {
                Outcome::Works => "same bytes".to_owned(),
                Outcome::Failed(why) | Outcome::Stale(why) => why.clone(),
            };
            eprintln!("TILE {q} {what}: {line}");
        }
    }
    verdict(
        "a (CPU write through the heap's mapping -> shader read through the reserved buffer)",
        &a,
        "2 scrambles of 64 tiles over a 128-tile heap, root and table views, every word",
    );
    verdict(
        "b (shader write through the reserved buffer -> CPU read through the heap's mapping)",
        &b,
        "2 scrambles of 64 tiles over a 128-tile heap, root and table views, every word, no strays",
    );
}

/// Tiles `0..n` of `resource` mapped to nothing (`D3D12_TILE_RANGE_FLAG_NULL`).
fn unmap(queue: &ID3D12CommandQueue, resource: &ID3D12Resource, n: u32) {
    let at = D3D12_TILED_RESOURCE_COORDINATE::default();
    let size = D3D12_TILE_REGION_SIZE { NumTiles: n, ..Default::default() };
    // SAFETY: `n` tiles of a live reserved buffer, one null range.
    unsafe {
        queue.UpdateTileMappings(
            resource,
            1,
            Some(&at),
            Some(&size),
            None::<&ID3D12Heap>,
            1,
            Some(&D3D12_TILE_RANGE_FLAG_NULL),
            None,
            Some(&n),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
}

/// (c): what changing the mappings of N tiles costs, on a 1,024-tile heap,
/// scrambled: mapping fresh tiles of a new reserved buffer, mapping the same
/// tiles again to other heap tiles (in one call, and one call a tile, as
/// Belobug's `map_tiles`), and mapping them to nothing. The time in the
/// call and then to a fence signalled after it (the queue's own work);
/// medians of 3, each on a new buffer.
fn mapping_cost(device: &ID3D12Device) {
    const HEAP_TILES: usize = 1024;
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return eprintln!("TILE VERDICT c: a queue of the probe's own: {}", code(&e)),
    };
    let heap = match PhysicalHeap::new(device, HEAP_TILES * TILE) {
        Ok(heap) => heap,
        Err(e) => return eprintln!("TILE VERDICT c: CreateHeap {}", code(&e)),
    };
    let cpu = heap.cpu_address();
    for t in 0..HEAP_TILES {
        // SAFETY: heap tile `t`; the GPU is idle.
        unsafe { fill(cpu + t * TILE, |k| sentinel(t, k)) };
    }
    let mut rng = Rng(0xc057_0f0e_11ed_0002);
    let mut scramble = |n: usize| {
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            order.extend(rng.pick(HEAP_TILES, HEAP_TILES.min(n - order.len())));
        }
        order
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut parts = Vec::new();
    let mut took = Vec::new();
    for n in [64usize, 1024, 16384] {
        // (call ms, fence ms) by step, three times.
        let steps = ["fresh", "again", "again a tile a call", "to nothing"];
        let mut times: Vec<Vec<(f64, f64)>> = vec![Vec::new(); steps.len()];
        for _ in 0..3 {
            let buffer = match reserved(device, n * TILE) {
                Ok(buffer) => buffer,
                Err(e) => return eprintln!("TILE VERDICT c: CreateReservedResource of {n} tiles {}", code(&e)),
            };
            let (first, second, third) = (scramble(n), scramble(n), scramble(n));
            for (step, times) in times.iter_mut().enumerate() {
                if step == 2 && n > 1024 {
                    continue;
                }
                let started = Instant::now();
                match step {
                    0 => map(&rig.queue, &buffer, &heap._heap, 0, &first, false),
                    1 => map(&rig.queue, &buffer, &heap._heap, 0, &second, false),
                    2 => map(&rig.queue, &buffer, &heap._heap, 0, &third, true),
                    _ => unmap(&rig.queue, &buffer, n as u32),
                }
                let call = started.elapsed().as_secs_f64() * 1e3;
                match rig.wait() {
                    Ok(fence) => times.push((call, fence)),
                    Err(e) => return eprintln!("TILE VERDICT c: the fence after {} of {n} tiles: {}", steps[step], code(&e)),
                }
                // The mapping took: a GPU copy of 8 of its tiles' first words.
                let want = match step {
                    0 => Some(&first),
                    1 => Some(&second),
                    2 => Some(&third),
                    _ => None,
                };
                let mut wrong = None;
                for tile in (0..8).map(|j| j * n / 8) {
                    match rig.fetch(&buffer, tile * TILE, 16) {
                        Ok(got) => {
                            let bad = (0..4).find(|&k| got[k] != want.map_or(0, |o| sentinel(o[tile] as usize, k)));
                            if let Some(k) = bad {
                                wrong.get_or_insert((tile, got[k], whose(got[k], tile * WORDS + k, &[], n)));
                            }
                        }
                        Err(e) => return eprintln!("TILE VERDICT c: reading the mapped tiles back {}", code(&e)),
                    }
                }
                if let Some((tile, got, what)) = wrong {
                    took.push(format!("{} of {n} tiles did not take (tile {tile} read {got:#010x}, {what})", steps[step]));
                }
            }
        }
        let mut line = Vec::new();
        for (step, times) in steps.iter().zip(times) {
            if times.is_empty() {
                continue;
            }
            eprintln!("TILE c {n} tiles, {step}: (call ms, fence ms) {times:.3?}");
            let call = median(times.iter().map(|t| t.0).collect());
            let fence = median(times.iter().map(|t| t.1).collect());
            line.push(format!(
                "{step} {:.2} ms ({call:.2} in the call, {:.1} us a tile)",
                call + fence,
                (call + fence) * 1e3 / n as f64
            ));
        }
        eprintln!("TILE c {n} tiles: {}", line.join(", "));
        parts.push(format!("N={n}: {}", line.join(", ")));
    }
    let took = if took.is_empty() { "every mapping read back as mapped".to_owned() } else { took.join("; ") };
    eprintln!("TILE VERDICT c (UpdateTileMappings of N tiles, scrambled, call + queue to a fence, medians of 3): {}; {took}", parts.join(" | "));
}

/// Alternative 1: one 64 KiB buffer placed at each of `TILES` scattered
/// offsets of one heap; the CPU and shaders through them.
fn placed_checks(device: &ID3D12Device) {
    const TILES: usize = 64;
    const HEAP_TILES: usize = 1024;
    let fail = |why: String| eprintln!("TILE VERDICT alt1 (placed buffers at scattered offsets): {why}");
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return fail(format!("a queue of the probe's own {}", code(&e))),
    };
    let heap = match PhysicalHeap::new(device, HEAP_TILES * TILE) {
        Ok(heap) => heap,
        Err(e) => return fail(format!("CreateHeap {}", code(&e))),
    };
    let whole = match placed(device, &heap._heap, 0, HEAP_TILES * TILE) {
        Ok(whole) => whole,
        Err(e) => return fail(format!("CreatePlacedResource over the whole heap {}", code(&e))),
    };
    // SAFETY: a live buffer.
    let base = unsafe { whole.GetGPUVirtualAddress() };
    let mut rng = Rng(0x91ac_ed00_5ca7_0003);
    // The cost of many: one buffer on every heap tile.
    let started = Instant::now();
    let many: windows::core::Result<Vec<ID3D12Resource>> =
        (0..HEAP_TILES).map(|t| placed(device, &heap._heap, t * TILE, TILE)).collect();
    let made_ms = started.elapsed().as_secs_f64() * 1e3;
    let many = match many {
        Ok(many) => many,
        Err(e) => return fail(format!("CreatePlacedResource of 64 KiB {}", code(&e))),
    };
    drop(many);
    let order = rng.pick(HEAP_TILES, TILES);
    let buffers: Vec<ID3D12Resource> =
        order.iter().map(|&h| placed(device, &heap._heap, h as usize * TILE, TILE).expect("placed again")).collect();
    // SAFETY: live buffers.
    let addresses: Vec<u64> = buffers.iter().map(|b| unsafe { b.GetGPUVirtualAddress() }).collect();
    let at_offset = order.iter().zip(&addresses).filter(|&(&h, &va)| va == base + (h as usize * TILE) as u64).count();
    eprintln!(
        "TILE alt1 {HEAP_TILES} placed 64 KiB buffers made in {made_ms:.2} ms ({:.1} us each); of {TILES} scattered ones, {at_offset} sit at \
         the heap buffer's GPU address plus their offset",
        made_ms * 1e3 / HEAP_TILES as f64
    );
    let target = rig.target(TILES * TILE).expect("a default buffer");
    // SAFETY: a live buffer.
    let target_gpu = unsafe { target.GetGPUVirtualAddress() };
    let cpu = heap.cpu_address();
    let seed = 0x3300_0000;
    for (i, &h) in order.iter().enumerate() {
        // SAFETY: heap tile `h`; the GPU is idle.
        unsafe { fill(cpu + h as usize * TILE, |k| word(i * WORDS + k, seed)) };
    }
    let ran = rig.run(|rig| {
        for (i, &va) in addresses.iter().enumerate() {
            rig.dispatch(View::Root(va), View::Root(target_gpu + (i * TILE) as u64), 0, WORDS, 0, 0);
        }
    });
    let a = match ran.and_then(|()| rig.fetch(&target, 0, TILES * TILE)) {
        Err(e) => Outcome::Failed(code(&e)),
        Ok(got) => match (0..TILES * WORDS).find(|&r| got[r] != word(r, seed)) {
            None => Outcome::Works,
            Some(r) => Outcome::Stale(format!(
                "CPU -> shader: first mismatch at buffer {} offset {:#x}: got {:#010x}, want {:#010x}",
                r / WORDS,
                (r % WORDS) * 4,
                got[r],
                word(r, seed)
            )),
        },
    };
    let seed = 0x4400_0000;
    let ran = rig.run(|rig| {
        for (i, &va) in addresses.iter().enumerate() {
            rig.dispatch(View::Root(target_gpu), View::Root(va), 1, WORDS, seed, i * WORDS);
        }
    });
    let b = match ran {
        Err(e) => Outcome::Failed(code(&e)),
        Ok(()) => {
            let first = order.iter().enumerate().find_map(|(i, &h)| {
                // SAFETY: heap tile `h`; the GPU is done.
                let seen = unsafe { cpu_words(cpu + h as usize * TILE, WORDS) };
                seen.iter().enumerate().find(|&(k, &w)| w != word(i * WORDS + k, seed)).map(|(k, &w)| (i, h, k, w))
            });
            match first {
                None => Outcome::Works,
                Some((i, h, k, w)) => Outcome::Stale(format!(
                    "shader -> CPU: first mismatch at heap offset {:#x} (buffer {i}): got {w:#010x}, want {:#010x}",
                    h as usize * TILE + k * 4,
                    word(i * WORDS + k, seed)
                )),
            }
        }
    };
    verdict(
        "alt1 (placed buffers at scattered offsets)",
        &[("CPU -> shader".into(), a), ("shader -> CPU".into(), b)],
        &format!(
            "{TILES} buffers of 64 KiB at scattered offsets of a {HEAP_TILES}-tile heap, both ways; {at_offset} of {TILES} at heap address + \
             offset, so each is its own GPU range, not one buffer in title order; {HEAP_TILES} made in {made_ms:.1} ms"
        ),
    );
}

/// Alternative 2: `pages` scattered 64 KiB pages of title memory, each
/// remapped (`mach_vm_remap`, `PhysicalHeap::alias_view`) onto the next page
/// of one heap buffer; then the CPU through the title's pages and shaders
/// through the buffer, on 1,000 of the pages at most.
fn remap_checks(device: &ID3D12Device, pages: usize) -> String {
    let span = 2 * pages;
    let mut rig = match Rig::new(device) {
        Ok(rig) => rig,
        Err(e) => return format!("{pages} pages: a queue of the probe's own {}", code(&e)),
    };
    let heap = match PhysicalHeap::new(device, pages * TILE) {
        Ok(heap) => heap,
        Err(e) => return format!("{pages} pages: CreateHeap {}", code(&e)),
    };
    let buffer = match placed(device, &heap._heap, 0, pages * TILE) {
        Ok(buffer) => buffer,
        Err(e) => return format!("{pages} pages: CreatePlacedResource {}", code(&e)),
    };
    // SAFETY: a fresh allocation of ours, released below.
    let title = unsafe { mem::VirtualAlloc(std::ptr::null(), span * TILE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) } as usize;
    if title == 0 {
        return format!("{pages} pages: VirtualAlloc of {} MB failed", span * TILE >> 20);
    }
    let mut rng = Rng(0x5eed_0f0a_11a5_0004 ^ pages as u64);
    let order = rng.pick(span, pages);
    let started = Instant::now();
    let mut failed = Vec::new();
    for (i, &p) in order.iter().enumerate() {
        // SAFETY: page `p` of our allocation onto page `i` of the live heap.
        if !unsafe { heap.alias_view(title + p as usize * TILE, i * TILE, TILE) } {
            failed.push(i);
        }
    }
    let remap_ms = started.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "TILE alt2 {pages} remaps of scattered title pages onto consecutive buffer pages: {remap_ms:.2} ms ({:.1} us each), {} failed{}",
        remap_ms * 1e3 / pages as f64,
        failed.len(),
        failed.first().map_or(String::new(), |i| format!(", the first at remap {i}"))
    );
    // Wine's view of a remapped page.
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let queried = unsafe {
        mem::VirtualQuery((title + order[0] as usize * TILE) as *const _, &mut info, size_of_val(&info))
    };
    eprintln!(
        "TILE alt2 VirtualQuery of a remapped page: {}",
        if queried == 0 {
            "failed".into()
        } else {
            format!("state {:#x}, protect {:#x}, region {} KiB", info.State, info.Protect, info.RegionSize >> 10)
        }
    );
    // Checked: 1,000 of the pages at most, spread over all of them.
    let checked: Vec<usize> = (0..pages.min(1000)).map(|j| j * pages / pages.min(1000)).collect();
    let target = rig.target(checked.len() * TILE).expect("a default buffer");
    // SAFETY: live buffers.
    let (gpu, target_gpu) = unsafe { (buffer.GetGPUVirtualAddress(), target.GetGPUVirtualAddress()) };
    let seed = 0x5500_0000;
    for &i in &checked {
        // SAFETY: a remapped page of our allocation; the GPU is idle.
        unsafe { fill(title + order[i] as usize * TILE, |k| word(i * WORDS + k, seed)) };
    }
    let ran = rig.run(|rig| {
        for (slot, &i) in checked.iter().enumerate() {
            rig.dispatch(View::Root(gpu + (i * TILE) as u64), View::Root(target_gpu + (slot * TILE) as u64), 0, WORDS, 0, 0);
        }
    });
    let a = match ran.and_then(|()| rig.fetch(&target, 0, checked.len() * TILE)) {
        Err(e) => Outcome::Failed(code(&e)),
        Ok(got) => {
            let first = checked.iter().enumerate().find_map(|(slot, &i)| {
                (0..WORDS).find(|&k| got[slot * WORDS + k] != word(i * WORDS + k, seed)).map(|k| (i, k, got[slot * WORDS + k]))
            });
            match first {
                None => Outcome::Works,
                Some((i, k, w)) => Outcome::Stale(format!(
                    "CPU (title page) -> shader (buffer): first mismatch at buffer offset {:#x}: got {w:#010x}, want {:#010x}",
                    i * TILE + k * 4,
                    word(i * WORDS + k, seed)
                )),
            }
        }
    };
    let seed = 0x6600_0000;
    let ran = rig.run(|rig| {
        for &i in &checked {
            rig.dispatch(View::Root(target_gpu), View::Root(gpu + (i * TILE) as u64), 1, WORDS, seed, i * WORDS);
        }
    });
    let b = match ran {
        Err(e) => Outcome::Failed(code(&e)),
        Ok(()) => {
            let first = checked.iter().find_map(|&i| {
                // SAFETY: a remapped page of our allocation; the GPU is done.
                let seen = unsafe { cpu_words(title + order[i] as usize * TILE, WORDS) };
                seen.iter().enumerate().find(|&(k, &w)| w != word(i * WORDS + k, seed)).map(|(k, &w)| (i, k, w))
            });
            match first {
                None => Outcome::Works,
                Some((i, k, w)) => Outcome::Stale(format!(
                    "shader (buffer) -> CPU (title page): first mismatch at title offset {:#x}: got {w:#010x}, want {:#010x}",
                    order[i] as usize * TILE + k * 4,
                    word(i * WORDS + k, seed)
                )),
            }
        }
    };
    // What the fault tracker's guards cost there: each remapped page made
    // read-only and back, against pages of plain committed memory.
    let protect = |at: &mut dyn Iterator<Item = usize>| {
        let (mut calls, mut bad) = (0, 0);
        let started = Instant::now();
        for page in at {
            let mut old = 0;
            // SAFETY: pages of our own allocations.
            unsafe {
                bad += usize::from(mem::VirtualProtect(page as *const _, TILE, mem::PAGE_READONLY, &mut old) == 0);
                bad += usize::from(mem::VirtualProtect(page as *const _, TILE, mem::PAGE_READWRITE, &mut old) == 0);
            }
            calls += 2;
        }
        (started.elapsed().as_secs_f64() * 1e6 / f64::from(calls.max(1)), bad)
    };
    let (remapped_us, remapped_bad) = protect(&mut order.iter().map(|&p| title + p as usize * TILE));
    // SAFETY: a fresh allocation of ours, released below.
    let plain = unsafe { mem::VirtualAlloc(std::ptr::null(), pages * TILE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) } as usize;
    let (plain_us, plain_bad) = if plain == 0 {
        (f64::NAN, 0)
    } else {
        for i in 0..pages {
            // SAFETY: our allocation; one byte a 16 KiB host page, so every page is backed.
            unsafe {
                for b in (0..TILE).step_by(16 << 10) {
                    ((plain + i * TILE + b) as *mut u8).write_volatile(1);
                }
            }
        }
        protect(&mut (0..pages).map(|i| plain + i * TILE))
    };
    eprintln!(
        "TILE alt2 VirtualProtect of one 64 KiB page: {remapped_us:.1} us a call on the {pages} remapped pages ({remapped_bad} refused), \
         {plain_us:.1} us on plain committed pages ({plain_bad} refused)"
    );
    let started = Instant::now();
    // SAFETY: our allocation; the heap outlives the views remapped onto it.
    let freed = unsafe { mem::VirtualFree(title as *mut _, 0, mem::MEM_RELEASE) } != 0;
    let free_ms = started.elapsed().as_secs_f64() * 1e3;
    if plain != 0 {
        // SAFETY: our allocation.
        unsafe { mem::VirtualFree(plain as *mut _, 0, mem::MEM_RELEASE) };
    }
    drop((buffer, heap));
    let what = format!("{pages} pages");
    let outcomes = [
        (
            format!("{what}, remaps"),
            if failed.is_empty() {
                Outcome::Works
            } else {
                Outcome::Failed(format!("{} of {pages} remaps refused, the first at {}", failed.len(), failed[0]))
            },
        ),
        (format!("{what}, CPU -> shader on {}", checked.len()), a),
        (format!("{what}, shader -> CPU on {}", checked.len()), b),
    ];
    let mut line = String::new();
    for (what, outcome) in &outcomes {
        match outcome {
            Outcome::Works => {}
            Outcome::Failed(why) | Outcome::Stale(why) if line.is_empty() => line = format!("{why} ({what})"),
            _ => {}
        }
    }
    let summary = format!(
        "{pages} remaps in {remap_ms:.1} ms ({:.1} us each), {} refused; protect {remapped_us:.1} us a call (plain {plain_us:.1}); \
         release {free_ms:.1} ms{}",
        remap_ms * 1e3 / pages as f64,
        failed.len(),
        if freed { "" } else { " (refused)" }
    );
    eprintln!("TILE alt2 {summary}");
    if line.is_empty() {
        format!("works, {summary}, both ways on {} pages", checked.len())
    } else {
        format!("{line}; {summary}")
    }
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn tile_probe_1_reserved_scrambled() {
    let Some(device) = host_device() else { return };
    reserved_checks(&device);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn tile_probe_2_mapping_cost() {
    let Some(device) = host_device() else { return };
    mapping_cost(&device);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn tile_probe_3_placed_scattered() {
    let Some(device) = host_device() else { return };
    placed_checks(&device);
}

#[test]
#[ignore = "probe: prints what the host does; run with --ignored --nocapture"]
fn tile_probe_4_remap_pages() {
    let Some(device) = host_device() else { return };
    if !crate::watch::rosetta() {
        return eprintln!("TILE VERDICT alt2: not under Rosetta, no mach_vm_remap");
    }
    let lines: Vec<String> = [1000, 10000].map(|pages| remap_checks(&device, pages)).into();
    eprintln!("TILE VERDICT alt2 (mach_vm_remap page by page): {}", lines.join(" | "));
}

#[test]
fn the_probe_tells_whose_word_it_read() {
    assert_eq!(GOLDEN.wrapping_mul(inverse()), 1);
    let r = 5 * WORDS + 17;
    assert!(whose(word(3 * WORDS + 17, 0x1234), r, &[(0x1234, "CPU's")], 64).contains("reserved tile 3"));
    assert!(whose(sentinel(97, 17), r, &[], 64).contains("heap tile 97"));
    assert!(whose(0, r, &[], 64).starts_with("zero"));
    let mut rng = Rng(1);
    let mut picked = rng.pick(128, 64);
    assert_eq!(picked.len(), 64);
    picked.sort_unstable();
    picked.dedup();
    assert_eq!(picked.len(), 64);
}
