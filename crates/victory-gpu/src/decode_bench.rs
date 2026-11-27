//! Bench (ignored): what it costs on the GPU to take a heap descriptor from
//! the title's own record instead of from a host slot made for it.
//!
//! Today a translated shader reads the view-stride table at the title's
//! heap index and then the host heap at that index, which the host keeps in
//! the title's order (`crate::descriptor::host`): every record the title
//! copies as bytes is a host slot made again. The proposed path reads the
//! title's record at the index (tag, view, generation), then a table with
//! one entry per view (generation, host slot, offset, bytes), then the host
//! heap at that slot: one more dependent load, and no host work per copied
//! record. `decode-bench.hlsl` runs both over the same data, at the world's
//! counts (524,288 heap slots, 4 M views), and the sums must agree.
//!
//! The shader is compiled outside the build (D3DMetal's DXC, as the
//! argument clamp's): `VICTORY_DECODE_BENCH_DXIL` names the container.

use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use crate::backend::HostMemory;
use crate::host::host;

const SLOTS: u32 = 1 << 19;
const VIEWS: u32 = 4 << 20;
const BUFFERS: u32 = 512;
const BUFFER_BYTES: u64 = 256 << 10;
const VIEW_BYTES: u32 = 4096;
const TAG: u32 = u32::from_le_bytes(*b"vVIE");
/// Where the buffers lie in title memory: 32 MiB apart from 4 GiB.
const TITLE_BASE: u64 = 4 << 30;
const TITLE_STEP: u64 = 32 << 20;
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The view a slot's record names, and whether the record is current.
fn slot_view(slot: u32) -> (u32, bool) {
    ((slot.wrapping_mul(7919) + 13) % VIEWS, !slot.is_multiple_of(16))
}

/// A view's generation, buffer and byte offset in it.
fn view_entry(view: u32) -> (u32, u32, u32) {
    (1 + (view & 3), view % BUFFERS, ((view / BUFFERS) % 60) * VIEW_BYTES)
}

fn word(buffer: u32, index: u32) -> u32 {
    (buffer.wrapping_mul(0x1_0001).wrapping_add(index)).wrapping_mul(2_654_435_761)
}

/// What thread `thread` sums over `accesses` reads (modes 0 and 1).
fn expected(thread: u32, accesses: u32, spread: u32) -> u32 {
    let draw = thread / spread;
    (0..accesses).fold(0u32, |acc, k| {
        let slot = hash(draw.wrapping_mul(0x9e37_79b9).wrapping_add(k)) % SLOTS;
        let at = (hash(thread.wrapping_add(k.wrapping_mul(977))) & 0x3ff) * 4;
        let (view, live) = slot_view(slot);
        if !live {
            return acc;
        }
        let (_, buffer, offset) = view_entry(view);
        acc.wrapping_add(word(buffer, (offset + at) / 4))
    })
}

fn raw_srv(first: u64, elements: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: first,
                NumElements: elements,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            },
        },
    }
}

#[test]
#[ignore = "bench: needs VICTORY_DECODE_BENCH_DXIL (decode-bench.hlsl compiled as cs_6_6)"]
fn bench_descriptor_from_title_record() {
    let Some(path) = std::env::var_os("VICTORY_DECODE_BENCH_DXIL") else {
        return;
    };
    let Some(h) = host() else {
        return;
    };
    let mut dxil = std::fs::read(path).expect("the compiled bench shader");
    let _ = victory_shader::emit::sign_container(&mut dxil);
    let device = &h.device;
    let shared = |bytes: u64| h.shared_buffer(bytes).expect("a shared buffer");

    // The title's data: BUFFERS buffers of words.
    let buffers: Vec<_> = (0..BUFFERS)
        .map(|j| {
            let (buffer, data) = shared(BUFFER_BYTES);
            for i in 0..(BUFFER_BYTES / 4) as u32 {
                // SAFETY: inside the buffer's mapping.
                unsafe { data.cast::<u32>().add(i as usize).write(word(j, i)) };
            }
            buffer
        })
        .collect();
    // The view-stride table (today), the title's records and the view table
    // (proposed), and the threads' sums.
    let (strides, strides_at) = shared(u64::from(SLOTS) * 4);
    let (records, records_at) = shared(u64::from(SLOTS) * 32);
    let (views, views_at) = shared(u64::from(VIEWS) * 16);
    let (sums, sums_at) = shared(u64::from(WIDTH * HEIGHT) * 4);
    // Views named whole in their records (#534), and the title page table:
    // 4096 level-1 dwords (the block of each 256 MiB), then per block 4096
    // pages of two dwords (heap slot, the memory's title address).
    let (whole_records, whole_records_at) = shared(u64::from(SLOTS) * 32);
    let blocks = (TITLE_STEP * u64::from(BUFFERS)).div_ceil(256 << 20) as u32 + 1;
    let (pages, pages_at) = shared(4096 * 4 + u64::from(blocks) * 4096 * 8);
    let first_block = (TITLE_BASE >> 28) as u32;
    for block in 0..blocks {
        // SAFETY: inside the page table's mapping.
        unsafe { pages_at.cast::<u32>().add((first_block + block) as usize).write(block) };
    }
    for j in 0..BUFFERS {
        let address = TITLE_BASE + u64::from(j) * TITLE_STEP;
        for page in 0..BUFFER_BYTES >> 16 {
            let at = address + (page << 16);
            let block = (at >> 28) as u32 - first_block;
            let entry = (4096 + block * 4096 + ((at >> 16) & 0xfff) as u32) * 2;
            // SAFETY: inside the page table's mapping.
            unsafe {
                pages_at.cast::<u32>().add(entry as usize).write(SLOTS + 1 + j);
                pages_at.cast::<u32>().add(entry as usize + 1).write(address as u32);
            }
        }
    }
    let null_slot = SLOTS;
    let whole = SLOTS + 1;
    // SAFETY: plain D3D12 creation; the descriptors written lie in the heap.
    let heap: ID3D12DescriptorHeap = unsafe {
        device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: SLOTS + 1 + BUFFERS,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .expect("a view heap")
    };
    // SAFETY: a live heap.
    let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
    // SAFETY: a live device.
    let step = unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as usize;
    let cpu = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: start + i as usize * step,
    };
    let t = Instant::now();
    for slot in 0..SLOTS {
        let (view, live) = slot_view(slot);
        let (generation, buffer, offset) = view_entry(view);
        // SAFETY: inside the tables' mappings; live buffers and heap slots.
        unsafe {
            // Today: a host slot per title slot, at the view's bytes.
            device.CreateShaderResourceView(
                &buffers[buffer as usize],
                Some(&raw_srv(u64::from(offset / 4), VIEW_BYTES / 4)),
                cpu(slot),
            );
            strides_at.cast::<u32>().add(slot as usize).write(if live { 4 } else { 0 });
            // Proposed: the title's record, stale in one slot of 16.
            let record = [TAG, view, if live { generation } else { generation + 1 }];
            records_at
                .add(slot as usize * 32)
                .cast::<[u32; 3]>()
                .write_unaligned(record);
            let address = TITLE_BASE + u64::from(buffer) * TITLE_STEP + u64::from(offset);
            let whole = [if live { TAG } else { 0 }, address as u32, 2 << 30 | (address >> 32) as u32];
            whole_records_at
                .add(slot as usize * 32)
                .cast::<[u32; 3]>()
                .write_unaligned(whole);
        }
    }
    let per_slot = t.elapsed();
    let t = Instant::now();
    for view in 0..VIEWS {
        let (generation, buffer, offset) = view_entry(view);
        // SAFETY: inside the view table's mapping.
        unsafe {
            views_at
                .add(view as usize * 16)
                .cast::<[u32; 4]>()
                .write([generation, whole + buffer, offset, VIEW_BYTES]);
        }
    }
    let per_view = t.elapsed();
    for j in 0..BUFFERS {
        // SAFETY: live buffers and heap slots.
        unsafe {
            device.CreateShaderResourceView(
                &buffers[j as usize],
                Some(&raw_srv(0, (BUFFER_BYTES / 4) as u32)),
                cpu(whole + j),
            );
        }
    }
    // SAFETY: a null view of the declared kind.
    unsafe { device.CreateShaderResourceView(None, Some(&raw_srv(0, 1)), cpu(null_slot)) };
    println!(
        "setup: {SLOTS} host slots made (view + stride word + record) in {:.1} ms = {:.0} ns/slot; {VIEWS} view table entries written in {:.1} ms = {:.1} ns/view",
        per_slot.as_secs_f64() * 1e3,
        per_slot.as_secs_f64() * 1e9 / f64::from(SLOTS),
        per_view.as_secs_f64() * 1e3,
        per_view.as_secs_f64() * 1e9 / f64::from(VIEWS),
    );

    // SAFETY: D3D12 validates the container and its root signature; the
    // queue, allocator, list and fence are this test's own.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &dxil).expect("root signature");
        let pso: ID3D12PipelineState = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: dxil.as_ptr().cast(),
                    BytecodeLength: dxil.len(),
                },
                ..Default::default()
            })
            .expect("pipeline");
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();
        list.Close().unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        let mut done = 0u64;
        let mut run = |mode: u32, accesses: u32, spread: u32| -> f64 {
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            let constants = [mode, accesses, SLOTS, WIDTH, spread, null_slot, TAG, 0];
            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, strides.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, records.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(3, views.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(4, whole_records.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(5, pages.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(6, sums.GetGPUVirtualAddress());
            list.Dispatch(WIDTH / 8, HEIGHT / 8, 1);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = list.cast().unwrap();
            let t = Instant::now();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            done += 1;
            queue.Signal(&fence, done).unwrap();
            if fence.GetCompletedValue() < done {
                fence.SetEventOnCompletion(done, event).unwrap();
                WaitForSingleObject(event, 60_000);
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        let sum_at = |thread: u32| sums_at.cast::<u32>().add(thread as usize).read_volatile();
        const ACCESSES: u32 = 16;
        for spread in [4096u32, 64, 1] {
            // Both paths read the same words: checked on four threads.
            let mut agree = true;
            for mode in [0u32, 1, 4] {
                run(mode, ACCESSES, spread);
                for thread in [0, 1, WIDTH * 500 + 77, WIDTH * HEIGHT - 1] {
                    agree &= sum_at(thread) == expected(thread, ACCESSES, spread);
                }
            }
            let mut ms = [const { Vec::new() }; 5];
            for _ in 0..7 {
                for mode in [0u32, 1, 4, 2, 3, 3, 2, 4, 1, 0] {
                    ms[mode as usize].push(run(mode, ACCESSES, spread));
                }
            }
            let mut line = format!(
                "{} threads x {ACCESSES} descriptor reads, {spread} threads per draw (sums agree: {agree}):",
                WIDTH * HEIGHT
            );
            let mut medians = [0f64; 5];
            for (mode, name) in [
                "stride table + slot (today)",
                "title record + view table",
                "slot only",
                "no loads",
                "whole record + page table",
            ]
                .iter()
                .enumerate()
            {
                ms[mode].sort_by(f64::total_cmp);
                medians[mode] = ms[mode][ms[mode].len() / 2];
                line += &format!(" {name} {:.2} ms (min {:.2});", medians[mode], ms[mode][0]);
            }
            let reads = f64::from(WIDTH * HEIGHT * ACCESSES);
            println!(
                "{line} against today the view table path costs {:+.2} ms = {:+.3} ns per descriptor read, the page table path {:+.2} ms = {:+.3} ns",
                medians[1] - medians[0],
                (medians[1] - medians[0]) * 1e6 / reads,
                medians[4] - medians[0],
                (medians[4] - medians[0]) * 1e6 / reads,
            );
        }
        let _ = windows::Win32::Foundation::CloseHandle(event);
        assert!(device.GetDeviceRemovedReason().is_ok());
    }
    drop((buffers, strides, records, views, sums, whole_records, pages));
}
