//! Title memory in place through tile mappings (`VICTORY_INPLACE_TILES`,
//! stage 2 of #221's unified memory): a D3D12 reserved buffer over a title
//! address range whose 64 KiB tiles are mapped, with `UpdateTileMappings`,
//! onto the D3DMetal heaps that hold the title's XMem physical pages
//! ([`crate::alias::PhysicalHeap`]). The GPU then translates title
//! addresses at execution, as the console does: when the title maps other
//! pages, the tiles are mapped again on the queue before the next
//! submission, and a list recorded before that reads the new pages.
//!
//! This first part holds the probes of what D3DMetal's tile mappings do.

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;

    use crate::alias::PhysicalHeap;

    const TILE: usize = 64 << 10;

    /// A queue, list and fence of our own: copies run and waited for.
    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        done: u64,
        event: HANDLE,
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            // SAFETY: our event.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }

    fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        }
    }

    impl Rig {
        fn new(device: &ID3D12Device) -> Rig {
            // SAFETY: plain creation of objects of our own.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                list.Close().unwrap();
                Rig {
                    device: device.clone(),
                    queue,
                    allocator,
                    list,
                    fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
                    done: 0,
                    event: CreateEventW(None, false, false, None).unwrap(),
                }
            }
        }

        fn record(&self, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
            // SAFETY: our own list, the last run waited for.
            unsafe {
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
                record(&self.list);
                self.list.Close().unwrap();
            }
        }

        /// Runs the recorded list and waits for it.
        fn run(&mut self) -> windows::core::Result<()> {
            // SAFETY: our closed list, queue, fence and event.
            unsafe {
                self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
                self.done += 1;
                self.queue.Signal(&self.fence, self.done)?;
                self.fence.SetEventOnCompletion(self.done, self.event)?;
                if WaitForSingleObject(self.event, 10_000).0 != 0 {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
                self.device.GetDeviceRemovedReason()
            }
        }

        fn readback(&self, size: usize) -> ID3D12Resource {
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &buffer_desc(size as u64, D3D12_RESOURCE_FLAG_NONE),
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        /// Maps `tiles` of `reserved` from tile `first` on to the tiles
        /// `offsets` of `heap` (one tile each), or to none.
        fn map(&self, reserved: &ID3D12Resource, first: u32, heap: Option<&ID3D12Heap>, offsets: &[u32]) {
            let start = D3D12_TILED_RESOURCE_COORDINATE {
                X: first,
                ..Default::default()
            };
            let size = D3D12_TILE_REGION_SIZE {
                NumTiles: offsets.len() as u32,
                ..Default::default()
            };
            let flags = vec![
                if heap.is_some() { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_NULL };
                offsets.len()
            ];
            let counts = vec![1u32; offsets.len()];
            // SAFETY: each range maps one tile of a live heap, or none.
            unsafe {
                self.queue.UpdateTileMappings(
                    reserved,
                    1,
                    Some(&start),
                    Some(&size),
                    heap,
                    offsets.len() as u32,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }

        fn reserved(&self, bytes: u64) -> windows::core::Result<ID3D12Resource> {
            let mut reserved: Option<ID3D12Resource> = None;
            // SAFETY: a reserved buffer of whole tiles; the output is ours.
            unsafe {
                self.device.CreateReservedResource(
                    &buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut reserved,
                )?;
            }
            reserved.ok_or_else(windows::core::Error::empty)
        }

        /// What a GPU copy reads of tile `tile` of `reserved`, recorded and
        /// run now.
        fn fetch_tile(&mut self, reserved: &ID3D12Resource, tile: u64) -> Vec<u8> {
            let back = self.readback(TILE);
            // SAFETY: two live buffers, the tile inside the reserved one.
            self.record(|l| unsafe { l.CopyBufferRegion(&back, 0, reserved, tile * TILE as u64, TILE as u64) });
            self.run().unwrap();
            read(&back)
        }
    }

    fn read(back: &ID3D12Resource) -> Vec<u8> {
        let mut p = std::ptr::null_mut();
        // SAFETY: a readback buffer of a tile, read after its copy ran.
        unsafe {
            back.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p.cast::<u8>(), TILE).to_vec();
            back.Unmap(0, None);
            bytes
        }
    }

    /// Two heaps of four tiles each, tile `t` of the first holding bytes
    /// `0xa0 + t`, of the second `0xb0 + t`.
    fn two_heaps(device: &ID3D12Device) -> [PhysicalHeap; 2] {
        let heaps = [0xa0u8, 0xb0].map(|_| PhysicalHeap::new(device, 4 * TILE).expect("a tile heap"));
        for (heap, base) in heaps.iter().zip([0xa0u8, 0xb0]) {
            for t in 0..4 {
                // SAFETY: inside the heap's CPU mapping; no GPU use yet.
                unsafe { (heap.cpu_address() as *mut u8).add(t * TILE).write_bytes(base + t as u8, TILE) };
            }
        }
        heaps
    }

    fn on_d3dmetal() -> Option<ID3D12Device> {
        crate::watch::rosetta().then(|| crate::host::host().map(|h| h.device.clone())).flatten()
    }

    /// D3DMetal follows tile mappings onto two heaps at once, in any order,
    /// and applies a mapping made on the queue after a list was recorded
    /// and before it runs: the list reads the new tile.
    #[test]
    fn tile_mappings_onto_heaps_apply_in_queue_order() {
        let Some(device) = on_d3dmetal() else {
            return;
        };
        let mut rig = Rig::new(&device);
        let heaps = two_heaps(&device);
        let reserved = rig.reserved(8 * TILE as u64).expect("a reserved buffer");
        rig.map(&reserved, 0, Some(&heaps[0]._heap), &[2, 0, 3, 1]);
        rig.map(&reserved, 4, Some(&heaps[1]._heap), &[1, 3, 0, 2]);
        let expected = [0xa2u8, 0xa0, 0xa3, 0xa1, 0xb1, 0xb3, 0xb0, 0xb2];
        for (tile, &byte) in expected.iter().enumerate() {
            assert!(rig.fetch_tile(&reserved, tile as u64).iter().all(|&b| b == byte), "tile {tile}");
        }
        // Recorded, then remapped on the queue, then run.
        let back = rig.readback(TILE);
        // SAFETY: two live buffers.
        rig.record(|l| unsafe { l.CopyBufferRegion(&back, 0, &reserved, 0, TILE as u64) });
        rig.map(&reserved, 0, Some(&heaps[1]._heap), &[3]);
        rig.run().unwrap();
        assert!(read(&back).iter().all(|&b| b == 0xb3), "the list reads the tile mapped before it ran");
        // Unmapped: reads zeros.
        rig.map(&reserved, 1, None, &[0]);
        assert!(rig.fetch_tile(&reserved, 1).iter().all(|&b| b == 0), "a null tile");
    }

    /// Whether a shader's writes through one tile are what a shader reads
    /// through another tile mapped onto the same heap tile, in one list with
    /// a UAV barrier between (the title maps physical pages at several
    /// addresses; #221 found two Metal buffers over one page incoherent
    /// within a submission). Reported, not asserted: it decides the design.
    #[test]
    fn aliased_tiles_within_one_list() {
        use crate::compute::{Param, pipeline};
        let Some(device) = on_d3dmetal() else {
            return;
        };
        let mut rig = Rig::new(&device);
        let heaps = two_heaps(&device);
        let reserved = rig.reserved(2 * TILE as u64).expect("a reserved buffer");
        rig.map(&reserved, 0, Some(&heaps[0]._heap), &[0, 0]);
        const COPY: &str = "cbuffer c : register(b0) { uint src; uint dst; uint fill; };\n\
            RWByteAddressBuffer b : register(u0);\n\
            [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
                uint at = id.x * 4;\n\
                uint v = fill;\n\
                if (fill == 0) v = b.Load(src + at) + 1;\n\
                b.Store((fill == 0 ? dst : src) + at, v);\n\
            }\n";
        let (root, pso) = pipeline(
            COPY,
            windows::core::s!("alias tiles"),
            0,
            &[Param::Constants { register: 0, count: 3 }, Param::Uav { register: 0 }],
        )
        .unwrap_or_else(|e| panic!("the probe's pipeline: {e}"));
        let back = rig.readback(TILE);
        let words = (TILE / 4) as u32;
        // Write 7s through tile 0, then (a barrier between) read them back
        // through tile 1 plus one into the second half of tile 0's page...
        // which is tile 1's page too: so write the result to an offset of
        // its own, past the first half.
        let half = (TILE / 2) as u32;
        // SAFETY: our pipeline and buffers live through the run.
        rig.record(|l| unsafe {
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(1, reserved.GetGPUVirtualAddress());
            l.SetComputeRoot32BitConstants(0, 3, [0u32, 0, 7].as_ptr().cast(), 0);
            l.Dispatch(words / 2 / 64, 1, 1);
            l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::transmute_copy(&reserved),
                    }),
                },
            }]);
            // Read through tile 1 (the same page), write to tile 0's second half.
            l.SetComputeRoot32BitConstants(0, 3, [TILE as u32, half, 0].as_ptr().cast(), 0);
            l.Dispatch(words / 2 / 64, 1, 1);
            l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::transmute_copy(&reserved),
                    }),
                },
            }]);
            l.CopyBufferRegion(&back, 0, &reserved, 0, TILE as u64);
        });
        rig.run().unwrap();
        let bytes = read(&back);
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let stale = (half as usize..TILE).step_by(4).filter(|&at| word(at) != 8).count();
        println!(
            "aliased tiles in one list: {stale} of {} words read stale through the second tile",
            TILE / 2 / 4
        );
    }

    /// Reserved buffers as large as the title's biggest XMem section, with a
    /// tile mapped past 4 GiB.
    #[test]
    fn reserved_buffers_reach_past_4_gib() {
        let Some(device) = on_d3dmetal() else {
            return;
        };
        let mut rig = Rig::new(&device);
        let heaps = two_heaps(&device);
        let bytes = 8u64 << 30;
        let reserved = rig.reserved(bytes).expect("an 8 GiB reserved buffer");
        let tile = (6u64 << 30) / TILE as u64;
        rig.map(&reserved, tile as u32, Some(&heaps[0]._heap), &[3]);
        assert!(rig.fetch_tile(&reserved, tile).iter().all(|&b| b == 0xa3));
    }

    /// What `UpdateTileMappings` costs under D3DMetal: on this thread, a call
    /// of one tile and a call of many; and on the queue, a remap before a
    /// copy against the copy alone.
    #[test]
    #[ignore = "a benchmark: run it alone, with --nocapture"]
    fn bench_update_tile_mappings() {
        let Some(device) = on_d3dmetal() else {
            return;
        };
        let mut rig = Rig::new(&device);
        let heaps = two_heaps(&device);
        let reserved = rig.reserved(4096 * TILE as u64).expect("a reserved buffer");
        rig.map(&reserved, 0, Some(&heaps[0]._heap), &[0]);
        rig.fetch_tile(&reserved, 0);
        const CALLS: u32 = 2000;
        let t = Instant::now();
        for i in 0..CALLS {
            rig.map(&reserved, i % 4096, Some(&heaps[(i & 1) as usize]._heap), &[i % 4]);
        }
        let one = t.elapsed() / CALLS;
        rig.fetch_tile(&reserved, 0);
        let offsets: Vec<u32> = (0..4096).map(|i| i % 4).collect();
        let t = Instant::now();
        rig.map(&reserved, 0, Some(&heaps[1]._heap), &offsets);
        let many = t.elapsed();
        rig.fetch_tile(&reserved, 0);
        let back = rig.readback(TILE);
        // SAFETY: two live buffers.
        rig.record(|l| unsafe { l.CopyBufferRegion(&back, 0, &reserved, 0, TILE as u64) });
        const RUNS: u32 = 200;
        for _ in 0..20 {
            rig.run().unwrap();
        }
        // Interleaved: a run alone, then a run after a remap, each timed.
        let (mut copy, mut remapped) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for i in 0..RUNS {
            let t = Instant::now();
            rig.run().unwrap();
            copy += t.elapsed();
            let t = Instant::now();
            rig.map(&reserved, 0, Some(&heaps[(i & 1) as usize]._heap), &[2]);
            rig.run().unwrap();
            remapped += t.elapsed();
        }
        let (copy, remapped) = (copy / RUNS, remapped / RUNS);
        assert!(read(&back).iter().all(|&b| b == 0xb2), "the last remap's tile");
        println!(
            "UpdateTileMappings under D3DMetal: one tile {:.1} us a call, 4096 tiles in one call {:.2} ms; \
             a 64 KiB copy run and waited {:.1} us, after a remap on the queue {:.1} us",
            one.as_secs_f64() * 1e6,
            many.as_secs_f64() * 1e3,
            copy.as_secs_f64() * 1e6,
            remapped.as_secs_f64() * 1e6,
        );
    }
}
