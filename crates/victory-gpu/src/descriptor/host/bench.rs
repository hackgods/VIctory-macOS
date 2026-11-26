//! Heap-sync benchmarks: what making a heap sync's changed slots costs on
//! the host. Run alone, in release, under Wine:
//! `tools/macos/test-wine.sh -p victory-gpu --release --lib -- --ignored --nocapture --test-threads=1 heap_sync_bench`

use super::*;
use std::time::Instant;

const PAGE_SLOTS: u32 = (crate::watch::PAGE / 32) as u32;

/// Distinct views the ring's records name, as in the world (~1,700): buffer
/// views and constant buffers of title buffers, written into records of
/// their own (not in a shader-visible heap).
fn distinct_records(n: usize) -> (Vec<crate::object::This>, Vec<[u8; 32]>) {
    let null = std::ptr::null();
    let buffers: Vec<_> = (0..34)
        .map(|_| crate::resource::test_buffer(1 << 16))
        .collect();
    let mut records = vec![[0u8; 32]; n];
    for (k, record) in records.iter_mut().enumerate() {
        let buffer = buffers[k % buffers.len()];
        let part = (k / buffers.len()) as u32;
        let at = record.as_mut_ptr() as u64;
        if k % 2 == 0 {
            // R32_UINT buffer view: 64 elements from element 64 × part.
            let desc = [42, 1, 0x1688, 0, 64 * part, 0, 64, 0, 0, 0];
            crate::descriptor::create_srv(null, buffer, &desc, at);
        } else {
            let (data, _) = crate::resource::buffer_bytes(buffer).expect("a buffer");
            let address = data as u64 + 256 * u64::from(part);
            crate::descriptor::create_cbv(
                null,
                &[address as u32, (address >> 32) as u32, 256, 0],
                at,
            );
        }
    }
    (buffers, records)
}

/// A record's tag, view index and generation.
fn record_words(record: &[u8; 32]) -> [u32; 3] {
    let word = |i: usize| {
        u32::from_le_bytes([
            record[4 * i],
            record[4 * i + 1],
            record[4 * i + 2],
            record[4 * i + 3],
        ])
    };
    [word(0), word(1), word(2)]
}

fn per(ns: u128, n: u32) -> f64 {
    ns as f64 / f64::from(n)
}

/// Host calls a heap sync makes, each timed over a page of slots at once.
#[test]
#[ignore = "a benchmark: run alone, in release"]
fn heap_sync_bench_calls() {
    let Some(host) = host() else { return };
    let n = PAGE_SLOTS;
    let heap = HostHeap::new(false, n).expect("a host heap");
    let (_buffers, records) = distinct_records(64);
    let view = |k: usize| {
        let [_, i, generation] = record_words(&records[k]);
        view_at(i, generation).expect("a current view")
    };
    let srv = view(0);
    let cbv = view(1);
    prepare(&srv);
    prepare(&cbv);
    // SAFETY: plain heap creation.
    let cpu_heap: ID3D12DescriptorHeap = unsafe {
        host.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                ..Default::default()
            })
    }
    .expect("a CPU heap");
    // SAFETY: a live heap.
    let cpu_start = unsafe { cpu_heap.GetCPUDescriptorHandleForHeapStart() };
    let cpu_at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: cpu_start.ptr + (i * heap.step) as usize,
    };
    null_buffer_view(cpu_at(0));
    let _ = make(&srv, cpu_at(1));
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    let time = |what: &str, f: &mut dyn FnMut()| {
        // Once to warm, then measured.
        f();
        let start = Instant::now();
        const ROUNDS: u32 = 8;
        for _ in 0..ROUNDS {
            f();
        }
        let ns = start.elapsed().as_nanos() / u128::from(ROUNDS);
        println!(
            "{what:<58} {:>8.0} ns/slot ({:.2} ms a page)",
            per(ns, n),
            ns as f64 / 1e6
        );
    };
    time("null buffer view (CreateShaderResourceView)", &mut || {
        for i in 0..n {
            null_buffer_view(heap.cpu(i));
        }
    });
    time("buffer view (make)", &mut || {
        for i in 0..n {
            let _ = make(&srv, heap.cpu(i));
        }
    });
    let (buffer, _) = make(&srv, cpu_at(1)).flatten().expect("a buffer view");
    let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 64,
                ..Default::default()
            },
        },
    };
    time("  of which CreateShaderResourceView alone", &mut || {
        for i in 0..n {
            // SAFETY: a live buffer and a slot of our heap.
            unsafe {
                host.device
                    .CreateShaderResourceView(&buffer, Some(&raw), heap.cpu(i))
            };
        }
    });
    time("  of which GetDesc of the buffer", &mut || {
        for _ in 0..n {
            // SAFETY: a live buffer.
            std::hint::black_box(unsafe { buffer.GetDesc() });
        }
    });
    time("  of which shadow::for_view", &mut || {
        let (address, len) = buffer_range(&srv).expect("a buffer view");
        for _ in 0..n {
            std::hint::black_box(crate::shadow::for_view(address, len, 4));
        }
    });
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain resource creation.
    unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 256,
                Height: 256,
                DepthOrArraySize: 1,
                MipLevels: 9,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut texture,
        )
    }
    .expect("a texture");
    let texture = texture.expect("a texture");
    let tex_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: u32::MAX,
                ..Default::default()
            },
        },
    };
    time("texture view (CreateShaderResourceView)", &mut || {
        for i in 0..n {
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                host.device
                    .CreateShaderResourceView(&texture, Some(&tex_srv), heap.cpu(i))
            };
        }
    });
    time(
        "texture unordered view (CreateUnorderedAccessView)",
        &mut || {
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            for i in 0..n {
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    host.device
                        .CreateUnorderedAccessView(&texture, None, Some(&desc), heap.cpu(i))
                };
            }
        },
    );
    time(
        "typed buffer unordered view (CreateUnorderedAccessView)",
        &mut || {
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: 64,
                        ..Default::default()
                    },
                },
            };
            for i in 0..n {
                // SAFETY: a live buffer and a slot of our heap.
                unsafe {
                    host.device
                        .CreateUnorderedAccessView(&buffer, None, Some(&desc), heap.cpu(i))
                };
            }
        },
    );
    time("constant buffer as raw view (make)", &mut || {
        for i in 0..n {
            let _ = make(&cbv, heap.cpu(i));
        }
    });
    time("CopyDescriptorsSimple of 1, a call a slot", &mut || {
        for i in 0..n {
            // SAFETY: slots of live heaps.
            unsafe {
                host.device
                    .CopyDescriptorsSimple(1, heap.cpu(i), cpu_at(1), kind)
            };
        }
    });
    let dst: Vec<_> = (0..n).map(|i| heap.cpu(i)).collect();
    let src: Vec<_> = (0..n).map(|i| cpu_at(i % 2)).collect();
    let ones = vec![1u32; n as usize];
    time(
        "CopyDescriptors, one call a page (ranges of 1)",
        &mut || {
            // SAFETY: slots of live heaps, `n` ranges of one each side.
            unsafe {
                host.device.CopyDescriptors(
                    n,
                    dst.as_ptr(),
                    Some(ones.as_ptr()),
                    n,
                    src.as_ptr(),
                    Some(ones.as_ptr()),
                    kind,
                )
            };
        },
    );
    time(
        "make_locked of a buffer view (all that a make does)",
        &mut || {
            let mut made = heap.made.lock().unwrap();
            for i in 0..n {
                heap.make_locked(&mut made, i, u64::from(i) + 1, &srv, false);
            }
        },
    );
    time("unmake_locked (made slot to null)", &mut || {
        let mut made = heap.made.lock().unwrap();
        for i in 0..n {
            made[i as usize] = 1;
            heap.unmake_locked(&mut made, i);
        }
    });
    time("viewed.set of a buffer view, then none", &mut || {
        for i in 0..n {
            heap.viewed.set(i, Some(&srv), true);
        }
        for i in 0..n {
            heap.viewed.set(i, None, false);
        }
    });
    time("keep of a buffer, then none", &mut || {
        let kept = make(&srv, cpu_at(1)).flatten().map(|(r, _)| r);
        for i in 0..n {
            heap.keep(i, kept.clone().map(|r| (r, None)));
        }
        for i in 0..n {
            heap.keep(i, None);
        }
    });
    time("view_at (pin) and drop", &mut || {
        let [_, v, generation] = record_words(&records[0]);
        for _ in 0..n {
            drop(view_at(v, generation));
        }
    });
    time("prepare of a buffer view", &mut || {
        for _ in 0..n {
            prepare(&srv);
        }
    });
    crate::host::release_retired();
}

/// The world's pattern: a ring of tables the title copies into its heap as
/// bytes every frame, at a new place each frame, so that each of its slots
/// holds another record at every sync (most of them null, the rest of a
/// few distinct views), among pages written with what they held.
#[test]
#[ignore = "a benchmark: run alone, in release"]
fn heap_sync_bench_ring() {
    if host().is_none() {
        return;
    }
    const PAGES: u32 = 40;
    const RING_PAGES: u32 = 20;
    let count = PAGES * PAGE_SLOTS;
    let heap = HostHeap::new(false, count).expect("a host heap");
    let (_buffers, records) = distinct_records(1700);
    let ring = (RING_PAGES * PAGE_SLOTS) as usize;
    // The tables' sequence: 55 % null, the rest of the distinct views.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let tables: Vec<Option<usize>> = (0..ring)
        .map(|_| (next() % 100 >= 55).then(|| next() as usize % records.len()))
        .collect();
    let mut memory = vec![0u8; count as usize * 32];
    // The pages past the ring: views written once, rewritten as they were.
    for slot in ring..count as usize {
        let k = next() as usize % records.len();
        memory[slot * 32..slot * 32 + 32].copy_from_slice(&records[k]);
    }
    let start = memory.as_ptr() as u64;
    let frames = 12;
    let mut sync_ns = 0u128;
    let mut changed = 0u64;
    let mut page_ns = Vec::new();
    for frame in 0..frames {
        let shift = frame * 977;
        for (slot, chunk) in memory[..ring * 32]
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .enumerate()
        {
            let new = tables[(slot + shift) % ring].map_or([0u8; 32], |k| records[k]);
            if frame >= 2 && chunk[..12] != new[..12] {
                changed += 1;
            }
            *chunk = new;
        }
        let begun = Instant::now();
        for page in 0..PAGES {
            let first = page * PAGE_SLOTS;
            let at = Instant::now();
            let bytes = &memory[first as usize * 32..(first + PAGE_SLOTS) as usize * 32];
            heap.sync_slots_bytes(
                first,
                first + PAGE_SLOTS,
                start,
                32,
                start + u64::from(first) * 32,
                bytes,
            );
            if frame >= 2 && page < RING_PAGES {
                page_ns.push(at.elapsed().as_nanos());
            }
        }
        if frame >= 2 {
            sync_ns += begun.elapsed().as_nanos();
        }
        crate::host::release_retired();
    }
    let measured = (frames - 2) as u64;
    page_ns.sort_unstable();
    println!(
        "ring: {} slots changed a frame, sync {:.2} ms a frame, {:.0} ns a changed slot; ring page median {:.2} ms",
        changed / measured,
        sync_ns as f64 / 1e6 / measured as f64,
        sync_ns as f64 / changed as f64,
        page_ns[page_ns.len() / 2] as f64 / 1e6,
    );
    let kept = (0..count).filter(|&i| heap.kept(i)).count();
    println!("ring: {kept} of {count} slots hold a host view at the end");
}
