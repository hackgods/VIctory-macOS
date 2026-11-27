//! Title memory in place through tile mappings (`VICTORY_INPLACE_TILES`,
//! stage 2 of #221's unified memory): a D3D12 reserved buffer over a title
//! address range whose 64 KiB tiles are mapped, with `UpdateTileMappings`,
//! onto the D3DMetal heaps that hold the title's XMem physical pages
//! ([`crate::alias::PhysicalHeap`]). The GPU then translates title
//! addresses at execution, as the console does: when the title maps other
//! pages, the tiles are mapped again on the queue before the next
//! submission, and a list recorded before that reads the new pages.
//!
//! A title reservation of 64 KiB pages (the placeholders XMem physical
//! pages are mapped into) becomes a region the first time a title buffer in
//! it is looked up: one reserved buffer the length of the reservation, so
//! that every title buffer there shares it (one GPU address for each title
//! address), its tiles mapped to what the title shows at each page. A
//! buffer whose pages are all shown from section heaps is that buffer at
//! its offset ([`tiled`]), whatever sections and in whatever order: the
//! run of one section in order that `crate::alias` needs is not needed.
//! When the title maps or unmaps pages ([`remapped`]), the tiles there are
//! mapped again on the host queue at once: submissions after it read the
//! new pages, lists recorded before it included.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;

use crate::alias::PhysicalHeap;
use crate::host::host;

/// The title's pages and the tiles of reserved buffers: 64 KiB both.
const TILE: u64 = 64 << 10;

/// What the title shows at a page: the heap that holds it, the tile of
/// the heap it is, and whether the title shows that page at another address
/// too ([`TileResolver`]).
pub struct TileAt {
    pub heap: Arc<PhysicalHeap>,
    pub tile: u32,
    pub shared: bool,
}

/// What the title shows at the page at a title address: `None` where no
/// page on a heap is shown there. The runtime's, installed with
/// [`set_tile_resolver`].
pub type TileResolver = fn(u64) -> Option<TileAt>;

static RESOLVER: OnceLock<TileResolver> = OnceLock::new();

/// Installs the runtime's [`TileResolver`] (the first call's).
pub fn set_tile_resolver(resolve: TileResolver) {
    let _ = RESOLVER.set(resolve);
}

#[cfg(test)]
thread_local! {
    /// The resolver this thread's lookups use in place of the runtime's.
    pub(crate) static TEST_RESOLVER: std::cell::Cell<Option<TileResolver>> = const { std::cell::Cell::new(None) };
    /// Overrides `VICTORY_INPLACE_TILES` on this thread.
    pub(crate) static TEST_TILES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn resolver() -> Option<TileResolver> {
    #[cfg(test)]
    if let Some(resolve) = TEST_RESOLVER.with(std::cell::Cell::get) {
        return Some(resolve);
    }
    RESOLVER.get().copied()
}

/// Whether title memory in place goes through tile mappings
/// (`VICTORY_INPLACE_TILES=1`, with `VICTORY_INPLACE_SECTIONS=1`).
pub fn in_place_tiles() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_TILES.with(std::cell::Cell::get) {
        return on && crate::alias::in_place_sections();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_INPLACE_TILES") && crate::alias::in_place_sections();
        if on {
            tracing::info!("title memory in place through tile mappings (VICTORY_INPLACE_TILES)");
        }
        on
    })
}

/// Whether two tiles mapped onto one heap tile read what a shader wrote
/// through the other within one list: they do under D3DMetal on an M5 Pro
/// (`tests::aliased_tiles_within_one_list`: all 8,192 words after a UAV
/// barrier, between two tiles and between a tile and the heap's own
/// buffer, either way), so pages the title shows at more than one address
/// stay in place through tiles. `VICTORY_INPLACE_TILES_SHOWN_ONCE=1` leaves
/// them to `crate::alias`'s one GPU address per page instead.
fn aliased_tiles_coherent() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !crate::switch::on("VICTORY_INPLACE_TILES_SHOWN_ONCE"))
}

/// The title's reservations of 64 KiB pages for physical pages, by start:
/// their end.
static RESERVATIONS: RwLock<BTreeMap<u64, u64>> = RwLock::new(BTreeMap::new());

/// The title reserved `[base, base + len)` for physical pages.
pub fn title_reservation(base: u64, len: u64) {
    if !in_place_tiles() || len == 0 || !base.is_multiple_of(TILE) {
        return;
    }
    RESERVATIONS
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(base, base + len.next_multiple_of(TILE));
    // A region left from a reservation there before is not this one's.
    REGIONS
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&base);
}

/// A reservation made a region: its reserved buffer and what each of its
/// tiles shows.
struct Region {
    base: u64,
    reserved: ID3D12Resource,
    gpu: u64,
    /// Per tile: the heap (its address, as a key) and the tile of it shown
    /// there; `None` for none.
    tiles: Mutex<Vec<Option<(usize, u32)>>>,
    /// The heaps its tiles map onto, kept while it lives.
    heaps: Mutex<Vec<Arc<PhysicalHeap>>>,
}

// SAFETY: D3D12 resources are free-threaded; the rest is behind locks.
unsafe impl Send for Region {}
// SAFETY: as above.
unsafe impl Sync for Region {}

static REGIONS: RwLock<BTreeMap<u64, Arc<Region>>> = RwLock::new(BTreeMap::new());

/// The reservation holding `[start, start + len)`: its start and end.
fn reservation(start: u64, len: u64) -> Option<(u64, u64)> {
    RESERVATIONS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .range(..=start)
        .next_back()
        .filter(|&(_, &end)| start.saturating_add(len) <= end)
        .map(|(&base, &end)| (base, end))
}

/// The region of the reservation `[base, end)`, made the first time.
fn region(base: u64, end: u64) -> Option<Arc<Region>> {
    if let Some(r) = REGIONS.read().unwrap_or_else(|e| e.into_inner()).get(&base) {
        return Some(r.clone());
    }
    let h = host()?;
    let mut reserved: Option<ID3D12Resource> = None;
    // SAFETY: a reserved buffer of whole tiles; the output is ours.
    unsafe {
        h.device
            .CreateReservedResource(
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: end - base,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut reserved,
            )
            .inspect_err(|error| tracing::warn!(len = end - base, %error, "no reserved buffer for title memory"))
            .ok()?;
    }
    let reserved = reserved?;
    crate::host::name(&reserved, || {
        format!("title reservation {base:#x}+{:#x} in place", end - base)
    });
    // SAFETY: a live buffer.
    let gpu = unsafe { reserved.GetGPUVirtualAddress() };
    let tiles = ((end - base) / TILE) as usize;
    let region = Arc::new(Region {
        base,
        reserved,
        gpu,
        tiles: Mutex::new(vec![None; tiles]),
        heaps: Mutex::new(Vec::new()),
    });
    let mut regions = REGIONS.write().unwrap_or_else(|e| e.into_inner());
    // Another thread may have made it meanwhile: that one is the region.
    if let Some(r) = regions.get(&base) {
        return Some(r.clone());
    }
    regions.insert(base, region.clone());
    drop(regions);
    region.map(0, tiles as u32);
    MADE.fetch_add(1, Ordering::Relaxed);
    tracing::info!(
        base = format_args!("{base:#x}"),
        len = end - base,
        "title reservation in place through tile mappings"
    );
    Some(region)
}

static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Tile mappings made again for the title's remaps, and the calls.
static REMAPPED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Region {
    /// Maps `count` tiles from `first` to what the title shows there now,
    /// on the host queue; returns whether any of them is now no heap's.
    fn map(&self, first: u32, count: u32) -> bool {
        let Some(resolve) = resolver() else {
            return true;
        };
        let Some(h) = host() else {
            return true;
        };
        let queue = h.queue_for_tiles();
        let mut tiles = self.tiles.lock().unwrap_or_else(|e| e.into_inner());
        let mut lost = false;
        // Runs of tiles onto one heap, in one call each.
        let mut run: Option<(u32, Arc<PhysicalHeap>, Vec<u32>)> = None;
        let mut nulls: Vec<u32> = Vec::new();
        let flush = |run: &mut Option<(u32, Arc<PhysicalHeap>, Vec<u32>)>| {
            if let Some((start, heap, offsets)) = run.take() {
                update(&queue, &self.reserved, start, Some(&heap._heap), &offsets);
            }
        };
        for t in first..first + count {
            let at = self.base + u64::from(t) * TILE;
            let now = resolve(at);
            let key = now
                .as_ref()
                .map(|a| (Arc::as_ptr(&a.heap) as usize, a.tile));
            // As it is (reserved tiles start with none): nothing to map.
            if tiles[t as usize] == key {
                lost |= key.is_none();
                flush(&mut run);
                continue;
            }
            tiles[t as usize] = key;
            match now {
                Some(TileAt { heap, tile, .. }) => {
                    let next = run
                        .as_ref()
                        .is_some_and(|(s, h, o)| Arc::ptr_eq(h, &heap) && *s + o.len() as u32 == t);
                    if !next {
                        flush(&mut run);
                        let mut heaps = self.heaps.lock().unwrap_or_else(|e| e.into_inner());
                        if !heaps.iter().any(|k| Arc::ptr_eq(k, &heap)) {
                            heaps.push(heap.clone());
                        }
                        run = Some((t, heap, Vec::new()));
                    }
                    if let Some((_, _, offsets)) = &mut run {
                        offsets.push(tile);
                    }
                }
                None => {
                    flush(&mut run);
                    lost = true;
                    nulls.push(t);
                }
            }
        }
        flush(&mut run);
        for t in nulls {
            update(&queue, &self.reserved, t, None, &[0]);
        }
        lost
    }

    fn end(&self) -> u64 {
        self.base + self.tiles.lock().unwrap_or_else(|e| e.into_inner()).len() as u64 * TILE
    }

    /// Whether every page of `[start, start + len)` is shown from a heap
    /// (and, where aliased tiles are not coherent, shown at no other
    /// address now: asked of the runtime, which knows; when that changes
    /// later, `crate::alias::forget_tiled_shown_twice` lets such memory go).
    fn backs(&self, start: u64, len: u64) -> bool {
        let first = ((start - self.base) / TILE) as usize;
        let last = ((start + len.max(1) - 1 - self.base) / TILE) as usize;
        let mapped = {
            let tiles = self.tiles.lock().unwrap_or_else(|e| e.into_inner());
            tiles
                .get(first..=last)
                .is_some_and(|t| t.iter().all(Option::is_some))
        };
        mapped
            && (aliased_tiles_coherent()
                || resolver().is_some_and(|resolve| {
                    (first..=last)
                        .all(|t| resolve(self.base + t as u64 * TILE).is_some_and(|a| !a.shared))
                }))
    }
}

/// Maps tiles from `start` of `reserved`, one each, to tiles `offsets` of
/// `heap`, or to none.
fn update(
    queue: &ID3D12CommandQueue,
    reserved: &ID3D12Resource,
    start: u32,
    heap: Option<&ID3D12Heap>,
    offsets: &[u32],
) {
    let coordinate = D3D12_TILED_RESOURCE_COORDINATE {
        X: start,
        ..Default::default()
    };
    let size = D3D12_TILE_REGION_SIZE {
        NumTiles: offsets.len() as u32,
        ..Default::default()
    };
    let flag = if heap.is_some() {
        D3D12_TILE_RANGE_FLAG_NONE
    } else {
        D3D12_TILE_RANGE_FLAG_NULL
    };
    let flags = vec![flag; offsets.len()];
    let counts = vec![1u32; offsets.len()];
    // SAFETY: each range maps one tile of a live heap, or none, of a live
    // reserved buffer; arrays of `offsets.len()` entries.
    unsafe {
        queue.UpdateTileMappings(
            reserved,
            1,
            Some(&coordinate),
            Some(&size),
            heap,
            offsets.len() as u32,
            Some(flags.as_ptr()),
            Some(offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
    REMAPPED[0].fetch_add(offsets.len() as u64, Ordering::Relaxed);
    REMAPPED[1].fetch_add(1, Ordering::Relaxed);
}

/// The title buffer `[start, start + len)` through its reservation's
/// reserved buffer, if [`in_place_tiles`] and every page of it is shown
/// from a section heap: the buffer, the GPU address of `start` and its
/// offset in the buffer.
pub(crate) fn tiled(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if !in_place_tiles() {
        return None;
    }
    let (base, end) = reservation(start, len)?;
    let region = region(base, end)?;
    region.backs(start, len).then(|| {
        let offset = start - region.base;
        (region.reserved.clone(), region.gpu + offset, offset)
    })
}

/// Whether title memory `[start, start + len)` would be in place through
/// tile mappings now ([`tiled`]).
pub fn title_memory_tiled(start: u64, len: u64) -> bool {
    tiled(start, len).is_some()
}

/// Whether shadows through tiles are let go where the title shows one of
/// their pages at another address too ([`aliased_tiles_coherent`]).
pub(crate) fn keep_pages_shown_once() -> bool {
    in_place_tiles() && !aliased_tiles_coherent()
}

/// Whether `buffer` is a region's reserved buffer (a shadow over it follows
/// the title's remaps by itself).
pub(crate) fn is_region_buffer(buffer: &ID3D12Resource) -> bool {
    use windows::core::Interface;
    MADE.load(Ordering::Relaxed) != 0
        && REGIONS
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .any(|r| r.reserved.as_raw() == buffer.as_raw())
}

/// Tiles mapped again for the title's remaps since the start, and the
/// calls that did it (tests and benches).
#[cfg(test)]
pub(crate) fn remap_counts() -> [u64; 2] {
    REMAPPED.each_ref().map(|c| c.load(Ordering::Relaxed))
}

/// The title mapped or unmapped pages over `[at, at + len)`: the tiles of
/// regions there are mapped again now, on the host queue. Returns whether
/// any tile there is now no heap's (memory in place through it has to be
/// let go).
pub(crate) fn remapped(at: u64, len: u64) -> bool {
    if !in_place_tiles() || len == 0 {
        return false;
    }
    let end = at.saturating_add(len);
    let regions: Vec<Arc<Region>> = REGIONS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .range(..end)
        .filter(|(_, r)| r.end() > at)
        .map(|(_, r)| r.clone())
        .collect();
    let mut lost = false;
    for region in regions {
        let first = at.max(region.base);
        let last = end.min(region.end());
        let (t0, t1) = (
            ((first - region.base) / TILE) as u32,
            ((last - region.base).div_ceil(TILE)) as u32,
        );
        lost |= region.map(t0, t1 - t0);
    }
    lost
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;

    use std::collections::BTreeMap;
    use std::sync::Mutex;

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
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
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
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
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
        fn map(
            &self,
            reserved: &ID3D12Resource,
            first: u32,
            heap: Option<&ID3D12Heap>,
            offsets: &[u32],
        ) {
            let start = D3D12_TILED_RESOURCE_COORDINATE {
                X: first,
                ..Default::default()
            };
            let size = D3D12_TILE_REGION_SIZE {
                NumTiles: offsets.len() as u32,
                ..Default::default()
            };
            let flags = vec![
                if heap.is_some() {
                    D3D12_TILE_RANGE_FLAG_NONE
                } else {
                    D3D12_TILE_RANGE_FLAG_NULL
                };
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
            self.record(|l| unsafe {
                l.CopyBufferRegion(&back, 0, reserved, tile * TILE as u64, TILE as u64)
            });
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
        let heaps =
            [0xa0u8, 0xb0].map(|_| PhysicalHeap::new(device, 4 * TILE).expect("a tile heap"));
        for (heap, base) in heaps.iter().zip([0xa0u8, 0xb0]) {
            for t in 0..4 {
                // SAFETY: inside the heap's CPU mapping; no GPU use yet.
                unsafe {
                    (heap.cpu_address() as *mut u8)
                        .add(t * TILE)
                        .write_bytes(base + t as u8, TILE)
                };
            }
        }
        heaps
    }

    fn on_d3dmetal() -> Option<ID3D12Device> {
        crate::watch::rosetta()
            .then(|| crate::host::host().map(|h| h.device.clone()))
            .flatten()
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
            assert!(
                rig.fetch_tile(&reserved, tile as u64)
                    .iter()
                    .all(|&b| b == byte),
                "tile {tile}"
            );
        }
        // Recorded, then remapped on the queue, then run.
        let back = rig.readback(TILE);
        // SAFETY: two live buffers.
        rig.record(|l| unsafe { l.CopyBufferRegion(&back, 0, &reserved, 0, TILE as u64) });
        rig.map(&reserved, 0, Some(&heaps[1]._heap), &[3]);
        rig.run().unwrap();
        assert!(
            read(&back).iter().all(|&b| b == 0xb3),
            "the list reads the tile mapped before it ran"
        );
        // Unmapped: reads zeros.
        rig.map(&reserved, 1, None, &[0]);
        assert!(
            rig.fetch_tile(&reserved, 1).iter().all(|&b| b == 0),
            "a null tile"
        );
    }

    /// Whether a shader's writes through one tile are what a shader reads
    /// through another tile mapped onto the same heap tile (the title maps
    /// physical pages at several addresses; #221 found two Metal buffers
    /// over one page incoherent within a submission), and whether one list
    /// orders work on a reserved buffer at all. Each case starts from a page
    /// of zeros, writes 7s into its first half, and reads them (plus one)
    /// into its second half: through the tile written (a control), the
    /// other tile, or the heap's own buffer; in one list after a barrier
    /// (all UAVs, as the runtime's, or the one resource), or in the next
    /// submission. The page is copied out in a submission of its own, but
    /// in two cases that ask how a copy in the same list is ordered.
    /// Reported, not asserted: it decides the design
    /// ([`super::aliased_tiles_coherent`]).
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
        let placed = heaps[0].backing().clone();
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        // Two programs as the title's would be: one writes a value, the
        // other reads words and writes them plus one, at offsets given as
        // constants, through one root UAV.
        const WRITE: &str = "cbuffer c : register(b0) { uint dst; uint src; uint value; };\n\
            RWByteAddressBuffer b : register(u0);\n\
            [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
                b.Store(dst + id.x * 4, value);\n\
            }\n";
        // Reads through a view of its own (a root SRV), as the title's
        // programs mostly read.
        const READ: &str = "cbuffer c : register(b0) { uint dst; uint src; uint value; };\n\
            RWByteAddressBuffer b : register(u0);\n\
            ByteAddressBuffer r : register(t0);\n\
            [numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
                b.Store(dst + id.x * 4, r.Load(src + id.x * 4) + 1);\n\
            }\n";
        let params = [
            Param::Constants {
                register: 0,
                count: 3,
            },
            Param::Uav { register: 0 },
            Param::Srv {
                register: 0,
                space: 0,
            },
        ];
        let (root, write) = pipeline(WRITE, windows::core::s!("alias write"), 0, &params)
            .unwrap_or_else(|e| panic!("the probe's write pipeline: {e}"));
        let (_, read_pso) = pipeline(READ, windows::core::s!("alias read"), 0, &params)
            .unwrap_or_else(|e| panic!("the probe's read pipeline: {e}"));
        let back = rig.readback(TILE);
        let words = (TILE / 4) as u32;
        let half = (TILE / 2) as u32;
        let uav = |on: Option<&ID3D12Resource>| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: a copy of the interface pointer (or none)
                    // without a reference of its own, never released; the
                    // buffer outlives the run.
                    pResource: unsafe { std::mem::transmute_copy(&on.cloned()) },
                }),
            },
        };
        let transition = |on: &ID3D12Resource, from, to| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: as in `uav`.
                    pResource: unsafe { std::mem::transmute_copy(on) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: from,
                    StateAfter: to,
                }),
            },
        };
        /// How the page is copied out.
        #[derive(Clone, Copy, PartialEq)]
        enum Copy {
            /// In a submission of its own.
            Apart,
            /// In the same list, after the UAV barrier only.
            AfterUav,
            /// In the same list, after a transition to a copy source.
            AfterTransition,
        }
        /// What a case writes through, reads through and how it is ordered.
        struct Case<'a> {
            name: &'static str,
            written: &'a ID3D12Resource,
            read: &'a ID3D12Resource,
            /// Where the first half is read, in `read`.
            src: u32,
            /// Read in the next submission rather than in the same list.
            split: bool,
            /// Barriers on the written buffer only, rather than all UAVs.
            narrow: bool,
            copy: Copy,
        }
        let case = |name, written, read, src, split, narrow, copy| Case {
            name,
            written,
            read,
            src,
            split,
            narrow,
            copy,
        };
        let other = TILE as u32;
        let cases = [
            case(
                "tile 0, control",
                &reserved,
                &reserved,
                0,
                false,
                false,
                Copy::Apart,
            ),
            case(
                "tile 0, one-resource barrier, control",
                &reserved,
                &reserved,
                0,
                false,
                true,
                Copy::Apart,
            ),
            case(
                "tile 0, then the other tile",
                &reserved,
                &reserved,
                other,
                false,
                false,
                Copy::Apart,
            ),
            case(
                "tile 0, then the other tile, one-resource barrier",
                &reserved,
                &reserved,
                other,
                false,
                true,
                Copy::Apart,
            ),
            case(
                "tile 0, then the other tile, next submission",
                &reserved,
                &reserved,
                other,
                true,
                false,
                Copy::Apart,
            ),
            case(
                "tile 0, then the heap's own buffer",
                &reserved,
                &placed,
                0,
                false,
                false,
                Copy::Apart,
            ),
            case(
                "heap's own buffer, control",
                &placed,
                &placed,
                0,
                false,
                false,
                Copy::Apart,
            ),
            case(
                "heap's own buffer, then tile 1",
                &placed,
                &reserved,
                other,
                false,
                false,
                Copy::Apart,
            ),
            case(
                "tile 0, control, copied after the barrier in the list",
                &reserved,
                &reserved,
                0,
                false,
                false,
                Copy::AfterUav,
            ),
            case(
                "tile 0, control, copied after a transition in the list",
                &reserved,
                &reserved,
                0,
                false,
                false,
                Copy::AfterTransition,
            ),
        ];
        for case in &cases {
            let fence = || uav(case.narrow.then_some(case.written));
            // SAFETY: our pipelines and buffers live through the runs.
            let bind = |l: &ID3D12GraphicsCommandList| unsafe {
                l.SetComputeRootSignature(&root);
                l.SetComputeRootUnorderedAccessView(1, case.written.GetGPUVirtualAddress());
                l.SetComputeRootShaderResourceView(2, case.read.GetGPUVirtualAddress());
            };
            // SAFETY: as `bind`.
            let store = |l: &ID3D12GraphicsCommandList, count: u32, value: u32| unsafe {
                l.SetPipelineState(&write);
                l.SetComputeRoot32BitConstants(0, 3, [0, 0, value].as_ptr().cast(), 0);
                l.Dispatch(count / 64, 1, 1);
                l.ResourceBarrier(&[fence()]);
            };
            // SAFETY: as `bind`.
            let load = |l: &ID3D12GraphicsCommandList| unsafe {
                l.SetPipelineState(&read_pso);
                l.SetComputeRoot32BitConstants(0, 3, [half, case.src, 0].as_ptr().cast(), 0);
                l.Dispatch(words / 2 / 64, 1, 1);
                l.ResourceBarrier(&[fence()]);
            };
            // SAFETY: as `bind`.
            let fetch = |l: &ID3D12GraphicsCommandList| unsafe {
                l.CopyBufferRegion(&back, 0, case.written, 0, TILE as u64);
            };
            let uav_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
            rig.record(|l| {
                bind(l);
                store(l, words, 0);
                store(l, words / 2, 7);
                if !case.split {
                    load(l);
                }
                match case.copy {
                    Copy::Apart => {}
                    Copy::AfterUav => fetch(l),
                    // SAFETY: a buffer of ours, in the state it was put in
                    // by its use as a UAV in this list.
                    Copy::AfterTransition => unsafe {
                        l.ResourceBarrier(&[transition(case.written, uav_state, source)]);
                        fetch(l);
                        l.ResourceBarrier(&[transition(
                            case.written,
                            source,
                            D3D12_RESOURCE_STATE_COMMON,
                        )]);
                    },
                }
            });
            rig.run().unwrap();
            if case.split {
                rig.record(|l| {
                    bind(l);
                    load(l);
                });
                rig.run().unwrap();
            }
            if case.copy == Copy::Apart {
                rig.record(fetch);
                rig.run().unwrap();
            }
            let bytes = read(&back);
            let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let mut seen = BTreeMap::<u32, usize>::new();
            for at in (half as usize..TILE).step_by(4) {
                *seen.entry(word(at)).or_default() += 1;
            }
            let first = (0..half as usize)
                .step_by(4)
                .filter(|&at| word(at) == 7)
                .count();
            let wrong = (TILE / 2 / 4) - seen.get(&8).copied().unwrap_or(0);
            println!(
                "aliased tiles, {}: {wrong} of {} words not 8 (seen {seen:?}; 7s in the first half {first})",
                case.name,
                TILE / 2 / 4
            );
        }
    }

    /// What the test resolver says each title page shows: the heap, its tile
    /// and whether the page is shown elsewhere too.
    type Shown = (std::sync::Arc<PhysicalHeap>, u32, bool);
    static SHOWS: Mutex<BTreeMap<u64, Shown>> = Mutex::new(BTreeMap::new());

    fn test_tile_at(at: u64) -> Option<super::TileAt> {
        let shows = SHOWS.lock().unwrap();
        let (heap, tile, shared) = shows.get(&at)?;
        Some(super::TileAt {
            heap: heap.clone(),
            tile: *tile,
            shared: *shared,
        })
    }

    /// A title reservation of four pages (committed memory of our own
    /// stands for the title's views: the CPU side of a tiled shadow is the
    /// title's address), the first three a title buffer, shown from two
    /// heaps as `shows` says; its in-place shadow.
    struct Title {
        memory: *mut u8,
        resource: usize,
        start: u64,
        heaps: [std::sync::Arc<PhysicalHeap>; 2],
    }

    impl Title {
        fn new(resource: usize, shows: [(usize, u32); 3]) -> Option<Title> {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            let device = on_d3dmetal()?;
            crate::alias::TEST_IN_PLACE.with(|c| c.set(Some(true)));
            super::TEST_TILES.with(|c| c.set(Some(true)));
            super::TEST_RESOLVER.with(|c| c.set(Some(test_tile_at)));
            let [a, b] = two_heaps(&device);
            let heaps = [std::sync::Arc::new(a), std::sync::Arc::new(b)];
            // SAFETY: a fresh reservation, ours.
            let memory = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    4 * TILE,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!memory.is_null());
            let start = memory as u64;
            {
                let mut table = SHOWS.lock().unwrap();
                for (k, (heap, tile)) in shows.into_iter().enumerate() {
                    table.insert(
                        start + (k * TILE) as u64,
                        (heaps[heap].clone(), tile, false),
                    );
                }
            }
            super::title_reservation(start, 4 * TILE as u64);
            crate::shadow::register(resource, start, 3 * TILE as u64);
            Some(Title {
                memory,
                resource,
                start,
                heaps,
            })
        }

        /// The title shows `(heap, tile)` at page `page` from now on.
        fn remap(&self, page: usize, heap: usize, tile: u32) {
            let at = self.start + (page * TILE) as u64;
            SHOWS
                .lock()
                .unwrap()
                .insert(at, (self.heaps[heap].clone(), tile, false));
            crate::watch::title_memory_remapped(at as usize, TILE);
        }

        fn shadow(&self) -> std::sync::Arc<crate::shadow::Shadow> {
            crate::shadow::find(self.start, 3 * TILE as u64).expect("a shadow")
        }

        /// A list copying page `page` of the buffer out, recorded now; and
        /// what it copies to.
        fn record(&self, page: usize) -> (crate::host::HostList, ID3D12Resource, *mut u8) {
            record_copy(&self.shadow(), self.start + (page * TILE) as u64)
        }

        fn run(list: &crate::host::HostList, data: *mut u8) -> Vec<u8> {
            use crate::backend::HostGpu;
            crate::host::host().unwrap().execute(&[list]);
            // SAFETY: the output buffer's CPU mapping, after the wait.
            unsafe { std::slice::from_raw_parts(data, TILE).to_vec() }
        }
    }

    /// A list copying the page at `at` out of `shadow`, recorded now; and
    /// what it copies to.
    fn record_copy(
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        at: u64,
    ) -> (crate::host::HostList, ID3D12Resource, *mut u8) {
        use crate::backend::{HostGpu, HostMemory};
        let host = crate::host::host().unwrap();
        let (out, data) = host.shared_buffer(TILE as u64).expect("an output buffer");
        let list = host.new_list().expect("a list");
        shadow.refresh(&list, at, TILE as u64);
        // SAFETY: both buffers live through the waited submission.
        unsafe {
            list.list
                .CopyBufferRegion(&out, 0, &shadow.buffer, shadow.offset(at), TILE as u64)
        };
        list.close();
        (list, out, data)
    }

    /// What a GPU copy reads of the page at `at` through `shadow` now.
    fn read_page(shadow: &std::sync::Arc<crate::shadow::Shadow>, at: u64) -> Vec<u8> {
        let (list, _out, data) = record_copy(shadow, at);
        Title::run(&list, data)
    }

    impl Drop for Title {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // The title unmaps it all: the tiles go to none, the buffer is
            // let go; then it goes.
            SHOWS
                .lock()
                .unwrap()
                .retain(|&at, _| at < self.start || at >= self.start + 4 * TILE as u64);
            crate::watch::title_memory_remapped(self.memory as usize, 4 * TILE);
            crate::shadow::unregister(self.resource, self.start);
            crate::alias::clear_remapped(self.start, self.start + 4 * TILE as u64);
            crate::alias::TEST_IN_PLACE.with(|c| c.set(None));
            super::TEST_TILES.with(|c| c.set(None));
            super::TEST_RESOLVER.with(|c| c.set(None));
            // SAFETY: our reservation, no longer used.
            unsafe { VirtualFree(self.memory.cast(), 0, MEM_RELEASE) };
        }
    }

    /// A list recorded before the title maps other pages at a buffer in
    /// place through tiles, and submitted after, reads the new pages, as the
    /// console would; the buffer stays in place, and no list is counted as
    /// reading old pages.
    #[test]
    fn a_list_recorded_before_a_remap_reads_the_new_pages() {
        let Some(title) = Title::new(0x7e57_3000, [(0, 0), (0, 1), (0, 2)]) else {
            return;
        };
        let shadow = title.shadow();
        assert!(
            shadow.is_aliased() && super::is_region_buffer(&shadow.buffer),
            "in place through tiles"
        );
        let stale = crate::alias::stale_lists();
        let (list, _out, data) = title.record(0);
        title.remap(0, 1, 3);
        assert_eq!(
            Title::run(&list, data),
            vec![0xb3; TILE],
            "the page mapped there when it ran"
        );
        assert!(
            std::sync::Arc::ptr_eq(&title.shadow(), &shadow),
            "still in place"
        );
        assert_eq!(
            crate::alias::stale_lists(),
            stale,
            "no list reads old pages"
        );
    }

    /// Remapped again and again, a buffer stays in place, and each run reads
    /// what the title shows then.
    #[test]
    fn a_range_remapped_repeatedly_stays_in_place() {
        let Some(title) = Title::new(0x7e57_4000, [(0, 0), (0, 1), (0, 2)]) else {
            return;
        };
        let shadow = title.shadow();
        let before = super::remap_counts();
        for round in 0..6u32 {
            let (heap, tile) = ((round % 2) as usize, round % 4);
            title.remap(1, heap, tile);
            let (list, _out, data) = title.record(1);
            let byte = [0xa0u8, 0xb0][heap] + tile as u8;
            assert_eq!(Title::run(&list, data), vec![byte; TILE], "round {round}");
            assert!(
                std::sync::Arc::ptr_eq(&title.shadow(), &shadow),
                "round {round}: still in place"
            );
        }
        assert!(
            super::remap_counts()[0] >= before[0] + 6,
            "a tile mapped again each time"
        );
    }

    /// A page the title shows at a second address stays in place through
    /// tiles (aliased tiles are coherent within a list under D3DMetal:
    /// `aliased_tiles_within_one_list`): the buffer is kept, and reads it.
    #[test]
    fn pages_shown_twice_stay_in_place_through_tiles() {
        let Some(title) = Title::new(0x7e57_9000, [(0, 0), (0, 1), (0, 2)]) else {
            return;
        };
        let shadow = title.shadow();
        let at = title.start + TILE as u64;
        SHOWS
            .lock()
            .unwrap()
            .insert(at, (title.heaps[0].clone(), 1, true));
        crate::watch::title_memory_aliased(at as usize, &[]);
        assert!(
            std::sync::Arc::ptr_eq(&title.shadow(), &shadow),
            "still in place"
        );
        assert!(super::title_memory_tiled(title.start, 3 * TILE as u64));
        let (list, _out, data) = title.record(1);
        assert_eq!(Title::run(&list, data), vec![0xa1; TILE]);
    }

    /// Pages of two heaps out of order (no run of one section, which
    /// `crate::alias` alone needs) make one buffer in place.
    #[test]
    fn views_out_of_order_are_in_place() {
        let Some(title) = Title::new(0x7e57_5000, [(1, 2), (0, 3), (1, 0)]) else {
            return;
        };
        let shadow = title.shadow();
        assert!(
            super::is_region_buffer(&shadow.buffer),
            "in place through tiles"
        );
        for (page, byte) in [(0, 0xb2u8), (1, 0xa3), (2, 0xb0)] {
            let (list, _out, data) = title.record(page);
            assert_eq!(Title::run(&list, data), vec![byte; TILE], "page {page}");
        }
    }

    /// Title memory outside any buffer (a loose window, `crate::shadow`)
    /// in place through tiles with `VICTORY_INPLACE_LOOSE`: a whole 4 MiB
    /// window over pages of two heaps out of order (on the heaps' own
    /// buffers, 64 KiB pages one by one). It stays in place when the title
    /// maps other heap pages there; when a page of it is no heap's it is
    /// let go, but nothing is refused: once the page is a heap's again, the
    /// whole window is in place again.
    #[test]
    fn loose_windows_are_in_place_through_tiles() {
        use std::sync::Arc;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const WINDOW: usize = 4 << 20;
        const PAGES: usize = WINDOW / TILE;
        let Some(device) = on_d3dmetal() else {
            return;
        };
        // Two heaps of 33 tiles: 32 each for the window, one more each to
        // remap to.
        let heaps = [0u8, 1].map(|h| {
            let heap = PhysicalHeap::new(&device, 33 * TILE).expect("a heap");
            for t in 0..33 {
                // SAFETY: inside the heap's CPU mapping; no GPU use yet.
                unsafe {
                    (heap.cpu_address() as *mut u8)
                        .add(t * TILE)
                        .write_bytes(0x80 + 0x20 * h + t as u8, TILE)
                };
            }
            Arc::new(heap)
        });
        let byte = |heap: usize, tile: u32| 0x80 + 0x20 * heap as u8 + tile as u8;
        // Window page `k`: even pages from heap 0, odd ones from heap 1,
        // each in an order of its own.
        let shows = |k: usize| -> (usize, u32) {
            if k.is_multiple_of(2) {
                (0, (k / 2 * 5 % 32) as u32)
            } else {
                (1, (k / 2 * 3 % 32) as u32)
            }
        };
        // SAFETY: a fresh allocation of ours; a 4 MiB window lies inside.
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * WINDOW,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!memory.is_null());
        let start = (memory as u64).next_multiple_of(WINDOW as u64);
        let page = |k: usize| start + (k * TILE) as u64;
        crate::alias::TEST_IN_PLACE.with(|c| c.set(Some(true)));
        crate::alias::TEST_IN_PLACE_LOOSE.with(|c| c.set(Some(true)));
        super::TEST_TILES.with(|c| c.set(Some(true)));
        super::TEST_RESOLVER.with(|c| c.set(Some(test_tile_at)));
        {
            let mut table = SHOWS.lock().unwrap();
            for k in 0..PAGES {
                let (heap, tile) = shows(k);
                table.insert(page(k), (heaps[heap].clone(), tile, false));
            }
        }
        super::title_reservation(start, WINDOW as u64);

        let at = page(5) + 100;
        let window = crate::shadow::find(at, 64).expect("loose memory in place");
        assert!(
            window.is_aliased() && super::is_region_buffer(&window.buffer),
            "in place through tiles"
        );
        assert_eq!(window.range(), (start, WINDOW as u64), "the whole window");
        for k in [0, 5, 6, PAGES - 1] {
            let (heap, tile) = shows(k);
            assert_eq!(
                read_page(&window, page(k)),
                vec![byte(heap, tile); TILE],
                "page {k}"
            );
        }

        // Other heap pages there: still in place, the new page read.
        let stale = crate::alias::stale_lists();
        SHOWS
            .lock()
            .unwrap()
            .insert(page(5), (heaps[1].clone(), 32, false));
        crate::watch::title_memory_remapped(page(5) as usize, TILE);
        let again = crate::shadow::find(at, 64).expect("loose memory in place");
        assert!(Arc::ptr_eq(&again, &window), "still in place");
        assert_eq!(read_page(&window, page(5)), vec![byte(1, 32); TILE]);
        assert_eq!(
            crate::alias::stale_lists(),
            stale,
            "no list reads old pages"
        );

        // A page no heap's: the window is let go; the pages a range touches
        // elsewhere are in place through tiles.
        SHOWS.lock().unwrap().remove(&page(9));
        crate::watch::title_memory_remapped(page(9) as usize, TILE);
        let touched = crate::shadow::find(at, 64).expect("loose memory in place");
        assert!(!Arc::ptr_eq(&touched, &window), "let go");
        assert!(
            touched.is_aliased() && super::is_region_buffer(&touched.buffer),
            "the page touched, through tiles"
        );
        assert_eq!(touched.range(), (page(5), TILE as u64));

        // A heap's page again, shown at a second address too: the whole
        // window is in place through tiles again (nothing through tiles is
        // refused, and aliased tiles are coherent).
        SHOWS
            .lock()
            .unwrap()
            .insert(page(9), (heaps[0].clone(), 32, true));
        crate::watch::title_memory_remapped(page(9) as usize, TILE);
        let whole = crate::shadow::find(page(9) + 8, 64).expect("loose memory in place");
        assert!(
            whole.is_aliased() && super::is_region_buffer(&whole.buffer),
            "through tiles"
        );
        assert_eq!(
            whole.range(),
            (start, WINDOW as u64),
            "the whole window again"
        );
        assert_eq!(read_page(&whole, page(9)), vec![byte(0, 32); TILE]);

        drop((window, again, touched, whole));
        SHOWS
            .lock()
            .unwrap()
            .retain(|&a, _| a < start || a >= start + WINDOW as u64);
        crate::watch::title_memory_remapped(start as usize, WINDOW);
        crate::shadow::forget_loose_windows(start, start + WINDOW as u64);
        crate::alias::clear_remapped(start, start + WINDOW as u64);
        crate::alias::TEST_IN_PLACE.with(|c| c.set(None));
        crate::alias::TEST_IN_PLACE_LOOSE.with(|c| c.set(None));
        super::TEST_TILES.with(|c| c.set(None));
        super::TEST_RESOLVER.with(|c| c.set(None));
        // SAFETY: our allocation, no longer used.
        unsafe { VirtualFree(memory.cast(), 0, MEM_RELEASE) };
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
            rig.map(
                &reserved,
                i % 4096,
                Some(&heaps[(i & 1) as usize]._heap),
                &[i % 4],
            );
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
        assert!(
            read(&back).iter().all(|&b| b == 0xb2),
            "the last remap's tile"
        );
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
