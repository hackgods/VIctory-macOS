//! Buffer views a heap sync made, kept for the next ones
//! (`VICTORY_VIEW_TEMPLATES=1`, with `VICTORY_QUICK_SYNC` resolving the
//! views: `quick::resolve`).
//!
//! A world frame makes some 12,000 buffer views in its heap syncs, most of
//! them the same views again: the title copies the same records into its
//! per-draw tables at new slots every frame. Making one costs 0.8-1.1 us
//! under Wine (`CreateShaderResourceView` on an M5 Pro), copying one 20-22 ns
//! (`CopyDescriptors`). A descriptor is what the host made of a resource and
//! a view description, and nothing else: one made for a buffer and a
//! description is the one a later view of that buffer with that description
//! would get. So each (host buffer, description) is made once into a slot of
//! a CPU-only heap of ours and copied from there into every slot that asks.
//!
//! The cache holds each buffer it has views of, so a buffer it names is never
//! released and its address never taken by another while the entry lives:
//! an entry can only ever be copied for the buffer it was made of. A host
//! buffer whose copy goes ([`forget`], from `Shadow`'s and `Widened`'s drop)
//! leaves the cache at the next batch, and with it the cache's hold on it.
//! Past [`SLOTS`] entries, one not used lately goes (a clock: an entry used
//! since the hand last passed it gets one more turn).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Slots of the cache's heap: 2 MB of descriptors.
pub(super) const SLOTS: u32 = 1 << 16;

/// Whether heap syncs copy buffer views from the cache.
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::ON.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_VIEW_TEMPLATES"))
}

/// A view of a buffer: the buffer (raw) and its description's words.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    buffer: usize,
    format: i32,
    mapping: u32,
    first: u64,
    count: u32,
    stride: u32,
    flags: i32,
}

impl Key {
    /// The key of a buffer view of `buffer` as `desc` describes it; `None`
    /// for any other view.
    fn of(buffer: &ID3D12Resource, desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> Option<Key> {
        if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
            return None;
        }
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        Some(Key {
            buffer: buffer.as_raw() as usize,
            format: desc.Format.0,
            mapping: desc.Shader4ComponentMapping,
            first: b.FirstElement,
            count: b.NumElements,
            stride: b.StructureByteStride,
            flags: b.Flags.0,
        })
    }
}

struct Entry {
    key: Key,
    /// Held while the entry lives (see the module's notes).
    _buffer: ID3D12Resource,
    /// The batch that last used it.
    used: u64,
    /// Used since the clock's hand last passed it.
    again: bool,
}

struct Cache {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: usize,
    slots: Vec<Option<Entry>>,
    index: HashMap<Key, u32>,
    by_buffer: HashMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// Where the search for an entry to give up goes on from.
    hand: u32,
    batch: u64,
}

impl Cache {
    fn new(device: &ID3D12Device, slots: u32) -> Option<Cache> {
        // SAFETY: a CPU-only heap, a source for descriptor copies.
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: slots,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
        }
        .ok()?;
        // SAFETY: plain queries.
        let (start, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize,
            )
        };
        Some(Cache {
            _heap: heap,
            start,
            step,
            slots: (0..slots).map(|_| None).collect(),
            index: HashMap::new(),
            by_buffer: HashMap::new(),
            free: (0..slots).rev().collect(),
            hand: 0,
            batch: 0,
        })
    }

    fn cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.start + slot as usize * self.step,
        }
    }

    /// Takes entry `slot` out of the cache.
    fn remove(&mut self, slot: u32) {
        let Some(entry) = self.slots[slot as usize].take() else {
            return;
        };
        self.index.remove(&entry.key);
        if let Some(slots) = self.by_buffer.get_mut(&entry.key.buffer) {
            slots.retain(|&s| s != slot);
            if slots.is_empty() {
                self.by_buffer.remove(&entry.key.buffer);
            }
        }
        self.free.push(slot);
    }

    /// A free slot, giving up an entry not used lately if none is (never one
    /// the batch copies from): the clock's hand passes an entry used since
    /// its last pass once more, and gives up the first one that was not.
    fn slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let n = self.slots.len() as u32;
        for _ in 0..2 * n {
            let slot = self.hand;
            self.hand = (self.hand + 1) % n;
            let Some(entry) = &mut self.slots[slot as usize] else {
                continue;
            };
            if entry.used == self.batch {
                continue;
            }
            if std::mem::take(&mut entry.again) {
                continue;
            }
            self.remove(slot);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TemplatesGivenUp, 1);
            return self.free.pop();
        }
        None
    }

    /// Notes entry `key` made into `slot` from `buffer` by this batch.
    fn insert(&mut self, key: Key, slot: u32, buffer: &ID3D12Resource) {
        self.slots[slot as usize] = Some(Entry {
            key,
            _buffer: buffer.clone(),
            used: self.batch,
            again: false,
        });
        self.index.insert(key, slot);
        self.by_buffer.entry(key.buffer).or_default().push(slot);
    }

    /// The entry of `key`, noted used by this batch.
    fn find(&mut self, key: &Key) -> Option<u32> {
        let slot = *self.index.get(key)?;
        if let Some(entry) = &mut self.slots[slot as usize] {
            entry.used = self.batch;
            entry.again = true;
        }
        Some(slot)
    }
}

/// The cache, made at first use (`None` inside: no device, or its heap
/// could not be made).
static CACHE: Mutex<Option<Option<Cache>>> = Mutex::new(None);

/// Host buffers (raw) whose copies went since the last batch.
static GONE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// The host buffer `buffer` (raw) goes with its copy: the cache gives up its
/// views of it, and its hold on it, at the next batch. Only this list's lock
/// is taken, so a copy dropped during a batch does not wait for it.
pub(crate) fn forget(buffer: usize) {
    GONE.lock().unwrap_or_else(|e| e.into_inner()).push(buffer);
}

/// One heap sync's use of the cache (a page's stale slots): the entries it
/// copies from are not given up until it ends.
pub(super) struct Batch {
    cache: MutexGuard<'static, Option<Option<Cache>>>,
}

/// The cache for one batch of a heap sync; `None` without a device.
pub(super) fn batch(device: &ID3D12Device) -> Option<Batch> {
    batch_of(device, SLOTS)
}

fn batch_of(device: &ID3D12Device, slots: u32) -> Option<Batch> {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let c = cache
        .get_or_insert_with(|| Cache::new(device, slots))
        .as_mut()?;
    c.batch += 1;
    let gone = std::mem::take(&mut *GONE.lock().unwrap_or_else(|e| e.into_inner()));
    for buffer in gone {
        for slot in c.by_buffer.get(&buffer).cloned().unwrap_or_default() {
            c.remove(slot);
        }
    }
    Some(Batch { cache })
}

impl Batch {
    fn cache(&mut self) -> &mut Cache {
        self.cache
            .as_mut()
            .and_then(Option::as_mut)
            .expect("made by `batch`")
    }

    /// The cache's descriptor of a view of `buffer` as `desc` describes it,
    /// made there the first time: a source to copy into a slot asking for
    /// that view. `None` for a view that is not of a buffer, or with no slot
    /// free.
    pub(super) fn view(
        &mut self,
        device: &ID3D12Device,
        buffer: &ID3D12Resource,
        desc: &D3D12_SHADER_RESOURCE_VIEW_DESC,
    ) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let key = Key::of(buffer, desc)?;
        let c = self.cache();
        if let Some(slot) = c.find(&key) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TemplateHits, 1);
            return Some(c.cpu(slot));
        }
        let slot = c.slot()?;
        #[cfg(feature = "diagnostics")]
        let _made =
            crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncCreateUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SyncCreates, 1);
        // SAFETY: a live buffer and a slot of our CPU-only heap.
        unsafe { device.CreateShaderResourceView(buffer, Some(desc), c.cpu(slot)) };
        c.insert(key, slot, buffer);
        Some(c.cpu(slot))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::backend::HostMemory;

    thread_local! {
        /// [`on`] for this thread (none: the variable's).
        pub(in crate::descriptor::host) static ON: std::cell::Cell<Option<bool>> =
            const { std::cell::Cell::new(None) };
    }

    /// The cache's entries made of `buffer` (raw).
    pub(in crate::descriptor::host) fn entries_of(buffer: usize) -> usize {
        let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .as_ref()
            .and_then(Option::as_ref)
            .and_then(|c| c.by_buffer.get(&buffer))
            .map_or(0, Vec::len)
    }

    fn buffer_view(first: u64, count: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first,
                    NumElements: count,
                    ..Default::default()
                },
            },
        }
    }

    /// The bytes of the descriptor at `at`, if the host's descriptors are
    /// plain memory a process can read. Under D3DMetal they are not (a CPU
    /// handle is no address): there the shader reads of
    /// `quick_sync_views_copied_from_the_cache_read_as_made_ones` stand for
    /// the comparison.
    fn bytes(at: D3D12_CPU_DESCRIPTOR_HANDLE, len: usize) -> Option<Vec<u8>> {
        let mut out = vec![0u8; len];
        crate::host::read_into(at.ptr as *const u8, &mut out).then_some(out)
    }

    /// A view copied from the cache is the view made afresh, byte for byte;
    /// a second ask copies from the same entry, and another description or
    /// buffer gets an entry of its own.
    #[test]
    fn a_cached_view_is_the_view_made_afresh() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let (buffer, _) = host.shared_buffer(4096).expect("a host buffer");
        let (second, _) = host.shared_buffer(4096).expect("a host buffer");
        let fresh = Cache::new(device, 4).expect("a CPU-only heap");
        let desc = buffer_view(2, 16);
        // SAFETY: a live buffer and a slot of our heap.
        unsafe { device.CreateShaderResourceView(&buffer, Some(&desc), fresh.cpu(0)) };
        let mut batch = batch_of(device, SLOTS).expect("the cache");
        let a = batch.view(device, &buffer, &desc).expect("made");
        let b = batch.view(device, &buffer, &desc).expect("found");
        assert_eq!(a.ptr, b.ptr, "the second ask copies from the same entry");
        let c = batch
            .view(device, &buffer, &buffer_view(3, 16))
            .expect("made");
        let d = batch.view(device, &second, &desc).expect("made");
        assert!(c.ptr != a.ptr && d.ptr != a.ptr && c.ptr != d.ptr);
        // SAFETY: slots of our heaps.
        unsafe {
            device.CopyDescriptorsSimple(1, fresh.cpu(1), a, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        if let (Some(made), Some(copied), Some(other), Some(next)) = (
            bytes(fresh.cpu(0), fresh.step),
            bytes(fresh.cpu(1), fresh.step),
            bytes(d, fresh.step),
            bytes(c, fresh.step),
        ) {
            assert_eq!(copied, made, "the copy is the view made afresh");
            assert_ne!(other, made, "a view of another buffer is another view");
            assert_ne!(next, made, "and of other elements");
        }
        drop(batch);
        forget(buffer.as_raw() as usize);
        forget(second.as_raw() as usize);
        drop(batch_of(device, SLOTS));
    }

    /// A host buffer whose copy goes leaves the cache at the next batch, and
    /// the cache lets go of it.
    #[test]
    fn a_buffer_gone_leaves_the_cache() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let (buffer, _) = host.shared_buffer(4096).expect("a host buffer");
        let at = buffer.as_raw() as usize;
        let refs = || {
            let raw = buffer.as_raw();
            // SAFETY: a live object of ours, whose table starts with
            // `IUnknown`'s; the count is put back at once.
            unsafe {
                let table = *(raw as *const *const windows::core::IUnknown_Vtbl);
                let n = ((*table).AddRef)(raw);
                ((*table).Release)(raw);
                n - 1
            }
        };
        let before = refs();
        let mut batch = batch_of(device, SLOTS).expect("the cache");
        batch
            .view(device, &buffer, &buffer_view(0, 8))
            .expect("made");
        batch
            .view(device, &buffer, &buffer_view(8, 8))
            .expect("made");
        drop(batch);
        assert_eq!(tests::entries_of(at), 2);
        assert_eq!(refs(), before + 2, "the cache holds the buffer");
        forget(at);
        drop(batch_of(device, SLOTS));
        assert_eq!(tests::entries_of(at), 0, "gone from the cache");
        assert_eq!(refs(), before, "and let go");
    }

    /// A buffer made after one the cache let go, at the same address or not,
    /// gets a view of its own: the entry of the one that went cannot be
    /// copied for it.
    #[test]
    fn a_buffer_made_again_gets_a_view_of_its_own() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let desc = buffer_view(0, 8);
        let (first, _) = host.shared_buffer(1 << 16).expect("a host buffer");
        let at = first.as_raw() as usize;
        let mut batch = batch_of(device, SLOTS).expect("the cache");
        batch.view(device, &first, &desc).expect("made");
        drop(batch);
        forget(at);
        drop(first);
        drop(batch_of(device, SLOTS));
        assert_eq!(tests::entries_of(at), 0);
        // Buffers made until one takes the address of the one that went
        // (or a few, if none does): each gets the view made of it.
        let fresh = Cache::new(device, 1).expect("a CPU-only heap");
        let mut made = Vec::new();
        for _ in 0..8 {
            let (buffer, _) = host.shared_buffer(1 << 16).expect("a host buffer");
            let mut batch = batch_of(device, SLOTS).expect("the cache");
            let cached = batch.view(device, &buffer, &desc).expect("made");
            drop(batch);
            assert_eq!(
                tests::entries_of(buffer.as_raw() as usize),
                1,
                "an entry of its own"
            );
            // SAFETY: a live buffer and a slot of our heap.
            unsafe { device.CreateShaderResourceView(&buffer, Some(&desc), fresh.cpu(0)) };
            if let (Some(a), Some(b)) = (bytes(fresh.cpu(0), fresh.step), bytes(cached, fresh.step))
            {
                assert_eq!(b, a, "the view made of this buffer");
            }
            let same = buffer.as_raw() as usize == at;
            made.push(buffer);
            if same {
                break;
            }
        }
        for buffer in made {
            forget(buffer.as_raw() as usize);
        }
        drop(batch_of(device, SLOTS));
    }

    /// Past its slots, the cache gives up the least recently used entry,
    /// never one the batch copies from.
    #[test]
    fn a_full_cache_gives_up_its_oldest_entry() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (buffer, _) = host.shared_buffer(4096).expect("a host buffer");
        let mut cache = Cache::new(&host.device, 2).expect("a CPU-only heap");
        let view = |cache: &mut Cache, first: u64| {
            cache.batch += 1;
            let key = Key::of(&buffer, &buffer_view(first, 4)).expect("a buffer view");
            if let Some(slot) = cache.find(&key) {
                return slot;
            }
            let slot = cache.slot().expect("a slot");
            cache.insert(key, slot, &buffer);
            slot
        };
        let a = view(&mut cache, 0);
        let b = view(&mut cache, 4);
        assert_eq!(view(&mut cache, 0), a, "found");
        let c = view(&mut cache, 8);
        assert_eq!(c, b, "the oldest entry went");
        assert_eq!(cache.index.len(), 2);
        // In one batch, an entry it copies from stays.
        cache.batch += 1;
        let key = Key::of(&buffer, &buffer_view(8, 4)).expect("a buffer view");
        assert_eq!(cache.find(&key), Some(c));
        let key = Key::of(&buffer, &buffer_view(0, 4)).expect("a buffer view");
        assert_eq!(cache.find(&key), Some(a));
        assert_eq!(cache.slot(), None, "both are this batch's");
    }
}
