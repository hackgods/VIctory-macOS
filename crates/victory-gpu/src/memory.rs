//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Each view heap has one page table, which every list using title memory
//! with that heap bound reads, and each host copy has a pair of views in
//! slots of the heap kept for it while it lives ([`HostHeap::pair`], past
//! the per-draw dynamic slots). At each submission with such a list the
//! table is rewritten if copies came or went since, and every copy it names
//! is brought up to date (page by page, as every refresh; once per
//! submission until a step writes title memory). Shaders mark the pages
//! they write in their list's bitmap; after the submission each marked page
//! goes back into title memory ([`crate::shadow::Shadow::copy_back_page`])
//! and the bitmap is cleared.
//!
//! Cost per submission with such a list: one change check of every host
//! copy (the refresh), copies of the pages the title wrote, a compare of
//! each 64 KiB page shaders wrote, and the table's rewrite when copies came
//! or went (two views per new copy); per list a bitmap of 256 KiB.
//!
//! Memory no copy holds reads 0 and takes no writes (null views,
//! bounds-checked); shaders mark such pages (or 256 MiB regions with no
//! block) in the bitmap and [`log_frame`] counts them.
//! shortcut: a page two copies share without either holding all of it is
//! the larger one's.
//! shortcut: a table names copies in at most [`CAPACITY`] 256 MiB regions;
//! past that, the smallest copies read as missing (logged once).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use victory_formats::pagetable::{self, BLOCK, CAPACITY, LEVEL1};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::backend::HostMemory;
use crate::descriptor::host::HostHeap;
use crate::host::HostList;
use crate::shadow::Shadow;

/// Bits in a list's dirty-page bitmap (one per level-2 entry across all blocks).
const BITMAP_BITS: usize = (CAPACITY + 1) * BLOCK;
/// 32-bit words in a dirty-page bitmap.
const BITMAP_WORDS: usize = BITMAP_BITS / 32;
/// Byte size of a dirty-page bitmap buffer (256-byte aligned).
const BITMAP_BYTES: u64 = (BITMAP_BITS / 8).next_multiple_of(256) as u64;
/// Maximum number of dirty-page bitmaps kept in [`BITMAP_POOL`].
const BITMAP_POOL_CAP: usize = 64;
/// The key of our own view heap ([`heap`]) among the title's.
pub(crate) const OURS: usize = usize::MAX;

/// A list's use of title memory with a view heap bound.
pub(crate) struct Map {
    /// The heap's page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The list's page bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
    /// Ranges that go back byte for byte instead (root UAVs, see
    /// [`Map::precise`]).
    precise: Arc<Mutex<Vec<(u64, u64)>>>,
}

impl Map {
    /// Shaders write `[address, address + len)` through a root UAV, which
    /// goes back by its bytes before and after the run
    /// ([`Shadow::gpu_writes`]): the marked pages it touches are left to
    /// that. A whole page from the copy would put back what it held at
    /// the submission over what the title (or DirectStorage, through
    /// physical pages no watch sees) wrote there since.
    /// shortcut: a marked page partly in such a range loses what shaders
    /// wrote in its other part.
    pub(crate) fn precise(&self, address: u64, len: u64) {
        self.precise
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, len));
    }
}

/// The most bytes of a copy one piece of it is named for in a page table,
/// and the most its views show: a raw view holds 2^27 elements (512 MiB).
const PIECE: u64 = 256 << 20;
const PIECE_VIEWED: u64 = 512 << 20;

/// The pieces a page table names the copy of title memory `(start, len)`
/// in: each piece's start, the bytes it is named for, and the bytes its
/// views show from there. A copy a view can hold whole is one piece. A
/// longer one is named 256 MiB at a time, and each piece's views show up to
/// 512 MiB from its start, so an access that begins in a piece's last
/// bytes reads on through them.
fn pieces((start, len): (u64, u64)) -> impl Iterator<Item = (u64, u64, u64)> {
    let whole = len <= PIECE_VIEWED;
    let count = if whole { 1 } else { len.div_ceil(PIECE) };
    (0..count).map(move |n| {
        if whole {
            (start, len, len)
        } else {
            let at = n * PIECE;
            (start + at, PIECE.min(len - at), PIECE_VIEWED.min(len - at))
        }
    })
}

#[test]
fn long_copies_are_named_in_pieces() {
    const MIB: u64 = 1 << 20;
    assert_eq!(pieces((0x1000, 4 * MIB)).collect::<Vec<_>>(), [(0x1000, 4 * MIB, 4 * MIB)]);
    assert_eq!(pieces((0, 512 * MIB)).collect::<Vec<_>>(), [(0, 512 * MIB, 512 * MIB)]);
    assert_eq!(
        pieces((64 * MIB, 700 * MIB)).collect::<Vec<_>>(),
        [
            (64 * MIB, 256 * MIB, 512 * MIB),
            (320 * MIB, 256 * MIB, 444 * MIB),
            (576 * MIB, 188 * MIB, 188 * MIB),
        ]
    );
}

/// What a heap's page table names: the copies, the level-1 index of each
/// block (block 0 is the null one), and the copy each entry names (or none).
#[derive(Default)]
struct Layout {
    copies: Vec<Arc<Shadow>>,
    /// Each copy's bit in the change tracker (`crate::watch::clean_bit`).
    bits: Vec<Option<u32>>,
    blocks: Vec<usize>,
    owner: Vec<Option<usize>>,
}

/// A 256 KiB dirty-page bitmap buffer kept zeroed while in [`BITMAP_POOL`].
struct Bitmap {
    _resource: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: `data` is the persistent mapping of `_resource`, owned exclusively by
// one `PooledBitmap` while bound or by `BITMAP_POOL` while idle; D3D12
// resources are free-threaded.
unsafe impl Send for Bitmap {}
// SAFETY: as above; `defer_missed` and `defer_after` access the mapping
// sequentially on the submitting thread after the submission's fence waits.
unsafe impl Sync for Bitmap {}

/// Reusable zeroed dirty-page bitmap buffers, capped at [`BITMAP_POOL_CAP`].
static BITMAP_POOL: Mutex<Vec<Bitmap>> = Mutex::new(Vec::new());

impl Bitmap {
    /// Pops a zeroed bitmap from [`BITMAP_POOL`] or allocates and zeroes a new one.
    fn acquire() -> Option<Self> {
        if let Some(bitmap) = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Some(bitmap);
        }
        let (resource, data) = crate::host::host()?.shared_buffer(BITMAP_BYTES)?;
        crate::host::name(&resource, || "title memory written pages".into());
        // SAFETY: `data` is a newly mapped shared buffer of at least `BITMAP_WORDS` dwords.
        unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, BITMAP_WORDS) };
        // SAFETY: `resource` is a live D3D12 buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        Some(Self {
            _resource: resource,
            data: data as usize,
            gpu,
        })
    }
}

/// RAII guard keeping a [`Bitmap`] owned by a `HostList` until `HostList::reset`
/// or `Drop`, ensuring it is zeroed before returning to [`BITMAP_POOL`].
struct PooledBitmap {
    bitmap: Option<Bitmap>,
}

impl PooledBitmap {
    fn new(bitmap: Bitmap) -> Self {
        Self {
            bitmap: Some(bitmap),
        }
    }

    fn gpu(&self) -> u64 {
        self.bitmap.as_ref().expect("bitmap present until drop").gpu
    }

    fn data(&self) -> usize {
        self.bitmap
            .as_ref()
            .expect("bitmap present until drop")
            .data
    }
}

impl Drop for PooledBitmap {
    fn drop(&mut self) {
        let Some(bitmap) = self.bitmap.take() else {
            return;
        };
        // SAFETY: `PooledBitmap` drops only when `HostList::reset` or `Drop`
        // releases its deferred closures after any GPU submission has finished;
        // `bitmap.data` is mapped for `BITMAP_WORDS` dwords.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(bitmap.data as *mut u32, BITMAP_WORDS) };
        // `defer_after` clears non-zero words after execution and un-executed
        // lists start clean; verify and clear any stray non-zero words before pooling.
        if let Some(first) = marks.iter().position(|&m| m != 0) {
            for mark in &mut marks[first..] {
                if *mark != 0 {
                    *mark = 0;
                }
            }
        }
        let overflow = {
            let mut pool = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < BITMAP_POOL_CAP {
                pool.push(bitmap);
                None
            } else {
                Some(bitmap)
            }
        };
        drop(overflow);
    }
}

/// A view heap's page table, shared by every list using title memory with
/// the heap bound, and the view pairs of the copies it names.
struct Shared {
    /// The host heap it was made for (a heap made later at a released one's
    /// address is another).
    host: usize,
    table: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// The pair of null views entries without a copy name.
    null: u32,
    /// Each copy's view pair (its first slot), by copy.
    pairs: HashMap<(usize, u32), u32>,
    /// The copies' generation it names ([`crate::shadow::generation`]).
    generation: u64,
    /// The missed regions' count it gives blocks to ([`missed_regions`]).
    regions: usize,
    layout: Arc<Layout>,
    /// The table's working buffers, kept between rewrites (grown as
    /// needed, never shrunk): the copies' order, the level-1 index, each
    /// entry's owner and the dwords ([`pagetable::Scratch`]).
    scratch: pagetable::Scratch,
}

// SAFETY: `data` is the persistent mapping of `table`, written only under
// the lock; D3D12 resources are free-threaded.
unsafe impl Send for Shared {}

/// Page tables by view heap (the title heap's address, or [`OURS`]).
static SHARED: Mutex<Vec<(usize, Shared)>> = Mutex::new(Vec::new());

fn shared() -> std::sync::MutexGuard<'static, Vec<(usize, Shared)>> {
    SHARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// How many page tables there are: each one's layout holds every copy it
/// names ([`crate::shadow::release_idle`] counts on it).
pub(crate) fn tables() -> usize {
    shared().len()
}

/// View heap `key` was released: its page table goes with it (lists that
/// name it hold the table), and a heap made later at its address starts
/// with none of its state.
#[cfg(windows)]
pub(crate) fn release(key: usize) {
    shared().retain(|(k, _)| *k != key);
}

/// Whether view heap `key` has a page table.
#[cfg(test)]
pub(crate) fn has_table(key: usize) -> bool {
    shared().iter().any(|(k, _)| *k == key)
}

/// Runs `f` on the host heap behind view heap `key` while it cannot go
/// away; None if it is gone or has no host heap.
fn with_heap<R>(key: usize, f: impl FnOnce(&HostHeap) -> R) -> Option<R> {
    if key == OURS {
        return heap().map(f);
    }
    crate::descriptor::with_host(key, f)
}

/// Since the last frame line: draws and dispatches with shaders using title
/// memory (by pipeline), page tables written, pages copied back, pages and
/// 256 MiB regions missed (no copy), and the first missed addresses.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    tables: u64,
    written: u64,
    /// Pages shaders wrote that were not copied back: their list read
    /// memory with no copy, so what they computed is not the console's.
    withheld: u64,
    /// The same for ranges written through views (bytes).
    withheld_bytes: u64,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
    /// Missed pages by 256 MiB region (its first address): mapped memory
    /// (a copy can be made), free addresses, and whole-region misses (no
    /// block: the page is not known).
    missed_by_region: std::collections::BTreeMap<u64, [u64; 3]>,
    /// Pages read with no copy that are reserved with nothing mapped: they
    /// read as zeros, as the console's unmapped tiles do, and are no miss.
    unmapped_pages: u64,
    /// Copies made of mapped pages shaders missed, for the lists after.
    copies_made: u64,
    /// Copies made ahead of programs' root constants ([`prefetch`]).
    prefetched: u64,
    /// Addresses root constants and their blocks name where nothing is
    /// committed ([`prefetch`]), and the first: `(named at, address, state)`.
    named_nowhere: u64,
    /// Lists that missed and were looked at for another run, the runs
    /// more they got, and the lists still missing after ([`rerun_tries`]).
    reran_lists: u64,
    reruns: u64,
    rerun_gave_up: u64,
    first_nowhere: Vec<(u64, u64, &'static str)>,
}

static STATS: std::sync::Mutex<Option<Stats>> = std::sync::Mutex::new(None);

fn stats<R: Default>(f: impl FnOnce(&mut Stats) -> R) -> R {
    if !tracing::event_enabled!(tracing::Level::INFO) {
        return R::default();
    }
    f(STATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_default())
}

/// `bytes` of a range shaders wrote through a view did not go back into
/// title memory: their list read memory with no copy.
pub(crate) fn note_withheld(bytes: u64) {
    stats(|s| s.withheld_bytes += bytes);
}

/// A draw or dispatch of `pipeline` uses title memory.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| {
        if let Some(count) = s.uses.get_mut(pipeline) {
            *count += 1;
        } else {
            s.uses.insert(pipeline.to_owned(), 1);
        }
    });
}

/// Frames presented, and the last one in which a list using title memory
/// read or wrote memory with no copy ([`MISS_NEVER`] if none has).
static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSED_IN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(MISS_NEVER);
const MISS_NEVER: u64 = u64::MAX;

/// Whether a list using title memory in the submission being completed
/// read or wrote memory with no copy (completions run one at a time).
static SUBMISSION_MISSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A submission's lists ran: whether one using title memory read or wrote
/// memory with no copy.
pub(crate) fn note_submission(missed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    SUBMISSION_MISSED.store(missed, Relaxed);
    if missed {
        MISSED_IN.store(FRAMES.load(Relaxed), Relaxed);
    }
}

/// Whether what lists using title memory wrote stays out of title memory
/// (in the host copies, for the GPU): some such list missed memory this
/// frame or the one before. A list that missed computed from zeros, and
/// the lists after it compute from its results in the host copies, the
/// BVH builder's write addresses among them, so withholding only the lists
/// that missed broke the title's own data (run drops1: an AABB's +inf/-inf
/// in a pointer the title followed). The frame before counts too: its
/// lists' results are read in this one, before this frame's first miss.
pub(crate) fn withholding() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    withheld_in(FRAMES.load(Relaxed), MISSED_IN.load(Relaxed))
}

fn withheld_in(frame: u64, missed: u64) -> bool {
    missed != MISS_NEVER && frame <= missed.saturating_add(1)
}

/// Title memory whose host copies hold what shaders wrote this frame and
/// the frame withheld ([`withholding`]; ranges).
static DROPPED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Shaders wrote `[address, address + len)` of title memory in host copies
/// and the submission withheld it: the copies take title memory's bytes
/// again at their next refresh, from right after the submission if a list
/// of it missed (its results would feed the frame's later submissions:
/// garbage BVH nodes over meshlet buffers, run shards20), else from the
/// frame's end ([`retake_dropped`]).
pub(crate) fn drop_withheld(address: usize, len: usize) {
    if SUBMISSION_MISSED.load(std::sync::atomic::Ordering::Relaxed) {
        crate::watch::retake(address, len);
    } else {
        DROPPED.lock().unwrap_or_else(|e| e.into_inner()).push((address, len));
    }
}

/// At a frame's end: what the frame withheld leaves the host copies, so
/// that results computed from memory with no copy (garbage record arrays
/// and node pointers of the BVH builder) do not feed later frames whose
/// writes go back (a box16 node's f16 bounds over a title object, run
/// textures9). Each copy holding such memory takes it from title memory at
/// its next refresh, through its tracker.
fn retake_dropped() {
    let dropped = std::mem::take(&mut *DROPPED.lock().unwrap_or_else(|e| e.into_inner()));
    for (address, len) in dropped {
        crate::watch::retake(address, len);
    }
}

/// Logs what title memory did since the last call, if anything: a line per
/// presented frame while shaders use it.
pub(crate) fn log_frame() {
    FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MISSED_COPIES.store(0, std::sync::atomic::Ordering::Relaxed);
    *PREFETCHED.lock().unwrap_or_else(|e| e.into_inner()) = None;
    retake_dropped();
    let s = stats(std::mem::take);
    if s.uses.is_empty()
        && s.written
            + s.withheld
            + s.withheld_bytes
            + s.missed_pages
            + s.missed_regions
            + s.unmapped_pages
            == 0
    {
        return;
    }
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    let missed: Vec<String> = s
        .first_missed
        .iter()
        .map(|&a| {
            #[cfg(feature = "diagnostics")]
            return format!("{a:#x} {}", memory_kind(a));
            #[cfg(not(feature = "diagnostics"))]
            format!("{a:#x}")
        })
        .collect();
    if s.reran_lists > 0 {
        tracing::info!(
            "title memory lists run again this frame: {} lists missed, run {} times more in all, {} still missing after",
            s.reran_lists,
            s.reruns,
            s.rerun_gave_up,
        );
    }
    if s.named_nowhere > 0 {
        let first: Vec<String> = s
            .first_nowhere
            .iter()
            .map(|(from, at, state)| format!("{at:#x} {state} (named at {from:#x})"))
            .collect();
        tracing::info!(
            "title memory programs' constants this frame name {} addresses of 4 GiB and up where nothing is committed (named at 0: in the root constants): {}",
            s.named_nowhere,
            first.join(", "),
        );
    }
    if !s.missed_by_region.is_empty() {
        // The regions shaders missed most in, with what the memory there is.
        let mut regions: Vec<(&u64, &[u64; 3])> = s.missed_by_region.iter().collect();
        regions.sort_by_key(|(_, n)| std::cmp::Reverse(n[0] + n[1] + n[2]));
        let (mapped, free, whole) = regions
            .iter()
            .fold((0, 0, 0), |(m, f, w), (_, n)| (m + n[0], f + n[1], w + n[2]));
        let top: Vec<String> = regions
            .iter()
            .take(24)
            .map(|(at, [m, f, w])| format!("{at:#x}: {m} mapped {f} free {w} no block"))
            .collect();
        tracing::info!(
            "title memory misses this frame by 256 MiB region: {} regions; {mapped} pages of mapped memory, {free} at free addresses, {whole} misses in regions with no block; most: {}",
            regions.len(),
            top.join(", "),
        );
    }
    tracing::info!(
        "title memory this frame: {} draws/dispatches ({}), {} page tables, {} pages \
         copied back, {} withheld and {:.1} MB of views kept (their list read memory with no copy), missed {} pages \
         and {} 256 MiB regions with no copy (first: {}); {} pages read as unmapped tiles (zeros, no \
         miss), {} copies made of missed pages, {} ahead of root constants",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.tables,
        s.written,
        s.withheld,
        s.withheld_bytes as f64 / 1e6,
        s.missed_pages,
        s.missed_regions,
        missed.join(" "),
        s.unmapped_pages,
        s.copies_made,
        s.prefetched,
    );
}

/// What a shader read where it found no copy, told after its list ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissKind {
    /// Memory is mapped there, or may have been while the list ran: the
    /// console read the title's bytes there. A miss; a copy is made of it
    /// for the lists after.
    Mapped,
    /// Reserved with nothing mapped, and no mapping change reported there
    /// since the list was recorded: an unmapped tile of a reservation, which
    /// reads as zeros on the console's GPU as D3D12 has unmapped tiles of
    /// reserved resources read (tier 2 and up; D3DMetal too, docs/macos.md),
    /// and the null views read zeros. No miss.
    Unmapped,
    /// Nothing reserved: memory the title never had. A miss.
    Free,
}

/// [`MissKind`] of page `at`, with mapping changes counted from `since`
/// (`crate::watch::remap_epoch` when the list was recorded).
fn miss_kind(at: u64, since: u64) -> MissKind {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; any address may be asked.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return MissKind::Free;
    }
    // Nothing mapped now; a page whose mapping changed meanwhile may have
    // had memory while the shaders read it.
    let changed = || crate::watch::remapped_since(since, at as usize / crate::watch::PAGE);
    match info.State {
        MEM_RESERVE if !changed() => MissKind::Unmapped,
        MEM_FREE if !changed() => MissKind::Free,
        _ => MissKind::Mapped,
    }
}

/// How often a list that read title memory with no copy is run again once
/// the copies are made (`VICTORY_MEMORY_RERUN=<tries>`, 0 by default: never;
/// `Host::execute_sync`).
///
/// Holding back what such a list wrote keeps garbage out of title memory,
/// but the title goes on as if the list had done its work (the
/// acceleration-structure builder's builds are asked for once), and the
/// lists after it in the submission read what it wrote in the copies. Run
/// again with the copies it missed, from title memory's bytes, the list does
/// the console's work, one level of addresses further each time.
pub(crate) fn rerun_tries() -> u32 {
    #[cfg(test)]
    if let Some(tries) = RERUN_IN_TEST.with(std::cell::Cell::get) {
        return tries;
    }
    static TRIES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TRIES.get_or_init(|| {
        std::env::var("VICTORY_MEMORY_RERUN")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

#[cfg(test)]
thread_local! {
    /// A test's [`rerun_tries`].
    static RERUN_IN_TEST: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// A list was run `again` times more for the copies it missed; `whole`:
/// its last run missed none.
pub(crate) fn note_rerun(again: u32, whole: bool) {
    stats(|s| {
        s.reran_lists += 1;
        s.reruns += u64::from(again);
        s.rerun_gave_up += u64::from(!whole);
    });
}

/// Whether a read at a free address holds back what its list wrote
/// (`VICTORY_MISS_FREE=0`: it does not; counted all the same). A diagnostic:
/// on the console no program reads where nothing is mapped, so such a read
/// comes of data that is already wrong, and if that data is wrong because
/// earlier writes were held back, holding back keeps it wrong.
fn free_withholds() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MISS_FREE").as_deref() != Ok("0"))
}

/// Whether nothing is reserved in the whole 256 MiB region `l1` (a level-1
/// index): a read there is a read at a free address, and the region needs
/// no block to say so.
fn region_free(l1: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let at = l1 << 28;
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; any address may be asked.
    let asked = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    asked != 0
        && misses_told_apart()
        && info.State == MEM_FREE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at + (1 << 28)
}

/// Whether misses are told apart (`VICTORY_TITLE_MISSES=0`: every page read
/// with no copy is a miss, and no copy is made for it, as before).
fn misses_told_apart() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TITLE_MISSES").as_deref() != Ok("0"))
}

/// Copies made of missed pages since the last frame line, at most
/// [`MISSED_COPIES_MAX`] a frame (shaders following garbage pointers into
/// mapped memory would make one per page).
static MISSED_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const MISSED_COPIES_MAX: u64 = 64;

/// The most copies made of missed pages a frame: `VICTORY_MISSED_COPIES`,
/// [`MISSED_COPIES_MAX`] by default. The acceleration-structure builder
/// reads geometry no draw has used through a view (14,000 pages in its
/// first frame): with the default its lists would miss for hundreds of
/// frames.
fn missed_copies_max() -> u64 {
    static MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_MISSED_COPIES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(MISSED_COPIES_MAX)
    })
}

/// A copy of the mapped page `at` a shader missed, for the lists after
/// (the next page table names it): whether one was asked for.
fn copy_missed(at: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    if MISSED_COPIES.fetch_add(1, Relaxed) >= missed_copies_max() {
        return false;
    }
    crate::shadow::find(at, crate::watch::PAGE as u64).is_some()
}

/// Whether copies are made ahead of what a program using title memory is
/// handed (`VICTORY_MEMORY_PREFETCH=1`; [`prefetch`]).
pub(crate) fn prefetching() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MEMORY_PREFETCH").as_deref() == Ok("1"))
}

/// The bytes of a block of constants [`prefetch`] looks for addresses in.
const PREFETCH_BLOCK: usize = 1024;

/// Pages [`prefetch`] has looked at since the last frame line.
static PREFETCHED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

/// The 40-bit addresses in root constants `words` (`(slot, dword, value)`):
/// a dword and the low byte of the next dword of its slot, as the
/// acceleration-structure builder's programs take the address of their
/// block of constants.
fn addresses_in(words: &[(u8, u32, u32)]) -> impl Iterator<Item = u64> + '_ {
    words.iter().filter_map(|&(slot, dword, low)| {
        let &(_, _, high) = words.iter().find(|w| w.0 == slot && w.1 == dword + 1)?;
        let at = u64::from(low) | u64::from(high & 0xff) << 32;
        (at >= crate::watch::PAGE as u64).then_some(at)
    })
}

/// The 40-bit addresses among the qwords of `block`, at every dword (the
/// builder's programs load theirs at 0x4c and on).
fn addresses_of(block: &[u8]) -> impl Iterator<Item = (usize, u64)> + '_ {
    block
        .windows(8)
        .step_by(4)
        .enumerate()
        .map(|(i, q)| (i * 4, u64::from_le_bytes(q.try_into().expect("8 bytes"))))
        .filter(|&(_, at)| at >= crate::watch::PAGE as u64 && at < 1 << 40)
}

/// What Windows has at `at`: "committed", "reserved" or "free".
fn state_of(at: u64) -> &'static str {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; any address may be asked.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return "free";
    }
    match info.State {
        MEM_COMMIT => "committed",
        MEM_RESERVE => "reserved",
        _ => "free",
    }
}

/// A program using title memory is handed root constants `words` (`(slot,
/// dword, value)`): copies are made now of the memory they name and of what
/// that memory names in turn, where such memory is the title's to write.
///
/// The builder's programs take a 40-bit address from their constants, load
/// the addresses of their buffers from the block there and read and write
/// those raw: no view names any of it, so no copy exists until one of them
/// missed it, and what they computed from zeros meanwhile is withheld. A
/// guess, and only ever of what to copy: a copy of memory nothing reads
/// costs memory, a page with no copy is still a miss.
pub(crate) fn prefetch(words: &[(u8, u32, u32)]) {
    let page = crate::watch::PAGE as u64;
    let mut seen = PREFETCHED.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(HashSet::new);
    let mut made = 0;
    // Addresses of 4 GiB and up the constants name where nothing is
    // committed: `(named at, address, state)`.
    let mut nowhere: Vec<(u64, u64, &'static str)> = Vec::new();
    let mut copy = |from: u64, at: u64, seen: &mut HashSet<u64>| -> bool {
        if !seen.insert(at / page) {
            return false;
        }
        if !writable(at) {
            if at >= 1 << 32 {
                nowhere.push((from, at, state_of(at)));
            }
            return false;
        }
        let before = crate::shadow::generation();
        let found = crate::shadow::find(at, 1).is_some();
        made += u64::from(found && crate::shadow::generation() != before);
        found
    };
    for at in addresses_in(words) {
        if !copy(0, at, seen) {
            continue;
        }
        let len = (PREFETCH_BLOCK as u64).min(page - at % page) as usize & !7;
        // SAFETY: the page is committed and readable (`writable`), and the
        // bytes end in it; the title may write them meanwhile (plain bytes).
        let block: Vec<u8> = (0..len).map(|i| unsafe { ((at as usize + i) as *const u8).read_volatile() }).collect();
        for (i, named) in addresses_of(&block) {
            copy(at + i as u64, named, seen);
        }
    }
    if made > 0 || !nowhere.is_empty() {
        stats(|s| {
            s.prefetched += made;
            s.named_nowhere += nowhere.len() as u64;
            let room = 16usize.saturating_sub(s.first_nowhere.len());
            s.first_nowhere.extend(nowhere.into_iter().take(room));
        });
    }
}

/// Whether the page at `at` is committed memory the title reads and writes
/// (private or a view; not an image, not read-only).
fn writable(at: u64) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; any address may be asked.
    let asked = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    asked != 0
        && info.State == MEM_COMMIT
        && info.Type != MEM_IMAGE
        && info.Protect == PAGE_READWRITE
}

/// What the memory at `at` is now, for the missed addresses of the frame
/// line (diagnostics): whether a copy could be made of it (committed, and a
/// view of a section or private memory) or the shaders read past what the
/// title maps (reserved, free).
#[cfg(feature = "diagnostics")]
fn memory_kind(at: u64) -> String {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEM_RESERVE,
        MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; any address may be asked.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return "(not asked)".into();
    }
    let state = match info.State {
        MEM_COMMIT => "committed",
        MEM_RESERVE => "reserved",
        MEM_FREE => "free",
        _ => "?",
    };
    let kind = match info.Type {
        MEM_MAPPED => " view",
        MEM_PRIVATE => " private",
        MEM_IMAGE => " image",
        _ => "",
    };
    format!("({state}{kind}, protect {:#x})", info.Protect)
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

impl Shared {
    /// A page table for `heap`, named nothing yet (written at the first
    /// submission that uses it).
    fn new(heap: &HostHeap) -> Option<Shared> {
        let bytes = (LEVEL1 + (CAPACITY + 1) * BLOCK * 2) as u64 * 4;
        let (table, data) = crate::host::host()?.upload_buffer(bytes)?;
        crate::host::name(&table, || "title memory page table".into());
        let null = heap.pair()?;
        heap.raw_pair_at(null, None, 0, 0);
        // SAFETY: a live buffer.
        let gpu = unsafe { table.GetGPUVirtualAddress() };
        Some(Shared {
            host: std::ptr::from_ref(heap) as usize,
            table,
            data: data as usize,
            gpu,
            null,
            pairs: HashMap::new(),
            generation: u64::MAX,
            regions: 0,
            layout: Arc::default(),
            scratch: pagetable::Scratch::default(),
        })
    }

    /// Names every host copy there is now (`now`: their generation): view
    /// pairs made for new copies, those of copies gone made null and given
    /// back, the table rewritten. Only while no submission runs (the GPU
    /// reads the table and the views).
    fn rewrite(&mut self, heap: &HostHeap, now: u64) {
        let copies = crate::shadow::all();
        let live: HashSet<usize> = copies.iter().map(|c| Arc::as_ptr(c) as usize).collect();
        self.pairs.retain(|(copy, _), &mut slot| {
            let keep = live.contains(copy);
            if !keep {
                heap.raw_pair_at(slot, None, 0, 0);
                heap.free_pair(slot);
            }
            keep
        });
        let (mut named, mut slots) = (Vec::with_capacity(copies.len()), Vec::with_capacity(copies.len()));
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(copies.len());
        for c in copies {
            let key = Arc::as_ptr(&c) as usize;
            // A copy longer than a view holds is named in pieces, each
            // with views of its own from where it starts in the copy's
            // buffer (which need not be the buffer's start: memory in
            // place lies inside its heap's).
            for (n, (start, len, viewed)) in pieces(c.range()).enumerate() {
                let slot = match self.pairs.get(&(key, n as u32)) {
                    Some(&slot) => slot,
                    None => {
                        let Some(slot) = heap.pair() else {
                            static ONCE: std::sync::Once = std::sync::Once::new();
                            ONCE.call_once(|| tracing::warn!("no descriptor slots for a title memory copy's views: it reads as missing"));
                            continue;
                        };
                        heap.raw_pair_at(slot, Some(&c.buffer), c.offset(start), viewed.next_multiple_of(4));
                        self.pairs.insert((key, n as u32), slot);
                        slot
                    }
                };
                named.push(c.clone());
                slots.push(slot);
                ranges.push((start, len));
            }
        }
        // The last table's blocks and owners, built back into if no list
        // still reads them (a reader clones the layout, under the lock
        // this holds); their buffers are kept between rewrites.
        let spent = std::mem::take(&mut self.layout);
        let mut out = match Arc::try_unwrap(spent) {
            Ok(Layout { blocks, owner, .. }) => pagetable::Outputs { blocks, owners: owner },
            Err(_) => pagetable::Outputs::default(),
        };
        let regions = missed_regions();
        let built = self
            .scratch
            .build_with_regions(&ranges, &slots, self.null, &regions, &mut out);
        // SAFETY: the mapping has room for `CAPACITY` blocks, which the
        // builder keeps to; no submission runs.
        unsafe { std::ptr::copy_nonoverlapping(built.dwords.as_ptr(), self.data as *mut u32, built.dwords.len()) };
        tracing::debug!(copies = ranges.len(), blocks = built.blocks.len() - 1, "title memory page table");
        if built.full {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(CAPACITY, "title memory copies in more 256 MiB regions than a page table has room for: the rest read as missing"));
        }
        self.generation = now;
        self.regions = regions.len();
        self.layout = Arc::new(Layout {
            bits: named.iter().map(|c| c.dirty_bit()).collect(),
            copies: named,
            blocks: out.blocks,
            owner: out.owners,
        });
        stats(|s| s.tables += 1);
    }
}

/// The page table of view heap `key`, as the last submission using it
/// left it.
fn layout(key: usize) -> Arc<Layout> {
    shared()
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, s)| s.layout.clone())
        .unwrap_or_default()
}

/// At submission, for a list using title memory with view heap `key`
/// bound: the heap's page table names every host copy there is, each
/// brought up to date (`add` takes a copy's range, once per submission
/// until a step writes title memory).
pub(crate) fn prepare(key: usize, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let layout = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        let (_, s) = shared.iter_mut().find(|(k, s)| *k == key && s.host == host)?;
        let now = crate::shadow::generation();
        if s.generation != now || s.regions != missed_regions_count() {
            s.rewrite(heap, now);
        }
        Some(s.layout.clone())
    })
    .flatten();
    // Copies the title has not written since they took it are passed by
    // without touching them (the table names every copy, ~4,700 in the world).
    for l in layout.iter() {
        let mut last = std::ptr::null();
        for (c, bit) in l.copies.iter().zip(&l.bits) {
            // A copy named in pieces is there once per piece, in a row.
            if std::mem::replace(&mut last, Arc::as_ptr(c)) == Arc::as_ptr(c) {
                continue;
            }
            if !c.clean() && !crate::watch::clean_bit(*bit) {
                let (start, len) = c.range();
                add(c, start as usize, len as usize);
            }
        }
    }
}

/// Whether any marked page in `marks` fell outside a valid copy in `l`.
/// The 256 MiB regions with no copy shaders read (level-1 indices), at
/// most [`MISSED_REGIONS_MAX`]: the page tables give each a block of
/// entries that name no copy, which read as missing as the null block
/// does, so that the next miss there names its page, not the region.
static MISSED_REGIONS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
const MISSED_REGIONS_MAX: usize = 64;

/// The most missed regions given a block: `VICTORY_MISSED_REGIONS`,
/// [`MISSED_REGIONS_MAX`] by default, at most half the page table's blocks
/// (the copies' regions need the rest).
fn missed_regions_max() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_MISSED_REGIONS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(MISSED_REGIONS_MAX)
            .min(CAPACITY / 2)
    })
}
static MISSED_REGIONS_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn missed_regions() -> Vec<usize> {
    MISSED_REGIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

fn missed_regions_count() -> usize {
    MISSED_REGIONS_COUNT.load(std::sync::atomic::Ordering::Acquire)
}

/// Shaders read 256 MiB region `l1` (a level-1 index), which no copy and
/// so no block has.
fn note_missed_region(l1: usize) {
    let mut regions = MISSED_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    if l1 < LEVEL1 && regions.len() < missed_regions_max() && !regions.contains(&l1) {
        regions.push(l1);
        MISSED_REGIONS_COUNT.store(regions.len(), std::sync::atomic::Ordering::Release);
    }
}

fn has_missed(marks: &[u32], l: &Layout, kind: impl Fn(u64) -> MissKind) -> bool {
    for (w, &mark) in marks.iter().enumerate() {
        let mut bits = mark;
        while bits != 0 {
            let entry = w * 32 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let (block, page) = (entry / BLOCK, entry % BLOCK);
            let Some(&l1) = l.blocks.get(block).filter(|_| block != 0) else {
                if block == 0 && region_free(entry) && !free_withholds() {
                    continue;
                }
                return true;
            };
            if l.owner.get(entry).is_none_or(|o| o.is_none()) {
                match kind((l1 << 28 | page << 16) as u64) {
                    MissKind::Unmapped => {}
                    MissKind::Free if !free_withholds() => {}
                    _ => return true,
                }
            }
        }
    }
    false
}

/// `list` uses title memory with view heap `key` bound: the heap's page
/// table (made on first use) and a page bitmap of the list's, the table
/// brought up to date at each submission of the list, its written pages
/// copied back after.
pub(crate) fn bind(list: &HostList, key: usize) -> Option<Map> {
    let table = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        if let Some((_, s)) = shared.iter().find(|(k, s)| *k == key && s.host == host) {
            list.keep(&s.table);
            return Some(s.gpu);
        }
        // A released heap's table goes with it (lists that name it hold it).
        shared.retain(|(k, _)| *k != key);
        let s = Shared::new(heap)?;
        list.keep(&s.table);
        let gpu = s.gpu;
        shared.push((key, s));
        Some(gpu)
    })
    .flatten()?;
    let pooled = Arc::new(PooledBitmap::new(Bitmap::acquire()?));
    let dirty = pooled.gpu();
    list.defer_memory(key);
    let precise: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let byte_for_byte = precise.clone();
    let missed_bitmap = Arc::clone(&pooled);
    // Mapping changes from now on: a page found with nothing mapped after
    // the list ran may have had memory while it ran if one touched it.
    let since = crate::watch::remap_epoch();
    let kind = move |at: u64| {
        if misses_told_apart() {
            miss_kind(at, since)
        } else {
            MissKind::Mapped
        }
    };
    list.defer_missed(move || {
        let l = layout(key);
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts(missed_bitmap.data() as *const u32, BITMAP_WORDS) };
        has_missed(marks, &l, kind)
    });
    let again_bitmap = Arc::clone(&pooled);
    list.defer_again(key, move |clear| {
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks = unsafe {
            std::slice::from_raw_parts_mut(again_bitmap.data() as *mut u32, BITMAP_WORDS)
        };
        if clear {
            marks.fill(0);
            return false;
        }
        let Layout { blocks, owner, .. } = &*layout(key);
        let mut changed = false;
        for (w, &mark) in marks.iter().enumerate() {
            let mut bits = mark;
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    if block == 0 && !region_free(entry) {
                        let before = missed_regions_count();
                        note_missed_region(entry);
                        changed |= missed_regions_count() != before;
                    }
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                match owner.get(entry) {
                    // What the list wrote leaves the copy: the next run
                    // starts from title memory, as the first did.
                    Some(&Some(_)) => crate::watch::retake(at as usize, 1 << 16),
                    _ => {
                        if kind(at) == MissKind::Mapped && misses_told_apart() {
                            let before = crate::shadow::generation();
                            changed |= copy_missed(at) && crate::shadow::generation() != before;
                        }
                    }
                }
            }
        }
        changed
    });
    list.defer_after(move || {
        let Layout { copies, blocks, owner, .. } = &*layout(key);
        let precise = byte_for_byte.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(pooled.data() as *mut u32, BITMAP_WORDS) };
        let (mut pages, mut regions, mut missed) = (0, 0, Vec::new());
        let (mut unmapped, mut copied) = (0, 0);
        // Misses that hold back what the list wrote.
        let mut holding = 0u64;
        let mut by_region: Vec<(u64, usize)> = Vec::new();
        let mut writes = Vec::new();
        for (w, mark) in marks.iter_mut().enumerate() {
            // Most words are 0: write back only the ones that are not.
            if *mark == 0 {
                continue;
            }
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                // Block 0's bits are 256 MiB regions with no block.
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    missed.push((entry as u64) << 28);
                    if block == 0 && region_free(entry) {
                        // Nothing in the whole region: a read at a free
                        // address, whichever page it was.
                        pages += 1;
                        by_region.push(((entry as u64) << 28, 1));
                        holding += u64::from(free_withholds());
                        continue;
                    }
                    regions += 1;
                    holding += 1;
                    by_region.push(((entry as u64) << 28, 2));
                    note_missed_region(entry);
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                match owner.get(entry) {
                    Some(_) if precise.iter().any(|&(a, n)| a < at + (1 << 16) && at < a + n) => {}
                    Some(&Some(k)) => writes.push((k, at)),
                    _ => match kind(at) {
                        MissKind::Unmapped => unmapped += 1,
                        k => {
                            pages += 1;
                            holding += u64::from(k != MissKind::Free || free_withholds());
                            missed.push(at);
                            by_region.push((at & !((1 << 28) - 1), usize::from(k == MissKind::Free)));
                            if k == MissKind::Mapped && misses_told_apart() && copy_missed(at) {
                                copied += 1;
                            }
                        }
                    },
                }
            }
        }
        // What shaders computed from memory with no copy (read as zeros) is
        // not the console's: it does not go into title memory, where it broke
        // the title's own data (garbage pointers in title threads, run c1),
        // nor while any list using title memory misses ([`withholding`]).
        let withheld = if holding > 0 || withholding() { writes.len() as u64 } else { 0 };
        if withheld > 0 {
            for &(_, at) in &writes {
                drop_withheld(at as usize, 1 << 16);
            }
        }
        let mut written = 0;
        for &(k, at) in writes.iter().take_while(|_| withheld == 0) {
            let n = copies[k].copy_back_page(at as usize);
            tracing::debug!(
                page = format_args!("{at:#x}"),
                copy = format_args!("{:#x?}", copies[k].range()),
                bytes = n,
                "title memory page written back"
            );
            written += 1;
        }
        stats(|s| {
            s.written += written;
            s.withheld += withheld;
            s.missed_pages += pages;
            s.missed_regions += regions;
            s.unmapped_pages += unmapped;
            s.copies_made += copied;
            let room = 4usize.saturating_sub(s.first_missed.len());
            s.first_missed.extend(missed.into_iter().take(room));
            for (region, kind) in by_region {
                s.missed_by_region.entry(region).or_default()[kind] += 1;
            }
        });
    });
    Some(Map {
        table,
        dirty,
        precise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HostGpu;
    use victory_formats::pagetable::table;

    #[test]
    fn memory_stats_follow_the_frame_report_filter() {
        use tracing_subscriber::layer::SubscriberExt;

        for (filter, expected) in [("warn,victory_gpu::host=info", false), ("info", true)] {
            let subscriber =
                tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::new(filter));
            tracing::subscriber::with_default(subscriber, || {
                assert_eq!(stats(|_| true), expected);
            });
        }
    }

    /// An address with nothing reserved near it: the middle of the largest
    /// free region of the process. A fixed high address can be reserved on
    /// Windows (the control-flow-guard bitmap takes terabytes up there),
    /// and an allocation just released can be taken again by another test.
    fn free_address() -> u64 {
        use windows_sys::Win32::System::Memory::{
            MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        let (mut at, mut best) = (0usize, (0usize, 0usize));
        loop {
            // SAFETY: a plain structure of integers and pointers; zero is valid.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: the query writes its own structure of the given size.
            if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
                break;
            }
            let (base, size) = (info.BaseAddress as usize, info.RegionSize);
            if info.State == MEM_FREE && size > best.1 {
                best = (base, size);
            }
            match base.checked_add(size) {
                Some(next) if next > at => at = next,
                _ => break,
            }
        }
        assert!(best.1 >= 1 << 20, "no free region of a MiB");
        (best.0 + best.1 / 2) as u64 & !0xFFFF
    }

    /// The frame line names what a missed address is: committed private
    /// memory here, nothing at all in the largest free region.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn missed_addresses_are_named_by_their_memory() {
        let ours = vec![0u8; 1 << 16];
        assert!(memory_kind(ours.as_ptr() as u64).starts_with("(committed private"));
        assert!(memory_kind(free_address()).starts_with("(free"));
    }

    /// A region missed is noted once, and the count the page tables compare
    /// moves with it; past the index or the cap nothing is noted.
    #[test]
    fn missed_regions_are_noted_once() {
        let before = missed_regions_count();
        note_missed_region(LEVEL1 - 7);
        note_missed_region(LEVEL1 - 7);
        note_missed_region(LEVEL1);
        assert!(missed_regions().contains(&(LEVEL1 - 7)));
        assert!(!missed_regions().contains(&LEVEL1));
        assert!(missed_regions_count() <= before + 1);
        assert!(missed_regions_count() <= missed_regions_max());
    }

    /// Writes of lists using title memory are held back from a frame with a
    /// miss through the next one, then go back again.
    #[test]
    fn a_miss_holds_back_its_frame_and_the_next() {
        assert!(!withheld_in(0, MISS_NEVER));
        assert!(!withheld_in(u64::MAX, MISS_NEVER));
        assert!(withheld_in(7, 7));
        assert!(withheld_in(8, 7), "the next frame reads this one's results");
        assert!(!withheld_in(9, 7));
    }

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }

    /// Copies in more regions than a table has room for: the largest are
    /// named, the rest read as missing; the table fits its buffer.
    #[test]
    fn a_full_table_names_the_largest_copies() {
        let copies: Vec<(u64, u64)> = (0..CAPACITY as u64 + 2).map(|k| (k << 28, 0x1_0000 + k)).collect();
        let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 10 + 2 * k).collect();
        let (dwords, blocks, _) = table(&copies, &pairs, 4);
        assert_eq!(blocks.len(), CAPACITY + 1);
        assert_eq!(dwords.len(), LEVEL1 + (CAPACITY + 1) * BLOCK * 2);
        assert_eq!(look_up(&dwords, (CAPACITY as u64 + 1) << 28).0, pairs[CAPACITY + 1]);
        assert_eq!(look_up(&dwords, 0).0, 4, "the smallest copy is left out");
    }

    #[test]
    fn table_skips_overflowing_and_out_of_range_copies() {
        let copies = [
            (u64::MAX - 0x100, 0x200),
            (1u64 << 40, 0x1_0000),
            (0x40_0000, 0x1_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        assert_eq!(blocks, vec![usize::MAX, 0]);
        assert_eq!(look_up(&dwords, 0x40_0010), (106, 0x10));
        assert_eq!(look_up(&dwords, 0).0, 100);
        assert_eq!(owner[BLOCK + 0x40], Some(2));
    }

    #[test]
    fn has_missed_checks_block_zero_out_of_range_unowned_and_owned() {
        let mut owner = vec![None; 2 * BLOCK];
        owner[BLOCK + 5] = Some(0);
        let l = Layout {
            copies: Vec::new(),
            bits: Vec::new(),
            blocks: vec![usize::MAX, 0],
            owner,
        };
        let mapped = |_| MissKind::Mapped;
        let mut marks = vec![0u32; BITMAP_WORDS];
        assert!(!has_missed(&marks, &l, mapped));

        // Owned entry in block 1.
        let owned = BLOCK + 5;
        marks[owned / 32] = 1 << (owned % 32);
        assert!(!has_missed(&marks, &l, mapped));

        // Unowned entry in the same word as an owned entry: a miss where
        // something is (or was) mapped, or nothing reserved; none where an
        // unmapped tile reads zeros.
        let unowned = BLOCK + 6;
        marks[unowned / 32] |= 1 << (unowned % 32);
        assert!(has_missed(&marks, &l, mapped));
        assert!(has_missed(&marks, &l, |_| MissKind::Free));
        assert!(!has_missed(&marks, &l, |at| {
            assert_eq!(at, 6 << 16, "the page's address");
            MissKind::Unmapped
        }));

        // Block 0 bit (missing 256 MiB region): its page is not known.
        marks.fill(0);
        marks[0] = 1 << 3;
        assert!(has_missed(&marks, &l, |_| MissKind::Unmapped));

        // Out-of-range block.
        marks.fill(0);
        let oob = 2 * BLOCK + 1;
        marks[oob / 32] = 1 << (oob % 32);
        assert!(has_missed(&marks, &l, |_| MissKind::Unmapped));
    }

    /// A page read with no copy is told by its memory: committed memory is
    /// mapped (copies are made of it), a reservation with nothing mapped is
    /// an unmapped tile (zeros, no miss) unless the runtime reported a
    /// mapping change there since the list was recorded, and nothing
    /// reserved is free.
    #[test]
    fn missed_pages_are_told_by_their_memory() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let page = crate::watch::PAGE;
        // SAFETY: fresh allocations of ours, freed below.
        let (committed, reserved) = unsafe {
            (
                VirtualAlloc(
                    std::ptr::null(),
                    page,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                ),
                VirtualAlloc(std::ptr::null(), 2 * page, MEM_RESERVE, PAGE_NOACCESS),
            )
        };
        assert!(!committed.is_null() && !reserved.is_null());
        let (committed, reserved) = (committed as u64, reserved as u64);
        let since = crate::watch::remap_epoch();
        assert_eq!(miss_kind(committed, since), MissKind::Mapped);
        assert_eq!(miss_kind(reserved, since), MissKind::Unmapped);
        assert_eq!(miss_kind(free_address(), since), MissKind::Free);
        // The title's memory there was reported unmapped meanwhile: it may
        // have had memory while the shaders read it.
        drop(crate::watch::title_memory_unmapping(
            reserved as usize,
            page,
        ));
        assert_eq!(miss_kind(reserved, since), MissKind::Mapped);
        assert_eq!(
            miss_kind(reserved + page as u64, since),
            MissKind::Unmapped,
            "the next page was not"
        );
        // SAFETY: ours, as allocated.
        unsafe {
            VirtualFree(committed as *mut _, 0, MEM_RELEASE);
            VirtualFree(reserved as *mut _, 0, MEM_RELEASE);
        }
    }

    /// A mapped page a shader missed gets a copy (the next page table names
    /// it), up to the frame's limit.
    #[test]
    fn a_mapped_page_missed_gets_a_copy() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let page = crate::watch::PAGE;
        // SAFETY: a fresh allocation of ours, kept (its copy may outlive the test).
        let at = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                page,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!at.is_null());
        let at = at as u64;
        let before = crate::shadow::generation();
        MISSED_COPIES.store(0, Relaxed);
        assert!(copy_missed(at), "a copy for the lists after");
        let copy = crate::shadow::find(at, page as u64).expect("found again");
        assert!(copy.range().0 <= at && at + page as u64 <= copy.range().0 + copy.range().1);
        assert!(crate::shadow::generation() >= before);
        MISSED_COPIES.store(MISSED_COPIES_MAX, Relaxed);
        assert!(!copy_missed(at), "past the frame's limit");
        MISSED_COPIES.store(0, Relaxed);
    }

    /// Root constants name a block, the block names a buffer: both get
    /// copies before any shader ran; small numbers, an address past 40 bits
    /// and unmapped memory do not.
    #[test]
    fn memory_named_by_root_constants_is_copied_ahead() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        let words = [(2u8, 0u32, 0x1234_0000u32), (2, 1, 0xabcd_ef05), (3, 0, 7), (3, 1, 0)];
        assert_eq!(addresses_in(&words).collect::<Vec<_>>(), [0x5_1234_0000]);
        let mut block = vec![0u8; 4];
        for q in [0u64, 0x200, 0x7_0000_1000, 1 << 40, 0xff_ffff_fff8] {
            block.extend_from_slice(&q.to_le_bytes());
            block.extend_from_slice(&[0; 8]);
        }
        assert_eq!(addresses_of(&block).collect::<Vec<_>>(), [(36, 0x7_0000_1000), (68, 0xff_ffff_fff8)]);
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let page = crate::watch::PAGE;
        // SAFETY: fresh allocations of ours, kept (their copies may outlive the test).
        let alloc = || unsafe {
            VirtualAlloc(std::ptr::null(), page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as u64
        };
        let (constants, buffer) = (alloc(), alloc());
        assert!(constants != 0 && buffer != 0 && buffer < 1 << 40 && constants < 1 << 40);
        // SAFETY: inside the fresh allocation.
        unsafe { ((constants + 0x4c) as *mut u64).write_unaligned(buffer + 0x80) };
        let before = crate::shadow::generation();
        prefetch(&[(0, 4, constants as u32 + 0x20), (0, 5, (constants >> 32) as u32 | 0x100)]);
        assert!(crate::shadow::generation() != before, "copies made");
        for at in [constants, buffer] {
            let before = crate::shadow::generation();
            crate::shadow::find(at, 1).expect("a copy");
            assert_eq!(crate::shadow::generation(), before, "made ahead: {at:#x}");
        }
        let before = crate::shadow::generation();
        prefetch(&[(0, 0, 0x30_0000), (0, 1, 0xff)]);
        assert_eq!(crate::shadow::generation(), before, "nothing mapped there");
    }

    fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn bitmap_pool_reuses_and_zeroes_across_resets() {
        let _guard = gpu_test_lock();
        let (Some(host), Some(_views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let a = host.new_list().unwrap();
        let b = host.new_list().unwrap();
        let map_a = bind(&a, OURS).expect("bitmap for list a");
        let map_b = bind(&b, OURS).expect("bitmap for list b");
        let va_a = map_a.dirty;
        let va_b = map_b.dirty;
        assert_ne!(va_a, va_b, "simultaneously open lists get distinct bitmaps");
        drop((map_a, map_b));
        a.close();
        a.reset();
        let map_a2 = bind(&a, OURS).expect("reused bitmap for list a");
        assert_eq!(map_a2.dirty, va_a, "reset returns bitmap to pool for reuse");
        assert_ne!(map_a2.dirty, va_b, "list b still holds its bitmap");
        drop(map_a2);
        a.close();
        b.close();
        a.reset();
        b.reset();

        // Also verify PooledBitmap zeroes dirty words when dropped without executing.
        let pooled = PooledBitmap::new(Bitmap::acquire().unwrap());
        let data = pooled.data();
        // SAFETY: exclusive ownership of this newly acquired bitmap.
        unsafe { (data as *mut u32).add(7).write(0xDEAD_BEEF) };
        drop(pooled);
        let reused = Bitmap::acquire().unwrap();
        // SAFETY: `reused.data` is mapped for `BITMAP_WORDS` dwords.
        let words = unsafe { std::slice::from_raw_parts(reused.data as *const u32, BITMAP_WORDS) };
        assert!(
            words.iter().all(|&w| w == 0),
            "pooled bitmap must be zeroed on return"
        );
        drop(PooledBitmap::new(reused));
    }

    /// Reads 4 dwords of title memory at (`lo`, `hi`) through the page
    /// table and the heap's views, as translated shaders do, into `dst`
    /// from byte `at`.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=3, b0)"
ByteAddressBuffer views[] : register(t0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; uint at; }
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint a = lo + id.x * 4;
    bool inside = hi < 256;
    uint l1 = inside ? (hi << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    dst.Store(at + id.x * 4, views[NonUniformResourceIndex(e.x)].Load(a - e.y));
}
"#;

    /// The pipeline, or none where the HLSL compiler cannot build its
    /// array of views: Wine's (vkd3d-shader) has no shader model 5.1
    /// resource arrays and answers `E_NOTIMPL`.
    fn pipeline(source: &str) -> Option<(
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    )> {
        use windows::Win32::Foundation::E_NOTIMPL;
        use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES, D3DCompile};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call; the errors are text.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("memory"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES,
                0,
                &mut code,
                Some(&mut errors),
            );
            if let Err(e) = &made
                && e.code() == E_NOTIMPL
            {
                eprintln!("skipped: the HLSL compiler lacks what the shader needs (Wine's has no shader model 5.1 resource arrays)");
                return None;
            }
            let text = errors.map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned());
            assert!(made.is_ok(), "{text:?}");
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes; a compiled shader with its root
        // signature inside.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            Some((root, pso))
        }
    }

    /// A copy's view pair, in view heap `key`'s table.
    fn pair_of(key: usize, copy: &Arc<Shadow>) -> Option<u32> {
        shared()
            .iter()
            .find(|(k, _)| *k == key)?
            .1
            .pairs
            .get(&(Arc::as_ptr(copy) as usize, 0))
            .copied()
    }

    /// A pool of title memory (below 2^40, as the runtime places it),
    /// `pages` 64 KiB pages, write-watched as its pools.
    fn pool(pages: usize) -> usize {
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE, MemExtendedParameterAddressRequirements,
            PAGE_READWRITE, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // Placed by address requirements, as the runtime places title
        // memory (`XMemVirtualAlloc`): Wine refuses fixed addresses there.
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: a fresh allocation; the parameter points at
        // `requirements`, which outlives the call.
        let p = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        } as usize;
        assert!(p != 0 && p < 1 << 40, "memory below 2^40");
        p
    }

    /// Two lists of a submission read title memory through one page table;
    /// the copies' views stay where they are for later lists and
    /// submissions, also when another copy comes; none takes a per-draw
    /// dynamic slot.
    #[test]
    fn lists_share_a_table_and_copies_keep_their_views() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let Some((root, pso)) = pipeline(SOURCE) else {
            return; // no compiler for the shader here
        };
        let (out, at) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // Title memory: 1, 2 and (from the second submission on) 1 page.
        let memory = [pool(1), pool(2), pool(1)];
        let copy = |p: usize, pages: usize| {
            crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
            crate::shadow::find(p as u64, (pages * PAGE) as u64).unwrap()
        };
        let (a, b) = (copy(memory[0], 1), copy(memory[1], 2));
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        // Each list reads 4 dwords at its address into its 16 bytes of `out`.
        let record = |reads: [usize; 2]| -> Vec<Map> {
            let maps: Vec<Map> = lists.iter().map(|l| bind(l, OURS).expect("a page table")).collect();
            for (k, (list, &address)) in lists.iter().zip(&reads).enumerate() {
                let constants = [address as u32, (address >> 32) as u32, 16 * k as u32];
                // SAFETY: an open list, live objects of ours.
                unsafe {
                    list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                    list.list.SetComputeRootShaderResourceView(1, maps[k].table);
                    list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    list.list.SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
                    list.list.SetPipelineState(&pso);
                    list.list.Dispatch(1, 1, 1);
                }
                list.close();
            }
            maps
        };
        // The title writes after recording, before submitting.
        let fill = |p: usize, base: u32| {
            for k in 0..4 {
                // SAFETY: our pool (a guarded page faults into the handler).
                unsafe { (p as *mut u32).add(k).write_volatile(base + k as u32) };
            }
        };
        let read = || -> Vec<u32> {
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { std::slice::from_raw_parts(at.cast::<u32>(), 8).to_vec() }
        };
        let maps = record([memory[0] + 16, memory[1] + PAGE]);
        assert_eq!(maps[0].table, maps[1].table, "one table for the heap");
        assert_ne!(maps[0].dirty, maps[1].dirty, "a bitmap per list");
        fill(memory[0] + 16, 0x100);
        fill(memory[1] + PAGE, 0x200);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x100, 0x101, 0x102, 0x103, 0x200, 0x201, 0x202, 0x203]);
        let pairs = (pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap());
        assert!(pairs.0 >= 1 << 18 && pairs.1 >= 1 << 18, "past the dynamic slots");
        // A second submission, after a third copy came.
        let c = copy(memory[2], 1);
        for list in &lists {
            list.reset();
        }
        record([memory[2] + 32, memory[0] + 16]);
        fill(memory[2] + 32, 0x300);
        fill(memory[0] + 16, 0x400);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x300, 0x301, 0x302, 0x303, 0x400, 0x401, 0x402, 0x403]);
        assert_eq!((pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap()), pairs, "views kept");
        assert!(pair_of(OURS, &c).is_some());
        for (&p, copy) in memory.iter().zip([a, b, c]) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
        }
    }

    /// Writes `value` at (`lo`, `hi`) through the page table, marking its
    /// page as translated shaders do, and with `missing_hi` non-zero reads
    /// (`missing_lo`, `missing_hi`), marking it if no copy holds it.
    const WRITE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0, space=2), RootConstants(num32BitConstants=5, b0)"
RWByteAddressBuffer views[] : register(u0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0, space2);
cbuffer c : register(b0) { uint lo; uint hi; uint value; uint missing_lo; uint missing_hi; }
void access(uint a, uint h, bool write) {
    bool inside = h < 256;
    uint l1 = inside ? (h << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    bool miss = e.x == table.Load(4096 * 4);
    uint n = miss && block == 0 ? l1 : entry;
    if (write || miss) {
        dirty.InterlockedOr(n >> 5 << 2, 1u << (n & 31));
    }
    if (write && !miss) {
        views[NonUniformResourceIndex(e.x + 1)].Store(a - e.y, value);
    }
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    access(lo, hi, true);
    if (missing_hi != 0) {
        access(missing_lo, missing_hi, false);
    }
}
"#;

    /// What a submission that missed memory wrote through the page table
    /// leaves the host copies right after it, so the next submission reads
    /// title memory's bytes (run shards20: garbage BVH nodes over meshlet
    /// buffers until the frame's end); one that missed nothing leaves its
    /// writes in the copies.
    #[test]
    fn a_submission_that_missed_leaves_the_copies_at_once() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let (Some((write_root, writer)), Some((read_root, reader))) = (pipeline(WRITE), pipeline(SOURCE)) else {
            return; // no compiler for the shader here
        };
        let (out, out_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // A page of title memory below 2^40, write-watched as the pools.
        let p = (0..256u64)
            .map(|k| 0x50_0000_0000 + (k << 28))
            // SAFETY: a fresh allocation where nothing is mapped, or none.
            .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
            .find(|&p| p != 0)
            .expect("memory below 2^40");
        let at = p + 16;
        // SAFETY: our pool, not guarded yet.
        unsafe { (at as *mut u32).write_volatile(0x11) };
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        // One submission of one dispatch; the list is open again after it.
        let submit = |root: &windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
                      pso: &windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
                      uav: Option<u64>,
                      constants: &[u32]| {
            let map = bind(&list, OURS).expect("a page table");
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, uav.unwrap_or(map.dirty));
                list.list.SetComputeRoot32BitConstants(3, constants.len() as u32, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
        };
        let write = |value: u32, miss: bool| {
            // Memory no copy holds, in a 256 MiB region with no block.
            let missing: u64 = if miss { 0xff_f000_0000 } else { 0 };
            let constants = [at as u32, (at >> 32) as u32, value, missing as u32, (missing >> 32) as u32];
            submit(&write_root, &writer, None, &constants);
        };
        let read = || -> u32 {
            submit(&read_root, &reader, Some(out_gpu), &[at as u32, (at >> 32) as u32, 0]);
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { out_cpu.cast::<u32>().read_volatile() }
        };
        // SAFETY: our pool.
        let title = || unsafe { (at as *const u32).read_volatile() };
        write(0x33, true);
        // Another test's submission may have refreshed the copy already.
        let held = copy.read_host(at as u64, 1)[0];
        assert!(!copy.clean() || held == 0x11, "the written page counts as changed at once");
        assert_eq!(title(), 0x11, "nothing went back");
        assert_eq!(read(), 0x11, "the next submission reads title memory's bytes");
        // Unless another test ends a frame meanwhile (its retake is right).
        let frame = FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        write(0x44, false);
        let kept = read();
        if FRAMES.load(std::sync::atomic::Ordering::Relaxed) == frame {
            assert_eq!(kept, 0x44, "a submission that missed nothing keeps its writes");
        }
        crate::shadow::unregister(p, p as u64);
        drop(copy);
    }

    const FOLLOW: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0, space=2), RootConstants(num32BitConstants=4, b0)"
RWByteAddressBuffer views[] : register(u0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0, space2);
cbuffer c : register(b0) { uint lo; uint hi; uint from_lo; uint from_hi; }
uint access(uint a, uint h, bool write, uint value) {
    bool inside = h < 256;
    uint l1 = inside ? (h << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    bool miss = e.x == table.Load(4096 * 4);
    uint n = miss && block == 0 ? l1 : entry;
    if (write || miss) {
        dirty.InterlockedOr(n >> 5 << 2, 1u << (n & 31));
    }
    uint read = 0;
    if (!miss) {
        if (write) {
            views[NonUniformResourceIndex(e.x + 1)].Store(a - e.y, value);
        } else {
            read = views[NonUniformResourceIndex(e.x)].Load(a - e.y);
        }
    }
    return read;
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    // The address to store at is read from memory, as the builder's are.
    uint offset = access(from_lo, from_hi, false, 0);
    access(lo + offset, hi, true, 0x70 + offset);
}
"#;

    /// A list that read memory with no copy is run again once the copy is
    /// made (`VICTORY_MEMORY_RERUN`): what its first run stored where zeros
    /// sent it leaves the copies, and its last run's stores, made from the
    /// title's bytes, go back in the same submission.
    #[test]
    fn a_list_that_missed_runs_again_with_the_copies_it_missed() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let Some((root, pso)) = pipeline(FOLLOW) else {
            return; // no compiler for the shader here
        };
        let alloc = |base: u64, flags: u32| {
            (0..256u64)
                .map(|k| base + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | flags, PAGE_READWRITE) as usize })
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        // Where the list stores: a page with a copy. What names the offset
        // there: memory nothing has a copy of, in a region with no block.
        let p = alloc(0x52_0000_0000, crate::watch::tests::MEM_WRITE_WATCH);
        let named = alloc(0x62_0000_0000, 0);
        // SAFETY: our fresh pages.
        unsafe {
            (p as *mut u8).write_bytes(0x11, 64);
            ((named + 8) as *mut u32).write_volatile(16);
        }
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        // No miss of another test's holds this one's writes back.
        log_frame();
        log_frame();
        let list = host.new_list().unwrap();
        let run = |tries: u32| {
            RERUN_IN_TEST.with(|t| t.set(Some(tries)));
            let map = bind(&list, OURS).expect("a page table");
            let from = named as u64 + 8;
            let constants = [p as u32, (p as u64 >> 32) as u32, from as u32, (from >> 32) as u32];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, map.dirty);
                list.list.SetComputeRoot32BitConstants(3, 4, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
            RERUN_IN_TEST.with(|t| t.set(None));
        };
        // SAFETY: our page.
        let title = |offset: usize| unsafe { ((p + offset) as *const u32).read_volatile() };
        run(4);
        assert_eq!(title(16), 0x80, "the store made from the title's bytes went back");
        assert_eq!(title(0), 0x1111_1111, "the store made from zeros did not");
        assert_eq!(copy.read_host(p as u64, 4), [0x11; 4], "nor does the copy keep it");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
    }

    /// A title-memory program reads a copy longer than a raw view holds
    /// (512 MiB, 2^27 elements) past its first 512 MiB as anywhere else:
    /// the page table's entry for the address, and the view it names, give
    /// the title's bytes there, not zeros. #322's aliased regions are this
    /// long. Two programs: one reads the entry, the other reads through the
    /// view it names, as translated shaders index `views[entry]`.
    #[test]
    fn a_long_copy_is_read_past_its_first_512_mib() {
        use crate::compute::{Param, pipeline, srv};
        use crate::watch::PAGE;
        let _guard = gpu_test_lock();
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        const LOOKUP: &str = "cbuffer c : register(b0) { uint lo; uint hi; uint at; };\n\
            ByteAddressBuffer table : register(t0, space2);\n\
            RWByteAddressBuffer dst : register(u0);\n\
            [numthreads(1, 1, 1)] void main() {\n\
                bool inside = hi < 256;\n\
                uint l1 = inside ? (hi << 4 | lo >> 28) : 0;\n\
                uint block = inside ? table.Load(l1 * 4) : 0;\n\
                uint entry = block << 12 | (lo >> 16 & 0xFFF);\n\
                dst.Store2(at, table.Load2((4096 + 2 * entry) * 4));\n\
            }\n";
        const READ: &str = "cbuffer c : register(b0) { uint offset; uint at; };\n\
            ByteAddressBuffer view : register(t0);\n\
            RWByteAddressBuffer dst : register(u0);\n\
            [numthreads(4, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n\
                dst.Store(at + id.x * 4, view.Load(offset + id.x * 4));\n\
            }\n";
        let lookup = pipeline(
            LOOKUP,
            windows::core::s!("page table entry"),
            0,
            &[
                Param::Constants {
                    register: 0,
                    count: 3,
                },
                Param::Srv {
                    register: 0,
                    space: 2,
                },
                Param::Uav { register: 0 },
            ],
        );
        let read = pipeline(
            READ,
            windows::core::s!("read a view"),
            0,
            &[
                Param::Constants {
                    register: 0,
                    count: 2,
                },
                Param::Table(&[srv(0, 1)]),
                Param::Uav { register: 0 },
            ],
        );
        let (Ok((lookup_root, lookup_pso)), Ok((read_root, read_pso))) = (lookup, read) else {
            return; // no compiler for the programs here
        };
        let (out, data) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // 600 MiB of title memory, a copy of all of it, and four words 520
        // MiB in.
        let bytes = 600 << 20;
        let memory = pool(bytes / PAGE);
        let at = memory + (520 << 20);
        for k in 0..4 {
            // SAFETY: our pool.
            unsafe { (at as *mut u32).add(k).write_volatile(0x5150 + k as u32) };
        }
        crate::shadow::register(memory, memory as u64, bytes as u64);
        let copy = crate::shadow::find(memory as u64, bytes as u64).expect("a copy");
        // The page table's entry for that address, from a submission that
        // binds it (which also brings the copy up to date).
        let list = host.new_list().unwrap();
        let map = bind(&list, OURS).expect("a page table");
        let constants = [at as u32, (at >> 32) as u32, 0];
        // SAFETY: an open list, live objects of ours.
        unsafe {
            list.list.SetComputeRootSignature(&lookup_root);
            list.list
                .SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootShaderResourceView(1, map.table);
            list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
            list.list.SetPipelineState(&lookup_pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mapped output, the GPU done with it.
        let [slot, base] = unsafe {
            [
                data.cast::<u32>().read_volatile(),
                data.cast::<u32>().add(1).read_volatile(),
            ]
        };
        assert_ne!(slot, 0, "the address has an entry");
        // Through the view that entry names, at the offset it gives.
        let list = host.new_list().unwrap();
        let constants = [(at as u32).wrapping_sub(base), 16];
        // SAFETY: an open list, live objects of ours.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
            list.list.SetComputeRootSignature(&read_root);
            list.list
                .SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootDescriptorTable(1, views.gpu(slot));
            list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
            list.list.SetPipelineState(&read_pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: as above.
        let words: Vec<u32> = (0..4)
            .map(|k| unsafe { data.cast::<u32>().add(4 + k).read_volatile() })
            .collect();
        assert_eq!(
            words,
            [0x5150, 0x5151, 0x5152, 0x5153],
            "the title's words 520 MiB into the copy"
        );
        crate::shadow::unregister(memory, memory as u64);
        drop(copy);
    }

    /// A copy unregistered between submissions: the next rewrite drops its
    /// name — a lagging table would name a freed pair, and a pair reused
    /// for another copy silently redirects shader reads, worse than a
    /// miss. The copy made after it (whose views may take the freed
    /// pair) and the one still live read their own bytes.
    #[test]
    fn an_unregistered_copy_leaves_the_next_table() {
        use crate::watch::PAGE;
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let Some((root, pso)) = pipeline(SOURCE) else {
            return; // no compiler for the shader here
        };
        let (out, at) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        let memory = [pool(1), pool(1), pool(1)];
        let register = |p: usize| {
            crate::shadow::register(p, p as u64, PAGE as u64);
            crate::shadow::find(p as u64, PAGE as u64).unwrap()
        };
        let (a, b) = (register(memory[0]), register(memory[1]));
        let list = host.new_list().unwrap();
        let read = |address: usize| -> Vec<u32> {
            let map = bind(&list, OURS).expect("a page table");
            let constants = [address as u32, (address >> 32) as u32, 0];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                list.list.SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { std::slice::from_raw_parts(at.cast::<u32>(), 4).to_vec() }
        };
        // SAFETY: our pools (a guarded page faults into the handler).
        let fill = |p: usize, base: u32| {
            for k in 0..4 {
                // SAFETY: p points into our pools, which guard their
                // pages; a faulting page goes to the guard's handler.
                unsafe { (p as *mut u32).add(k).write_volatile(base + k as u32) };
            }
        };
        // The table names both copies, each read through its own pair.
        fill(memory[0] + 16, 0x100);
        fill(memory[1] + 16, 0x200);
        assert_eq!(read(memory[0] + 16), [0x100, 0x101, 0x102, 0x103]);
        assert_eq!(read(memory[1] + 16), [0x200, 0x201, 0x202, 0x203]);
        assert!(pair_of(OURS, &a).is_some());
        // The title released the first copy between submissions and made
        // another, whose views may well take the freed pair's slots.
        crate::shadow::unregister(memory[0], memory[0] as u64);
        let c = register(memory[2]);
        fill(memory[2] + 16, 0x300);
        assert_eq!(read(memory[0] + 16), [0, 0, 0, 0], "memory the freed copy held reads as missing, not the new copy's bytes");
        assert!(pair_of(OURS, &a).is_none(), "the freed copy's pair is not named");
        assert_eq!(read(memory[1] + 16), [0x200, 0x201, 0x202, 0x203], "the live copy reads its own bytes");
        assert_eq!(read(memory[2] + 16), [0x300, 0x301, 0x302, 0x303], "the new copy reads its own bytes");
        drop(a);
        crate::shadow::unregister(memory[1], memory[1] as u64);
        crate::shadow::unregister(memory[2], memory[2] as u64);
        drop((b, c));
    }
}
