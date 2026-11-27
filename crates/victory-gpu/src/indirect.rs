//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! With a count buffer ExecuteIndirect runs only the first min(max, count)
//! records, so the clamp reads the count word and stops there: the maximum
//! can be far above the count, and the records between never run.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension except in x where [`widest`] allows more, 2^20
//! per dispatch, one-dimensional mesh dispatches, the budget) cut
//! legitimate work beyond them; raise them if the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, host, transition, uav_barrier};

/// Whether indirect argument clamps without an earlier conflicting GPU write
/// in the same host list are hoisted into the list's prelude as a single
/// compute batch (`VICTORY_BATCH_CLAMPS=0` disables and clamps inline).
pub(crate) fn batch_clamps_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

#[cfg(test)]
std::thread_local! {
    static TEST_BATCH_CLAMPS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_batch_clamps<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_BATCH_CLAMPS.with(|c| c.set(self.0));
        }
    }
    let previous = TEST_BATCH_CLAMPS.with(|c| c.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

pub(crate) fn batch_clamps_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_BATCH_CLAMPS.with(|c| c.get()) {
        return enabled;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        batch_clamps_enabled_value(std::env::var_os("VICTORY_BATCH_CLAMPS").as_deref())
    })
}

/// Whether indirect executions whose clamps their host list's prelude cannot
/// take (an earlier command of the list wrote their arguments, may have, or
/// an earlier clamp of the list is not in the prelude) are clamped in group
/// passes (`VICTORY_CLAMP_GROUPS=1`; off by default): one pass, recorded
/// where the first of them would have clamped inline, clamps every
/// execution recorded after it until a command writes what the next one
/// reads ([`GROUP_SOURCE`], `HostList::clamp_group_seat`). Inline, each
/// clamp is a compute dispatch between two draws: the host ends the render
/// pass for it and starts another after it, two Metal encoders on D3DMetal
/// for each of ~4,000 executions a world frame (a Metal System Trace of the
/// prison cell on an M5 Pro: 6,917 encoders a frame, compute and render in
/// turn). Needs the batched clamps (`VICTORY_BATCH_CLAMPS`, on by default).
///
/// After avatarmanz's batches (VIctory-7900XTX 0375962, GPL-3.0: a pass
/// reading a table of jobs the CPU fills while the list records, and the
/// clamped records left readable as arguments, cc11ccc), with the list's
/// tracker of written ranges deciding where a group ends and every
/// execution its own job, charged in recorded order: what a group pass
/// writes is word for word what the executions' inline clamps write.
pub(crate) fn groups_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_CLAMP_GROUPS.with(std::cell::Cell::get) {
        return enabled;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CLAMP_GROUPS"))
}

#[cfg(test)]
std::thread_local! {
    static TEST_CLAMP_GROUPS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a record that does not fit in what is left of its list's budget
/// takes the rest, emptying every record after it. This tree's inline clamp
/// (`indirect-clamp.hlsl`) always does, so group passes do too; main's
/// `VICTORY_INDIRECT_REST` switch (#403), which makes it opt-in, is not here.
fn indirect_rest() -> bool {
    true
}

#[cfg(test)]
pub(crate) fn with_clamp_groups<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_CLAMP_GROUPS.with(|c| c.set(self.0));
        }
    }
    let previous = TEST_CLAMP_GROUPS.with(|c| c.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

/// Whether an indirect execution's records `[arguments, arguments + len)`
/// or its count word (title addresses) are read by the GPU in place in title
/// memory, which commands the list's tracker does not see may write while
/// it runs: such an execution is clamped inline, right before it runs, not
/// hoisted into the prelude nor grouped (avatarmanz 631400f: hoisted, trees
/// flickered). The hook for in-place section views
/// (`VICTORY_INPLACE_SECTIONS`, `crate::alias::clamp_in_place_inline`);
/// none are read in place yet.
pub(crate) fn clamped_in_place(arguments: u64, len: u64, count: Option<u64>) -> bool {
    let _ = (arguments, len, count);
    false
}

/// What keeps an indirect execution's clamp out of its list's prelude:
/// memory read in place ([`clamped_in_place`]), else what `list` recorded
/// before ([`HostList::clamp_blocker`]).
pub(crate) fn blocker(
    list: &HostList,
    (arguments, len): (u64, u64),
    count: Option<u64>,
) -> Option<crate::host::Blocker> {
    if clamped_in_place(arguments, len, count) {
        return Some(crate::host::Blocker::InPlace);
    }
    list.clamp_blocker(arguments, len, count)
}

/// An indirect execution's clamp was kept out of its list's prelude by
/// `blocker` (it was grouped or clamped inline).
pub(crate) fn note_held_out(blocker: Option<crate::host::Blocker>) {
    use crate::host::Blocker;
    let at = match blocker {
        None => return,
        Some(Blocker::Inline) => 2,
        Some(Blocker::Unknown) => 3,
        Some(Blocker::Written) => 4,
        Some(Blocker::InPlace) => 5,
    };
    GROUPS[at].fetch_add(1, Ordering::Relaxed);
}

/// The clamps recorded a `ResourceBarrier` call of `barriers` barriers
/// (inline, in a prelude's batch, in a group pass, or moving their buffers
/// back to common): each a wait D3DMetal records and submits at a cost.
pub(crate) fn note_barriers(barriers: usize) {
    BARRIERS[0].fetch_add(1, Ordering::Relaxed);
    BARRIERS[1].fetch_add(barriers as u64, Ordering::Relaxed);
}

/// Since the last call: the clamps' `ResourceBarrier` calls and barriers.
pub(crate) fn take_barriers() -> [u64; 2] {
    BARRIERS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static BARRIERS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the last call: group passes in lists, executions clamped in them,
/// clamps kept out of their list's prelude after an earlier clamp of the
/// list that is not in it, after a command that may write any title buffer,
/// over bytes an earlier command of the list wrote, and read in place; and
/// group passes in preludes and the hoisted clamps in them.
pub(crate) fn take_groups() -> [u64; 8] {
    GROUPS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static GROUPS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// The jobs of one group pass: a group of more goes on in another. The
/// trace's long runs of clamped draws hold ~480 executions each (5 runs,
/// 2,409 of a frame's 3,258 clamp/draw pairs). [`GROUP_SOURCE`] caps its
/// count at this.
pub(crate) const GROUP_JOBS: u32 = 512;
/// The bytes of a group pass's table of jobs in upload memory: the count
/// (and 12 bytes unused), then per job four words ([`GROUP_SOURCE`]).
pub(crate) const GROUP_TABLE: u64 = 16 + 16 * GROUP_JOBS as u64;
/// The bytes of one clamped record in a group's output: the longest record
/// a group takes.
pub(crate) const GROUP_SLOT: u32 = 32;
/// The title buffers (host copies) one group pass reads records from.
pub(crate) const GROUP_SOURCES: usize = 8;

/// The group pass: the clamp of [`WORDS_SOURCE`] over the jobs of a table
/// instead of consecutive records, each with its own source buffer (one of
/// eight, `src0`..`src7`), record (a word index) and place in `dest`. Jobs
/// are charged in their order, a tile of 64 at a time, as records are. In
/// the words form, for Wine's HLSL compiler: no `Load` on read-write
/// buffers and no stores to indexed group-shared memory (scratch slots in
/// `budget` bytes `64..320`).
const GROUP_SOURCE: &str = r#"
cbuffer Pass : register(b0) { uint flags; uint widest; };
StructuredBuffer<uint> jobs : register(t0);
RWByteAddressBuffer dest : register(u0);
RWByteAddressBuffer budget : register(u1);
RWStructuredBuffer<uint> budgetWords : register(u2);
StructuredBuffer<uint> src0 : register(t1);
StructuredBuffer<uint> src1 : register(t2);
StructuredBuffer<uint> src2 : register(t3);
StructuredBuffer<uint> src3 : register(t4);
StructuredBuffer<uint> src4 : register(t5);
StructuredBuffer<uint> src5 : register(t6);
StructuredBuffer<uint> src6 : register(t7);
StructuredBuffer<uint> src7 : register(t8);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
static const uint PAST = 0xffffffffu;
uint word(uint s, uint at) {
    if (s == 0u) { return src0[at]; }
    if (s == 1u) { return src1[at]; }
    if (s == 2u) { return src2[at]; }
    if (s == 3u) { return src3[at]; }
    if (s == 4u) { return src4[at]; }
    if (s == 5u) { return src5[at]; }
    if (s == 6u) { return src6[at]; }
    return src7[at];
}
// The record of `kind` and `stride` bytes at word `from` of source `s`
// clamped to byte `to` of `dest`, as `WORDS_SOURCE` clamps one; its cost.
uint clamped(uint s, uint kind, uint stride, uint from, uint to) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        uint raw_count = word(s, from);
        uint raw_instances = word(s, from + 1u);
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(to, uint2(count, instances));
        if (kind == 1u) {
            dest.Store3(to + 8u, uint3(word(s, from + 2u), word(s, from + 3u), word(s, from + 4u)));
            base_bytes = 20u;
        } else {
            dest.Store2(to + 8u, uint2(word(s, from + 2u), word(s, from + 3u)));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = uint3(word(s, from), word(s, from + 1u), word(s, from + 2u));
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(to, g);
        base_bytes = 12u;
    } else {
        uint3 raw_g = uint3(word(s, from), word(s, from + 1u), word(s, from + 2u));
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(to, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(to + extra, word(s, from + extra / 4u));
    }
    return cost;
}
// A job: source (bits 0-3), kind (4-7) and stride in bytes (8-15); the
// record's first word in the source; its byte in `dest`; unused.
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint lane = id.x;
    uint left = 0u;
    if (lane == 0u) {
        left = BUDGET - min(budgetWords[0], BUDGET);
    }
    uint count = min(jobs[0], 512u);
    uint take_rest = flags & 2u;
    uint tiles = (count + 63u) / 64u;
    for (uint tile = 0u; tile < tiles; tile++) {
        uint j = tile * 64u + lane;
        uint head = 0u;
        uint to = 0u;
        uint cost = 0u;
        if (j < count) {
            head = jobs[4u + j * 4u];
            to = jobs[6u + j * 4u];
            uint from = jobs[5u + j * 4u];
            cost = clamped(head & 15u, (head >> 4) & 15u, (head >> 8) & 255u, from, to);
        }
        budget.Store(64u + lane * 4u, cost);
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            for (uint i = 0u; i < 64u; i++) {
                uint c = budgetWords[16u + i];
                if (c > left) {
                    budget.Store(64u + i * 4u, PAST);
                    if (take_rest != 0u) {
                        left = 0u;
                    }
                } else {
                    left -= c;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (j < count && budgetWords[16u + lane] == PAST) {
            if (((head >> 4) & 15u) <= 1u) {
                dest.Store2(to, uint2(0u, 0u));
            } else {
                dest.Store3(to, uint3(0u, 0u, 0u));
            }
        }
        if (tile + 1u < tiles) {
            AllMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) {
        budget.Store(0u, BUDGET - left);
    }
}
"#;

/// The group pass's root signature and pipeline ([`GROUP_SOURCE`]): root
/// constants (flags, widest), the table, `dest`, the budget as bytes and as
/// words, the eight sources.
struct GroupClamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for GroupClamp {}
// SAFETY: as above.
unsafe impl Sync for GroupClamp {}

fn group_clamp() -> Option<&'static GroupClamp> {
    static CLAMP: std::sync::OnceLock<Option<GroupClamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            use crate::compute::Param;
            let mut params = vec![
                Param::Constants { register: 0, count: 2 },
                Param::Srv { register: 0, space: 0 },
                Param::Uav { register: 0 },
                Param::Uav { register: 1 },
                Param::Uav { register: 2 },
            ];
            let sources =
                (1..=GROUP_SOURCES as u32).map(|register| Param::Srv { register, space: 0 });
            params.extend(sources);
            match crate::compute::pipeline(GROUP_SOURCE, s!("clamp groups"), 0, &params) {
                Ok((root, pso)) => Some(GroupClamp { root, pso }),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no group clamp: indirect arguments are clamped one execution at a time"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// The group pass's job for a record of `kind` and `stride` bytes at word
/// `from` of source `source`, clamped to byte `to` of the group's output.
pub(crate) fn group_job(source: u32, kind: u32, stride: u32, from: u32, to: u32) -> [u32; 4] {
    [source | kind << 4 | stride << 8, from, to, 0]
}

/// Whether an execution of `max` records of `kind`, `stride` bytes apart,
/// with a count or not, can be clamped in a group pass: one record of the
/// kinds the host runs, no longer than [`GROUP_SLOT`], no count (all the
/// title's indirect executions).
pub(crate) fn groupable(kind: u32, stride: u32, max: u32, counted: bool) -> bool {
    matches!(kind, 0 | 1 | 2 | 10)
        && max == 1
        && !counted
        && stride.is_multiple_of(4)
        && (12..=GROUP_SLOT).contains(&stride)
}

/// Records, in `l`, a group pass over the table at host address `table`
/// into `out` (in state `before`, left readable as arguments), charging the
/// budget at `budget`, reading the eight `sources` (host addresses of title
/// buffers' copies, the first repeated where there are fewer). Replaces the
/// list's compute root signature and pipeline. False if the pass is not
/// available.
pub(crate) fn record_group(
    l: &ID3D12GraphicsCommandList,
    (out, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    table: u64,
    sources: &[u64; GROUP_SOURCES],
) -> bool {
    if group_clamp().is_none() {
        return false;
    }
    // SAFETY: an open list; our buffer.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        let mut barriers = vec![uav_barrier()];
        if before != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            barriers.push(transition(out, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        l.ResourceBarrier(&barriers);
        note_barriers(barriers.len());
        dispatch_group(l, out.GetGPUVirtualAddress(), budget, table, sources);
        GROUPS[0].fetch_add(1, Ordering::Relaxed);
        l.ResourceBarrier(&[transition(
            out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
        note_barriers(1);
    }
    true
}

/// The group pass over the table at host address `table` into the buffer at
/// `out` (unordered access), with no barrier: [`record_group`]'s, or a
/// prelude's batch ([`record_batch`]).
fn dispatch_group(
    l: &ID3D12GraphicsCommandList,
    out: u64,
    budget: u64,
    table: u64,
    sources: &[u64; GROUP_SOURCES],
) {
    let Some(c) = group_clamp() else {
        return;
    };
    let constants = [u32::from(indirect_rest()) << 1, widest()];
    // SAFETY: an open list; our root signature, pipeline and buffers; title
    // buffers' copies the list keeps, read in the common state (promoted).
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, table);
        l.SetComputeRootUnorderedAccessView(2, out);
        l.SetComputeRootUnorderedAccessView(3, budget);
        l.SetComputeRootUnorderedAccessView(4, budget);
        for (i, &source) in sources.iter().enumerate() {
            l.SetComputeRootShaderResourceView(5 + i as u32, source);
        }
        l.Dispatch(1, 1, 1);
    }
}

/// What [`prelude_run`] finds: clamps taken, output buffer, sources, jobs.
type PreludeRun = (usize, u64, [u64; GROUP_SOURCES], Vec<[u32; 4]>);

/// A run of a prelude's clamps ([`record_batch`]) in one group pass: how
/// many of `clamps` it takes (those of no record are skipped, as alone),
/// the buffer of `outs` (host address, bytes) their records go to, the
/// sources it binds and its jobs. `None` if the first cannot be grouped
/// ([`PendingClamp::src_base`]).
fn prelude_run(clamps: &[PendingClamp], outs: &[(u64, u64)]) -> Option<PreludeRun> {
    let first = clamps.first().filter(|c| c.src_base != 0)?;
    let (out, out_len) =
        *outs.iter().find(|&&(at, len)| first.dst_gpu >= at && first.dst_gpu - at < len)?;
    let mut sources = [first.src_base; GROUP_SOURCES];
    let mut bound = 1;
    let mut jobs = Vec::new();
    let mut taken = 0;
    for c in clamps {
        if c.records == 0 {
            taken += 1;
            continue;
        }
        if c.src_base == 0 || jobs.len() == GROUP_JOBS as usize {
            break;
        }
        let to = c.dst_gpu - out;
        if c.dst_gpu < out || to + u64::from(c.stride) > out_len || to > u64::from(u32::MAX) {
            break;
        }
        let source = match sources[..bound].iter().position(|&s| s == c.src_base) {
            Some(s) => s,
            None if bound < GROUP_SOURCES => {
                sources[bound] = c.src_base;
                bound += 1;
                bound - 1
            }
            None => break,
        };
        let from = ((c.src_gpu - c.src_base) / 4) as u32;
        jobs.push(group_job(source as u32, c.kind, c.stride, from, to as u32));
        taken += 1;
    }
    (!jobs.is_empty()).then_some((taken, out, sources, jobs))
}

/// An execution of one record was clamped in a group pass.
pub(crate) fn note_grouped() {
    GROUPS[1].fetch_add(1, Ordering::Relaxed);
    RECORDS[0].fetch_add(1, Ordering::Relaxed);
}

/// Whether the group pass's pipeline is available on this host.
pub(crate) fn group_available() -> bool {
    group_clamp().is_some()
}

/// The host copies the last grouped executions read their records from,
/// newest first: what a group pass binds besides its first execution's
/// ([`recent_sources`]).
static RECENT: std::sync::Mutex<Vec<std::sync::Weak<crate::shadow::Shadow>>> =
    std::sync::Mutex::new(Vec::new());

/// `source` was read by a grouped execution: first of [`RECENT`].
pub(crate) fn note_source(source: &std::sync::Arc<crate::shadow::Shadow>) {
    let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
    let this = std::sync::Arc::as_ptr(source);
    if recent.first().is_some_and(|r| std::ptr::eq(r.as_ptr(), this)) {
        return;
    }
    recent.retain(|r| r.strong_count() > 0 && !std::ptr::eq(r.as_ptr(), this));
    recent.insert(0, std::sync::Arc::downgrade(source));
    recent.truncate(2 * GROUP_SOURCES);
}

/// The buffers of the host copies of [`RECENT`] that live, newest first.
pub(crate) fn recent_sources() -> Vec<ID3D12Resource> {
    let recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
    recent.iter().filter_map(std::sync::Weak::upgrade).map(|s| s.buffer.clone()).collect()
}

/// Whether the argument clamp compute pipeline is available on this host.
pub(crate) fn available() -> bool {
    clamp().is_some()
}

/// One indirect execution's clamp hoisted into a host list's prelude.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingClamp {
    pub(crate) src_gpu: u64,
    pub(crate) dst_gpu: u64,
    pub(crate) count_gpu: Option<u64>,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
    /// The host address of the buffer `src_gpu` is in, if a group pass can
    /// clamp it ([`groupable`], the record a word of it a 32-bit index
    /// reaches); 0 otherwise.
    pub(crate) src_base: u64,
}

const SOURCE: &str = include_str!("indirect-clamp.hlsl");

/// The most groups in x the clamp lets a dispatch run (2^20 groups in all
/// still bound it). The title's tile passes are dispatches of one group per
/// 8x4 tile of the picture, 67,440 and 47,760 for its two tile classes in a
/// frame at 2560x1440 (measured on Windows in VIctory-7900XTX, 84eae35);
/// cut to 65,535, the last tiles, the lowest of the picture, never run.
/// The console runs what it is asked, and so does D3DMetal: an M5 Pro ran
/// every group of indirect dispatches up to 2^20 wide. D3D12's own limit,
/// 65,535, stays for other hosts. `VICTORY_INDIRECT_WIDE=0` or `1`
/// overrides.
pub(crate) fn widest() -> u32 {
    static WIDEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WIDEST.get_or_init(|| {
        let wide = match std::env::var("VICTORY_INDIRECT_WIDE").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => crate::shader_cache::under_wine(),
        };
        if wide { 1 << 20 } else { 65_535 }
    })
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Built from [`WORDS_SOURCE`], which also reads the budget as words
    /// through root view 5 (the other views are those of [`SOURCE`]).
    words: bool,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    #[cfg(victory_dxc_clamp)]
    match precompiled() {
        Ok(clamp) => {
            tracing::info!("argument clamp: using precompiled 64-thread shader");
            return Ok(clamp);
        }
        Err(e) => tracing::warn!(error = %e, "precompiled argument clamp unavailable"),
    }
    make_fxc().or_else(|e| {
        tracing::info!(error = %e, "argument clamp: building the one-thread form");
        make_words()
    })
}

/// The clamp compiled by DXC at build time. D3DMetal finds a shader by the
/// `HASH` part of its container, and DXC leaves that and the container
/// digest as zeros when it cannot find `dxil.dll` to sign with, which would
/// make this shader share a cache entry with every other unsigned one (issue
/// #4). So both are checked here and set when they are zero.
#[cfg(victory_dxc_clamp)]
fn precompiled() -> windows::core::Result<Clamp> {
    let mut bytes = include_bytes!(concat!(env!("OUT_DIR"), "/indirect-clamp.dxil")).to_vec();
    match victory_shader::emit::sign_container(&mut bytes) {
        Ok(true) => tracing::info!("argument clamp: the compiled shader was unsigned, its digests are set"),
        Ok(false) => {}
        Err(e) => tracing::warn!(error = %e, "argument clamp: could not check the compiled shader's digests"),
    }
    from_bytes(&bytes)
}

/// The clamp for Wine's HLSL compiler (vkd3d-shader), which has no `Load`
/// on `RWByteAddressBuffer`, no stores to indexed group-shared memory and
/// no `[RootSignature]`: [`SOURCE`]'s clamp, read through structured views
/// of the same buffers, and one thread charging the records in order
/// instead of a group of 64. The results are the same.
const WORDS_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
uint3 load3(uint at) { return uint3(argWords[at / 4], argWords[at / 4 + 1], argWords[at / 4 + 2]); }
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        uint count = min(argWords[at / 4], 1u << 24);
        uint instances = min(argWords[at / 4 + 1], (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1) {
            dest.Store3(at + 8, uint3(argWords[at / 4 + 2], argWords[at / 4 + 3], argWords[at / 4 + 4]));
        } else {
            dest.Store2(at + 8, uint2(argWords[at / 4 + 2], argWords[at / 4 + 3]));
        }
    } else if (kind == 2) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        uint3 g = load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint walked = records;
    if (counted != 0) {
        walked = min(records, countWords[0]);
    }
    for (uint record = 0; record < walked; record++) {
        uint at = record * stride;
        uint cost = clamped(at);
        if (cost > left) {
            // The first record that does not fit takes what is left, so
            // those after it run empty too.
            dest.Store(at, 0);
            left = 0;
        } else {
            left -= cost;
        }
    }
    budget.Store(0, BUDGET - left);
}
"#;

fn make_words() -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 5 },
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Uav { register: 0 },
        Param::Srv { register: 1, space: 0 },
        Param::Uav { register: 2 },
    ];
    let (root, pso) = crate::compute::pipeline(WORDS_SOURCE, s!("clamp"), 0, &params)?;
    Ok(Clamp { root, pso, words: true })
}

fn make_fxc() -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the compiler owns the blob until this call returns.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    from_bytes(bytes)
}

fn from_bytes(bytes: &[u8]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    // SAFETY: D3D12 validates the bytecode and its embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso, words: false })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// `count` is the host address of the title's record count (a title
/// buffer's copy, only read): the clamp then stops at the lesser of it and
/// `records`, which is where ExecuteIndirect stops. A count that is not
/// 4-byte aligned is not read: every record is clamped.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    bound_with(clamp()?, list, gpu, (kind, stride, records), count)
}

/// [`bound`] with a given clamp.
fn bound_with(
    c: &Clamp,
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    let budget = list.budget()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    CLAMPS[1].fetch_add(1, Ordering::Relaxed);
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(count.is_some()), widest()];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        note_barriers(2);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
        note_barriers(2);
    }
    list.note_work(true);
    Some(out)
}

/// Queues an indirect argument clamp to run in `list`'s prelude before
/// `list.list` starts, allocating a disjoint slice from `list`'s hoisted
/// argument arena. Returns the arena buffer and byte offset ready for
/// `ExecuteIndirect` (the prelude transitions the buffer into
/// `INDIRECT_ARGUMENT`, and `list.close()` transitions it back to `COMMON`).
#[cfg(test)]
pub(crate) fn enqueue(
    list: &HostList,
    gpu: u64,
    spec: (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    enqueue_from(list, (None, gpu), spec, count)
}

/// [`enqueue`] of a record in the buffer `source` if it is known: with clamp
/// groups on, the prelude may clamp it in a group pass ([`record_batch`]).
pub(crate) fn enqueue_from(
    list: &HostList,
    (source, gpu): (Option<&ID3D12Resource>, u64),
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    let _ = clamp()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count_gpu = count.filter(|at| at.is_multiple_of(4));
    let bytes = u64::from(stride) * u64::from(records.max(1));
    let src_base = source
        .filter(|_| groups_enabled() && groupable(kind, stride, records, count_gpu.is_some()))
        .map(|buffer| {
            // SAFETY: plain queries on a live buffer.
            let (at, len) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc().Width) };
            let inside = gpu >= at
                && gpu + bytes <= at.saturating_add(len)
                && (gpu - at) / 4 <= u64::from(u32::MAX);
            if inside && at.is_multiple_of(4) { at } else { 0 }
        })
        .unwrap_or(0);
    let out = list.enqueue_hoisted_clamp(bytes, |dst_gpu| PendingClamp {
        src_gpu: gpu,
        dst_gpu,
        count_gpu,
        kind,
        stride,
        records,
        src_base,
    })?;
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count_gpu.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    CLAMPS[0].fetch_add(1, Ordering::Relaxed);
    Some(out)
}

/// Records all hoisted argument clamps for a host list into its prelude
/// command list as one compute batch: one pre-batch barrier, one root
/// signature + PSO + budget bind, one dispatch per non-empty clamp (with a
/// UAV barrier between consecutive dispatches so the shared list budget is
/// charged in exact recorded order), and one post-batch transition into
/// `INDIRECT_ARGUMENT`.
///
/// With clamp groups on ([`groups_enabled`]), runs of clamps a group pass
/// can take are one dispatch each instead, over a table of jobs in memory
/// from `table` (bytes: host address, CPU pointer, the list's upload
/// memory): the barriers between dispatches, one per clamp, become one per
/// pass.
pub(crate) fn record_batch(
    cmd: &ID3D12GraphicsCommandList,
    budget: u64,
    buffers: &[ID3D12Resource],
    clamps: &[PendingClamp],
    table: &mut dyn FnMut(u64) -> Option<(u64, *mut u8)>,
) {
    let Some(c) = clamp() else {
        return;
    };
    if buffers.is_empty() || clamps.is_empty() {
        return;
    }
    let mut pre = Vec::with_capacity(buffers.len() + 1);
    pre.push(uav_barrier());
    for buf in buffers {
        pre.push(transition(
            buf,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        ));
    }
    let mut post = Vec::with_capacity(buffers.len() + 1);
    post.push(uav_barrier());
    for buf in buffers {
        post.push(transition(
            buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ));
    }
    let widest = widest();
    let grouped = groups_enabled() && group_available();
    // The buffers' host addresses and bytes, where group passes write.
    let outs: Vec<(u64, u64)> = if grouped {
        buffers
            .iter()
            // SAFETY: plain queries on live buffers.
            .map(|b| unsafe { (b.GetGPUVirtualAddress(), b.GetDesc().Width) })
            .collect()
    } else {
        Vec::new()
    };
    // SAFETY: `cmd` is the host list's open prelude command list; our clamp
    // pipeline, budget buffer, hoisted argument buffers, and title shadow
    // copies are kept alive by the host list until reset.
    unsafe {
        cmd.ResourceBarrier(&pre);
        note_barriers(pre.len());
        // Whether the one-execution clamp's root signature and pipeline are
        // bound (a group pass binds its own).
        let mut single = false;
        let mut dispatched = false;
        let mut at = 0;
        while at < clamps.len() {
            let clamp = &clamps[at];
            if clamp.records == 0 {
                at += 1;
                continue;
            }
            if dispatched {
                // Serialize budget read-modify-write between consecutive
                // clamps in recorded order while staying in one compute pass.
                cmd.ResourceBarrier(&[uav_barrier()]);
                note_barriers(1);
            }
            dispatched = true;
            let run = grouped.then(|| prelude_run(&clamps[at..], &outs)).flatten();
            if let Some((taken, out, sources, jobs)) = run
                && let Some((table_gpu, words)) = table(GROUP_TABLE)
            {
                let words = words.cast::<u32>();
                words.write_volatile(jobs.len() as u32);
                for (i, job) in jobs.iter().enumerate() {
                    for (k, &word) in job.iter().enumerate() {
                        words.add(4 + 4 * i + k).write_volatile(word);
                    }
                }
                dispatch_group(cmd, out, budget, table_gpu, &sources);
                GROUPS[6].fetch_add(1, Ordering::Relaxed);
                GROUPS[7].fetch_add(jobs.len() as u64, Ordering::Relaxed);
                single = false;
                at += taken;
                continue;
            }
            if !single {
                cmd.SetComputeRootSignature(&c.root);
                cmd.SetPipelineState(&c.pso);
                cmd.SetComputeRootUnorderedAccessView(2, budget);
                if c.words {
                    cmd.SetComputeRootUnorderedAccessView(5, budget);
                }
                single = true;
            }
            let constants = [
                clamp.kind,
                clamp.stride,
                clamp.records,
                u32::from(clamp.count_gpu.is_some()),
                widest,
            ];
            cmd.SetComputeRoot32BitConstants(
                0,
                constants.len() as u32,
                constants.as_ptr().cast(),
                0,
            );
            cmd.SetComputeRootShaderResourceView(1, clamp.src_gpu);
            cmd.SetComputeRootUnorderedAccessView(3, clamp.dst_gpu);
            cmd.SetComputeRootShaderResourceView(4, clamp.count_gpu.unwrap_or(clamp.src_gpu));
            cmd.Dispatch(1, 1, 1);
            at += 1;
        }
        cmd.ResourceBarrier(&post);
        note_barriers(post.len());
    }
}

/// Records the clamps were asked to walk (the maximum of each execution)
/// since the last call, and the part of them under a count buffer, where
/// the walk stops at the count.
pub(crate) fn take_records() -> [u64; 2] {
    RECORDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static RECORDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Clamp invocations since the last call: `[hoisted, inline]`.
pub(crate) fn take_clamps() -> [u64; 2] {
    CLAMPS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static CLAMPS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
    note_barriers(1);
}

#[cfg(test)]
mod tests {
    use crate::backend::{HostGpu, HostMemory};

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Every word the clamp `c` writes for `records` records of `kind` made
    /// from `seed` by a small generator, in one list with no count buffer.
    /// `None` without a host GPU.
    #[cfg(victory_dxc_clamp)]
    fn clamp_all(c: &super::Clamp, kind: u32, seed: u32, records: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let all = per * records as usize;
        let (buffer, data) = host.shared_buffer(all as u64 * 4 + 16).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        let mut state = seed;
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Mostly small values, now and then a wild one.
                let word = if state >> 28 == 0 { state } else { (state >> 8) % 9000 };
                data.add(at).write_volatile(word);
            }
            let spec = (kind, per as u32 * 4, records);
            let out = super::bound_with(c, &list, buffer.GetGPUVirtualAddress(), spec, None).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut out_data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut out_data)).unwrap();
            Some((0..all).map(|i| out_data.cast::<u32>().add(i).read_volatile()).collect())
        }
    }

    /// The precompiled clamp is the one in use when it builds, and it writes
    /// what the clamp compiled at run time writes for the same records of
    /// every kind.
    #[cfg(victory_dxc_clamp)]
    #[test]
    #[ignore = "requires a hardware D3D12 queue and the build-time DXC shader"]
    fn hardware_uses_precompiled_clamp() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let precompiled = super::precompiled().expect("create pipeline from the compiled clamp");
        assert!(!precompiled.words);
        assert!(!super::clamp().expect("argument clamp").words);
        let reference = super::make_fxc().or_else(|_| super::make_words()).expect("the existing clamp");
        for (kind, seed, records) in [(0, 1, 200), (1, 2, 200), (2, 3, 130), (10, 4, 70), (0, 5, 3)] {
            let new = clamp_all(&precompiled, kind, seed, records).unwrap();
            let old = clamp_all(&reference, kind, seed, records).unwrap();
            assert_eq!(new, old, "kind {kind}, {records} records");
        }
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_under(kind, words, records, calls, None)
    }

    /// As [`clamped`], with the title's record count `count` (a word in a
    /// buffer of its own) if there is one.
    fn clamped_under(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        count: Option<u32>,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let (counter, word) = host.shared_buffer(4).unwrap();
            word.cast::<u32>().write_volatile(count.unwrap_or(0));
            let counted = count.map(|_| counter.GetGPUVirtualAddress());
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, counted).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [70_000.min(super::widest()), 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// The walk stops at the count: of 128 garbage draws under a count of 3
    /// the first two run and the third runs empty, as they do without one,
    /// and the others are never read (the argument buffer's fresh words stay
    /// zero). A count above the records changes nothing; a count of zero
    /// walks none.
    #[test]
    fn the_walk_stops_at_the_count() {
        let Some(under) = clamped_under(0, &[u32::MAX], 128, 1, Some(3)) else {
            return; // no host GPU here
        };
        assert_eq!(under[..3], [1 << 24, 1 << 24, 0]);
        assert!(under[3..].iter().all(|&count| count == 0));
        let all = clamped(0, &[u32::MAX], 128, 1).unwrap();
        assert_eq!(under[..3], all[..3]);
        let above = clamped_under(0, &[u32::MAX], 128, 1, Some(1 << 20)).unwrap();
        assert_eq!(above, all);
        let none = clamped_under(0, &[u32::MAX], 128, 1, Some(0)).unwrap();
        assert!(none.iter().all(|&count| count == 0));
    }

    /// Only the records that run are charged to the budget: after a call
    /// under a count of 1 a second finds room for that record again, where
    /// the same draws with no count spent it all on records that never ran.
    #[test]
    fn records_past_the_count_are_not_charged() {
        let Some(twice) = clamped_under(0, &[u32::MAX], 128, 2, Some(1)) else {
            return; // no host GPU here
        };
        assert_eq!(twice[0], 1 << 24);
        let spent = clamped(0, &[u32::MAX], 128, 2).unwrap();
        assert!(spent.iter().all(|&count| count == 0));
    }

    /// The groups of one indirect dispatch of `groups` x 1 x 1 that run on
    /// the host, through the clamp when `clamped`, else straight from the
    /// record. Each group marks its own word. `None` without a host GPU.
    fn groups_that_run(groups: u32, clamped: bool) -> Option<usize> {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store(g.x * 4, 1);
}
"#;
        let host = host()?;
        let (root, pso) =
            crate::compute::pipeline(MARK, s!("marks"), 0, &[Param::Uav { register: 0 }]).unwrap();
        let (record, words) = host.shared_buffer(12).unwrap();
        let (marks, seen) = host.shared_buffer(u64::from(groups) * 4).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; our pipeline, signature and buffers.
        unsafe {
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
                .unwrap();
            let signature = signature.unwrap();
            words.cast::<[u32; 3]>().write_volatile([groups, 1, 1]);
            std::ptr::write_bytes(seen, 0, groups as usize * 4);
            let arguments = if clamped {
                bound(&list, record.GetGPUVirtualAddress(), (2, 12, 1), None).unwrap()
            } else {
                list.list.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                record.clone()
            };
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
            l.ExecuteIndirect(&signature, 1, &arguments, 0, None, 0);
            if clamped {
                done(&list, &arguments);
            } else {
                l.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let seen = seen.cast::<u32>();
            Some(
                (0..groups as usize)
                    .filter(|&i| seen.add(i).read_volatile() != 0)
                    .count(),
            )
        }
    }

    /// A tile pass of the title (67,440 groups, one class of its 8x4 tiles at
    /// 2560x1440) runs every group where the host runs dispatches that wide,
    /// and its first 65,535 elsewhere. The host runs the widest dispatch the
    /// clamp lets through, so a host that stops running them fails here.
    #[test]
    fn a_wide_tile_pass_runs_every_group_the_host_runs() {
        let Some(ran) = groups_that_run(67_440, true) else {
            return; // no host GPU here
        };
        let widest = super::widest();
        assert_eq!(ran, 67_440.min(widest) as usize);
        assert_eq!(
            groups_that_run(widest, false).unwrap(),
            widest as usize,
            "the host runs the widest dispatch"
        );
        assert_eq!(
            groups_that_run(widest, true).unwrap(),
            widest as usize,
            "the clamp lets it through"
        );
    }

    /// Dispatches wider than [`super::widest`] are cut to 65,535 groups in x
    /// as before, and one whose groups pass 2^20 in all runs empty, also
    /// where x times y would wrap 32 bits.
    #[test]
    fn dispatches_past_the_bounds_are_cut() {
        let widest = super::widest();
        let Some(past) = clamped(2, &[widest + 1, 1, 1], 1, 1) else {
            return; // no host GPU here
        };
        assert_eq!(past, [65_535]);
        assert_eq!(clamped(2, &[widest, 1, 1], 1, 1).unwrap(), [widest]);
        // Past 2^20 groups in all: 65,535 by 17, 2^10 by 2^10 by 2, and
        // 2^20 by 4,096, whose x * y wraps 32 bits to zero.
        assert_eq!(clamped(2, &[65_535, 17, 1], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 10, 1 << 10, 2], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 20, 4_096, 1], 1, 1).unwrap(), [0]);
    }

    #[test]
    fn batch_clamps_env_toggle_defaults_on() {
        use std::ffi::OsStr;
        assert!(super::batch_clamps_enabled_value(None));
        assert!(super::batch_clamps_enabled_value(Some(OsStr::new("1"))));
        assert!(super::batch_clamps_enabled_value(Some(OsStr::new("yes"))));
        assert!(!super::batch_clamps_enabled_value(Some(OsStr::new("0"))));
        super::with_batch_clamps(false, || {
            assert!(!super::batch_clamps_enabled());
            super::with_batch_clamps(true, || {
                assert!(super::batch_clamps_enabled());
            });
            assert!(!super::batch_clamps_enabled());
        });
    }

    /// Clamp groups are a switch (`VICTORY_CLAMP_GROUPS=1`), off unless set;
    /// tests turn them on and off per thread.
    #[test]
    fn clamp_groups_switch_toggles_per_thread() {
        super::with_clamp_groups(true, || {
            assert!(super::groups_enabled());
            super::with_clamp_groups(false, || assert!(!super::groups_enabled()));
            assert!(super::groups_enabled());
        });
        if std::env::var_os("VICTORY_CLAMP_GROUPS").is_none() {
            assert!(!super::groups_enabled(), "off by default");
        }
    }

    #[test]
    fn write_hazard_tracks_overlaps_coalescing_and_order() {
        let mut h = crate::host::WriteHazard::default();
        assert!(h.can_hoist(0x1000, 32, None));
        h.note_write(0x1000, 0); // zero-length writes nothing
        assert!(h.can_hoist(0x1000, 32, None));

        // Disjoint writes and boundary checks.
        h.note_write(0x2000, 16);
        h.note_write(0x1000, 16);
        assert!(h.can_hoist(0x1010, 0x0FF0, None), "adjacent ranges do not overlap");
        assert!(!h.can_hoist(0x100C, 16, None), "partial head overlap");
        assert!(!h.can_hoist(0x0FF8, 16, None), "partial tail overlap");
        assert!(!h.can_hoist(0x3000, 16, Some(0x200C)), "count word overlap");
        assert!(h.can_hoist(0x3000, 16, Some(0x2010)), "count word right after write");

        // Coalescing bridging write merges [0x1000, 0x1010) and [0x2000, 0x2010).
        h.note_write(0x1010, 0x0FF0);
        assert!(!h.can_hoist(0x1800, 4, None));

        // Inline clamp blocks later hoisting to preserve sequential budget order.
        let mut h2 = crate::host::WriteHazard::default();
        assert!(h2.can_hoist(0x4000, 16, None));
        h2.note_inline();
        assert!(!h2.can_hoist(0x4000, 16, None));
        h2.clear();
        assert!(h2.can_hoist(0x4000, 16, None));

        // Unknown GPU write (bindless heap UAV / memory writes) blocks everything until clear.
        h2.note_unknown();
        assert!(!h2.can_hoist(0x9000, 16, None));
        h2.clear();
        assert!(h2.can_hoist(0x9000, 16, None));
    }

    /// A batch of hoisted clamps in one prelude produces bit-for-bit identical
    /// clamped words and sequential budget accounting as inline clamps across
    /// kinds (draw, indexed, dispatch, mesh), count buffers, and budget
    /// exhaustion.
    #[test]
    fn hoisted_batch_matches_inline_across_kinds_counts_and_budget() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        struct Case {
            kind: u32,
            stride: u32,
            records: u32,
            words: Vec<u32>,
            count: Option<u32>,
        }
        let cases = vec![
            Case {
                kind: 0,
                stride: 16,
                records: 3,
                words: vec![
                    12, 2, 0, 0,
                    1 << 25, 10, 1, 2, // vertex count clamped to 2^24, instances clamped to 4
                    5, 1, 3, 4,
                ],
                count: None,
            },
            Case {
                kind: 1,
                stride: 20,
                records: 4,
                words: vec![
                    100, 3, 7, 8, 9,
                    u32::MAX, u32::MAX, 1, 2, 3, // past-count: untouched
                    u32::MAX, u32::MAX, 4, 5, 6,
                    u32::MAX, u32::MAX, 7, 8, 9,
                ],
                count: Some(1),
            },
            Case {
                kind: 2,
                stride: 12,
                records: 2,
                words: vec![
                    super::widest() + 5, 1, 1, // cut to 65535
                    128, 2, 1,
                ],
                count: None,
            },
            Case {
                kind: 10,
                stride: 12,
                records: 2,
                words: vec![
                    100, 5, 3, // y,z clamped to 1
                    70_000, 1, 1, // x clamped to 65535
                ],
                count: None,
            },
            // Exhausts the remaining list budget across the two records.
            Case {
                kind: 0,
                stride: 16,
                records: 4,
                words: vec![
                    1 << 24, 4, 0, 0,
                    1 << 24, 4, 0, 0,
                    3, 1, 0, 0,
                    3, 1, 0, 0,
                ],
                count: None,
            },
        ];

        let mut inputs = Vec::new();
        for case in &cases {
            let (src, src_ptr) = host.shared_buffer((case.words.len() * 4) as u64).unwrap();
            let (cnt, cnt_ptr) = host.shared_buffer(4).unwrap();
            // SAFETY: our mapped buffers, written before submission.
            unsafe {
                for (i, &w) in case.words.iter().enumerate() {
                    src_ptr.cast::<u32>().add(i).write_volatile(w);
                }
                cnt_ptr.cast::<u32>().write_volatile(case.count.unwrap_or(0));
            }
            inputs.push((src, cnt));
        }

        // 1. Run hoisted in one prelude.
        let _ = super::take_clamps();
        let hoisted_list = host.new_list().unwrap();
        let mut hoisted_outs = Vec::new();
        for (case, (src, cnt)) in cases.iter().zip(&inputs) {
            // SAFETY: live buffers.
            let (src_gpu, cnt_gpu) = unsafe {
                (
                    src.GetGPUVirtualAddress(),
                    case.count.map(|_| cnt.GetGPUVirtualAddress()),
                )
            };
            let out = super::enqueue(
                &hoisted_list,
                src_gpu,
                (case.kind, case.stride, case.records),
                cnt_gpu,
            )
            .expect("hoisted clamp enqueued");
            hoisted_outs.push(out);
        }
        hoisted_list.close();
        host.execute(&[&hoisted_list]);
        let [hoisted_n, inline_n] = super::take_clamps();
        // The counters are global, so a test running at the same time may
        // add to them: the words compared below are the exact check.
        assert!(hoisted_n >= cases.len() as u64, "{hoisted_n} {inline_n}");

        let mut hoisted_words: Vec<Vec<u32>> = Vec::new();
        for (case, (buf, offset)) in cases.iter().zip(&hoisted_outs) {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: mapped shared buffer, list has run.
            unsafe {
                buf.Map(0, None, Some(&mut ptr)).unwrap();
                let base = ptr.cast::<u8>().add(*offset as usize).cast::<u32>();
                let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                    * (case.stride as usize / 4);
                hoisted_words.push((0..n).map(|i| base.add(i).read_volatile()).collect());
                buf.Unmap(0, None);
            }
        }

        // 2. Run inline in one list, copying each clamped slice out before the
        //    next inline clamp overwrites offset 0.
        let inline_list = host.new_list().unwrap();
        let mut readbacks = Vec::new();
        for (case, (src, cnt)) in cases.iter().zip(&inputs) {
            // SAFETY: live buffers and open list.
            unsafe {
                let src_gpu = src.GetGPUVirtualAddress();
                let cnt_gpu = case.count.map(|_| cnt.GetGPUVirtualAddress());
                let out = super::bound(
                    &inline_list,
                    src_gpu,
                    (case.kind, case.stride, case.records),
                    cnt_gpu,
                )
                .expect("inline clamp");
                let bytes = u64::from(case.stride) * u64::from(case.records);
                let (copy, copy_ptr) = host.shared_buffer(bytes).unwrap();
                inline_list.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                inline_list.list.CopyBufferRegion(&copy, 0, &out, 0, bytes);
                inline_list.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                super::done(&inline_list, &out);
                readbacks.push((copy, copy_ptr));
            }
        }
        inline_list.close();
        host.execute(&[&inline_list]);
        let [hoisted_n, inline_n] = super::take_clamps();
        assert!(inline_n >= cases.len() as u64, "{hoisted_n} {inline_n}");

        for (i, (case, (_buf, ptr))) in cases.iter().zip(&readbacks).enumerate() {
            let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                * (case.stride as usize / 4);
            // SAFETY: mapped shared buffer, list has run.
            let inline: Vec<u32> =
                unsafe { (0..n).map(|k| ptr.cast::<u32>().add(k).read_volatile()).collect() };
            assert_eq!(hoisted_words[i], inline, "case {i} kind {}", case.kind);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// Hoisted argument buffers transition cleanly COMMON -> UAV ->
    /// INDIRECT_ARGUMENT in prelude and INDIRECT_ARGUMENT -> COMMON at close,
    /// in any submission order, across re-submission of a closed list, and
    /// across list reset and reuse.
    #[test]
    fn hoisted_argument_buffers_transition_resubmit_and_reset_cleanly() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(64).unwrap();
        // SAFETY: mapped buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([123, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;

        let list = host.new_list().unwrap();
        let _ = issued();
        let (out1, off1) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        let (out2, off2) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        assert_eq!(key(&out1), key(&out2), "small clamps pack into one chunk");
        assert_ne!(off1, off2, "hoisted clamps get disjoint offsets");
        list.close();
        let close_transitions = issued();

        // Execute once: prelude transitions happen during execute().
        host.execute(&[&list]);
        let prelude_transitions = issued();
        let mut run = prelude_transitions;
        run.extend_from_slice(&close_transitions);
        let mut states: std::collections::HashMap<usize, u32> =
            [(key(&title), 0), (key(&out1), 0)].into_iter().collect();
        replay(&run, &mut states).unwrap();
        assert!(run.iter().all(|t| t.0 != key(&title)), "title copy never transitioned");
        assert_eq!(states[&key(&out1)], 0, "hoisted buffer ends COMMON");

        // Update source and re-submit the SAME closed list without reset.
        // SAFETY: mapped buffer.
        unsafe {
            data.cast::<[u32; 3]>().write_volatile([456, 2, 1]);
        }
        host.execute(&[&list]);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: mapped buffer, list has run.
        unsafe {
            out1.Map(0, None, Some(&mut ptr)).unwrap();
            let got1 = ptr.cast::<u8>().add(off1 as usize).cast::<[u32; 3]>().read_volatile();
            let got2 = ptr.cast::<u8>().add(off2 as usize).cast::<[u32; 3]>().read_volatile();
            out1.Unmap(0, None);
            assert_eq!(got1, [456, 2, 1]);
            assert_eq!(got2, [456, 2, 1]);
        }

        // Reset the list, reuse the chunk for new data, and execute again.
        list.reset();
        // SAFETY: mapped buffer.
        unsafe {
            data.cast::<[u32; 3]>().write_volatile([789, 1, 1]);
        }
        let (reused, off_reused) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        assert_eq!(key(&reused), key(&out1), "chunk is reused after reset");
        assert_eq!(off_reused, 0, "bump cursor resets to 0");
        list.close();
        host.execute(&[&list]);
        // SAFETY: mapped buffer, list has run.
        unsafe {
            reused.Map(0, None, Some(&mut ptr)).unwrap();
            let got = ptr.cast::<[u32; 3]>().read_volatile();
            reused.Unmap(0, None);
            assert_eq!(got, [789, 1, 1]);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// Multiple hoisted indirect dispatches execute in one list, while a
    /// same-list GPU write into an argument buffer disables hoisting for that
    /// buffer (falling back to inline clamping) without blocking disjoint
    /// buffers, and a mixed hoisted-prefix + inline-suffix list shares its
    /// budget sequentially.
    #[test]
    fn hoisted_indirect_execution_and_same_list_write_hazard() {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
cbuffer Offset : register(b0) { uint base; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store((base + g.x) * 4, 1);
}
"#;
        const WRITE_ARGS: &str = r#"
RWByteAddressBuffer outArgs : register(u0);
cbuffer Val : register(b0) { uint gx; uint gy; uint gz; };
[numthreads(1, 1, 1)]
void main() {
    outArgs.Store3(0, uint3(gx, gy, gz));
}
"#;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let (mark_root, mark_pso) = crate::compute::pipeline(
            MARK,
            s!("mark_batch"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        let (write_root, write_pso) = crate::compute::pipeline(
            WRITE_ARGS,
            s!("write_args"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 3 }],
        )
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        // SAFETY: plain command signature creation.
        let signature = unsafe {
            let mut sig: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };

        // Part 1: 4 hoisted indirect dispatches in one list, no PSO re-bind between them.
        let (args_buf, args_ptr) = host.shared_buffer(64).unwrap();
        let (marks_buf, marks_ptr) = host.shared_buffer(64 * 4).unwrap();
        let group_counts = [3u32, 5, 2, 4];
        let bases = [0u32, 8, 16, 24];
        let list = host.new_list().unwrap();
        // SAFETY: mapped buffers, open list, live PSO/root/signature.
        unsafe {
            std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
            for (i, &g) in group_counts.iter().enumerate() {
                args_ptr.cast::<u32>().add(i * 4).write_volatile(g);
                args_ptr.cast::<u32>().add(i * 4 + 1).write_volatile(1);
                args_ptr.cast::<u32>().add(i * 4 + 2).write_volatile(1);
            }
            let args_gpu = args_buf.GetGPUVirtualAddress();
            let l = &list.list;
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            for (i, &base) in bases.iter().enumerate() {
                let (clamped, off) =
                    enqueue(&list, args_gpu + (i as u64) * 16, (2, 12, 1), None).unwrap();
                l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                l.ExecuteIndirect(&signature, 1, &clamped, off, None, 0);
            }
            list.close();
            host.execute(&[&list]);
            for (&g, &base) in group_counts.iter().zip(&bases) {
                for k in 0..8u32 {
                    let expected = u32::from(k < g);
                    assert_eq!(
                        marks_ptr.cast::<u32>().add((base + k) as usize).read_volatile(),
                        expected,
                        "hoisted dispatch base={base} k={k}"
                    );
                }
            }
        }

        // Part 2: Same-list GPU write into shadow buffer A blocks hoisting for A
        // (and runs inline with the freshly written args), while disjoint B is
        // still hoistable before the inline clamp runs.
        let backing = vec![0u8; 4096];
        let addr_a = backing.as_ptr() as u64;
        let addr_b = addr_a + 1024;
        crate::shadow::register(addr_a as usize, addr_a, 4096);
        let shadow_a = crate::shadow::find(addr_a, 12).expect("registered shadow");
        let list2 = host.new_list().unwrap();
        // SAFETY: mapped buffers and open list.
        unsafe {
            std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
            // Pre-populate shadow A with stale [1, 1, 1] and shadow B with [4, 1, 1].
            let mut ptr = std::ptr::null_mut();
            shadow_a.buffer.Map(0, None, Some(&mut ptr)).unwrap();
            ptr.cast::<[u32; 3]>().write_volatile([1, 1, 1]);
            ptr.cast::<u8>().add(1024).cast::<[u32; 3]>().write_volatile([4, 1, 1]);
            shadow_a.buffer.Unmap(0, None);

            // GPU dispatch writes [7, 1, 1] into shadow A.
            shadow_a.gpu_writes(&list2, addr_a, 12, false);
            let vals = [7u32, 1, 1];
            let l = &list2.list;
            l.SetComputeRootSignature(&write_root);
            l.SetPipelineState(&write_pso);
            l.SetComputeRootUnorderedAccessView(0, shadow_a.gpu(addr_a));
            l.SetComputeRoot32BitConstants(1, 3, vals.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            // Disjoint buffer B can still hoist into prelude; buffer A cannot!
            assert!(list2.can_hoist_clamp(addr_b, 12, None));
            assert!(!list2.can_hoist_clamp(addr_a, 12, None));

            // Hoist B first (prefix), then run A inline (suffix).
            let (clamped_b, off_b) =
                enqueue(&list2, shadow_a.gpu(addr_b), (2, 12, 1), None).unwrap();
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            let base_b = 0u32;
            l.SetComputeRoot32BitConstants(1, 1, (&raw const base_b).cast(), 0);
            l.ExecuteIndirect(&signature, 1, &clamped_b, off_b, None, 0);

            let clamped_a = bound(&list2, shadow_a.gpu(addr_a), (2, 12, 1), None).unwrap();
            list2.note_inline_clamp();
            // After an inline clamp, even disjoint B must stay inline for budget order.
            assert!(!list2.can_hoist_clamp(addr_b, 12, None));
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            let base_a = 16u32;
            l.SetComputeRoot32BitConstants(1, 1, (&raw const base_a).cast(), 0);
            l.ExecuteIndirect(&signature, 1, &clamped_a, 0, None, 0);
            done(&list2, &clamped_a);

            list2.close();
            host.execute(&[&list2]);
            for k in 0..8u32 {
                assert_eq!(
                    marks_ptr.cast::<u32>().add(k as usize).read_volatile(),
                    u32::from(k < 4),
                    "hoisted disjoint B k={k}"
                );
                assert_eq!(
                    marks_ptr.cast::<u32>().add((16 + k) as usize).read_volatile(),
                    u32::from(k < 7),
                    "inline hazard A saw GPU-written [7,1,1] k={k}"
                );
            }
        }
        crate::shadow::unregister(addr_a as usize, addr_a);
        assert_eq!(host.log_messages(), None);
    }

    /// What keeps a clamp out of the prelude is named; group passes take
    /// executions of one record of 12 to 32 bytes with no count.
    #[test]
    fn hazards_name_what_keeps_a_clamp_out_of_the_prelude() {
        use crate::host::{Blocker, WriteHazard};
        let mut h = WriteHazard::default();
        assert_eq!(h.blocker(0x1000, 16, None), None);
        h.note_write(0x1000, 16);
        assert_eq!(h.blocker(0x1000, 16, None), Some(Blocker::Written));
        assert_eq!(h.blocker(0x2000, 16, Some(0x100c)), Some(Blocker::Written), "the count word");
        assert_eq!(h.blocker(0x2000, 16, Some(0x1010)), None);
        h.note_unknown();
        assert_eq!(h.blocker(0x9000, 4, None), Some(Blocker::Unknown));
        h.note_inline();
        assert_eq!(h.blocker(0x9000, 4, None), Some(Blocker::Inline));
        h.clear();
        assert_eq!(h.blocker(0x1000, 16, None), None);

        use super::groupable;
        assert!(groupable(10, 12, 1, false) && groupable(0, 16, 1, false) && groupable(1, 20, 1, false));
        assert!(groupable(2, 12, 1, false) && groupable(0, 32, 1, false));
        assert!(!groupable(0, 16, 2, false), "one record");
        assert!(!groupable(0, 16, 0, false), "one record");
        assert!(!groupable(0, 16, 1, true), "no count");
        assert!(!groupable(0, 36, 1, false), "no longer than a slot");
        assert!(!groupable(0, 18, 1, false), "whole words");
        assert!(!groupable(3, 12, 1, false), "the kinds the host runs");
        let cap = format!("min(jobs[0], {}u)", super::GROUP_JOBS);
        assert!(super::GROUP_SOURCE.contains(&cap), "the pass walks at most a table's jobs");
    }

    /// The executions the world frame clamps inline come after the compute
    /// passes that write their arguments. In one list: a dispatch writes
    /// every record over stale words, an inline clamp spends half the
    /// budget, then executions of every kind, some from other buffers than
    /// the first (bound by the pass from the recent ones, or past the eight
    /// it binds: another pass), one clamped inline among them (its count:
    /// the group after it is another), and records the budget no longer
    /// holds. Their group passes write word for word what inline clamps
    /// after the same producer write, reading what it wrote.
    #[test]
    fn group_passes_clamp_as_inline_clamps_do() {
        use super::*;
        use crate::compute::Param;
        const PRODUCE: &str = r#"
StructuredBuffer<uint> src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Count : register(b0) { uint n; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    for (uint i = id.x; i < n; i += 64u) {
        dst.Store(i * 4u, src[i]);
    }
}
"#;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let (root, pso) = crate::compute::pipeline(
            PRODUCE,
            s!("produce"),
            0,
            &[Param::Srv { register: 0, space: 0 }, Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        // (buffer, kind, stride, words of the record, a count word)
        type Case = (usize, u32, u32, Vec<u32>, Option<u32>);
        let cases: Vec<Case> = vec![
            (0, 0, 16, vec![12, 2, 7, 9], None),
            (0, 1, 20, vec![100, 3, 7, 8, 9], None),
            (1, 2, 12, vec![widest() + 5, 1, 1], None),
            (0, 10, 12, vec![100, 5, 3], None),
            (1, 10, 32, vec![70_000, 1, 1, 4, 5, 6, 7, 8], None),
            (0, 0, 32, vec![1 << 25, 10, 1, 2, 11, 12, 13, 14], None),
            (2, 2, 12, vec![65_535, 17, 1], None),
            // Clamped inline (a count): the next execution opens a pass.
            (0, 0, 16, vec![5, 1, 3, 4], Some(1)),
            (0, 0, 16, vec![6, 2, 3, 4], None),
            // Nine more buffers: past the eight one pass binds.
            (3, 10, 12, vec![9, 1, 1], None),
            (4, 10, 12, vec![10, 1, 1], None),
            (5, 10, 12, vec![11, 1, 1], None),
            (6, 10, 12, vec![12, 1, 1], None),
            (7, 10, 12, vec![13, 1, 1], None),
            (8, 10, 12, vec![14, 1, 1], None),
            (9, 10, 12, vec![15, 1, 1], None),
            (10, 10, 12, vec![16, 1, 1], None),
            (11, 10, 12, vec![17, 1, 1], None),
            // Half the budget each: after the half spent and the rest, the
            // first no longer fits; then small ones still do.
            (0, 0, 16, vec![1 << 22, 16, 0, 0], None),
            (1, 0, 16, vec![1 << 22, 16, 0, 0], None),
            (0, 0, 16, vec![3, 1, 0, 0], None),
            (1, 1, 20, vec![3, 1, 0, 0, 0], None),
        ];
        const STALE: u32 = 0x5a5a_5a5a;
        const BUFFERS: usize = 12;
        let at_of = |i: usize| -> u64 { 64 * i as u64 + 16 };
        // Per buffer: the title's (stale until produced, at 16 bytes in) and
        // what the producer writes there.
        let buffers: Vec<_> = (0..BUFFERS)
            .map(|b| {
                let (title, title_ptr) = host.shared_buffer(64 * cases.len() as u64 + 64).unwrap();
                let (produced, produced_ptr) = host.shared_buffer(64 * cases.len() as u64 + 64).unwrap();
                for (i, (buffer, _, _, words, _)) in cases.iter().enumerate() {
                    if *buffer == b {
                        // SAFETY: our mapped buffers, written before any list runs.
                        unsafe {
                            for (w, &word) in words.iter().enumerate() {
                                produced_ptr.cast::<u8>().add(at_of(i) as usize).cast::<u32>().add(w).write_volatile(word);
                            }
                        }
                    }
                }
                (title, title_ptr, produced)
            })
            .collect();
        let (counter, counter_ptr) = host.shared_buffer(4).unwrap();
        let (half, half_ptr) = host.shared_buffer(16).unwrap();
        // SAFETY: as above.
        unsafe {
            counter_ptr.cast::<u32>().write_volatile(1);
            half_ptr.cast::<[u32; 4]>().write_volatile([1 << 22, 16, 0, 0]);
        }
        // SAFETY: live buffers.
        let gpu = |b: &ID3D12Resource| unsafe { b.GetGPUVirtualAddress() };
        let words = (64 * cases.len() + 64) / 4;
        let stale = || {
            for (_, ptr, _) in &buffers {
                // SAFETY: our mapped buffers; no list is running.
                unsafe { (0..words).for_each(|i| ptr.cast::<u32>().add(i).write_volatile(STALE)) };
            }
        };
        // The producer, then an inline clamp spending half the budget.
        let produce = |list: &HostList| {
            let l = &list.list;
            let n = words as u32;
            // SAFETY: an open list; our pipeline and live buffers.
            unsafe {
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                for (title, _, produced) in &buffers {
                    l.SetComputeRootShaderResourceView(0, gpu(produced));
                    l.SetComputeRootUnorderedAccessView(1, gpu(title));
                    l.SetComputeRoot32BitConstants(2, 1, (&raw const n).cast(), 0);
                    l.Dispatch(1, 1, 1);
                }
                l.ResourceBarrier(&[uav_barrier()]);
            }
            list.note_unknown_gpu_write();
            let spent = bound(list, gpu(&half), (0, 16, 1), None).unwrap();
            list.note_inline_clamp();
            done(list, &spent);
        };
        let read = |buffer: &ID3D12Resource, offset: u64, n: usize| -> Vec<u32> {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours; the lists have run.
            unsafe {
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let base = p.cast::<u8>().add(offset as usize).cast::<u32>();
                let out = (0..n).map(|i| base.add(i).read_volatile()).collect();
                buffer.Unmap(0, None);
                out
            }
        };
        // Copies an inline clamp's record out of the list's argument buffer
        // before the next clamp writes there.
        let copy_out = |list: &HostList, out: &ID3D12Resource, bytes: u64| {
            let (copy, _) = host.shared_buffer(bytes).unwrap();
            // SAFETY: an open list; live buffers.
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    out,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.list.CopyBufferRegion(&copy, 0, out, 0, bytes);
                list.list.ResourceBarrier(&[transition(
                    out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
            }
            done(list, out);
            copy
        };

        // Grouped: the producer and the executions in one list A.
        stale();
        let a = host.new_list().unwrap();
        produce(&a);
        // The buffers other executions read lately, newest first.
        let recent: Vec<ID3D12Resource> =
            [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 3].iter().map(|&b| buffers[b].0.clone()).collect();
        let mut opened = Vec::new();
        let places: Vec<(ID3D12Resource, u64)> = cases
            .iter()
            .enumerate()
            .map(|(i, (b, kind, stride, _, count))| {
                let (title, ..) = &buffers[*b];
                let record = gpu(title) + at_of(i);
                if count.is_some() {
                    let out = bound(&a, record, (*kind, *stride, 1), Some(gpu(&counter))).unwrap();
                    a.note_inline_clamp();
                    opened.push(false);
                    return (copy_out(&a, &out, u64::from(*stride)), 0);
                }
                assert!(groupable(*kind, *stride, 1, false));
                let (seat, open) = a
                    .clamp_group_seat((title, record), (record, u64::from(*stride)), || recent.clone())
                    .expect("a seat");
                opened.push(open);
                a.clamp_group_join(&seat, record, (*kind, *stride)).expect("a place")
            })
            .collect();
        a.close();
        host.execute(&[&a]);
        let grouped: Vec<Vec<u32>> = cases
            .iter()
            .zip(&places)
            .map(|((_, _, stride, ..), (buffer, offset))| read(buffer, *offset, *stride as usize / 4))
            .collect();
        // A pass opens at the first execution (binding buffers 0, 1, 2 and 4
        // to 8), after the inline one, and at the first execution of each
        // buffer the pass before does not bind.
        let passes: Vec<usize> = opened.iter().enumerate().filter(|&(_, &o)| o).map(|(i, _)| i).collect();
        assert_eq!(passes, [0, 8, 9, 15, 16, 17, 18]);

        // Inline: the producer and every clamp in one list C.
        stale();
        let c = host.new_list().unwrap();
        produce(&c);
        let copies: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(i, (b, kind, stride, _, count))| {
                let (title, ..) = &buffers[*b];
                let counted = count.map(|_| gpu(&counter));
                let out = bound(&c, gpu(title) + at_of(i), (*kind, *stride, 1), counted).unwrap();
                c.note_inline_clamp();
                copy_out(&c, &out, u64::from(*stride))
            })
            .collect();
        c.close();
        host.execute(&[&c]);
        for (i, (case, copy)) in cases.iter().zip(&copies).enumerate() {
            let inline = read(copy, 0, case.2 as usize / 4);
            assert_eq!(grouped[i], inline, "execution {i} kind {}", case.1);
            assert!(!inline.contains(&STALE), "execution {i} read the producer's words");
        }
        let n = cases.len();
        assert_eq!(grouped[n - 4][0], 0, "the budget left no room for the first half");
        assert_eq!(grouped[n - 2][..2], [3, 1], "small ones fit");
        assert_eq!(host.log_messages(), None);
    }

    /// With clamp groups on, a prelude's clamps go in group passes where
    /// they can (one record, no count; up to eight buffers a pass) and one
    /// at a time where not (several records, a count, none), in their
    /// order: word for word what inline clamps write, across kinds and
    /// strides, ten buffers, and records the budget no longer holds.
    #[test]
    fn prelude_group_passes_clamp_as_inline_clamps_do() {
        use super::*;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        // (buffer, kind, stride, records, words, a count word)
        type Case = (usize, u32, u32, u32, Vec<u32>, Option<u32>);
        let mut cases: Vec<Case> = vec![
            (0, 0, 16, 1, vec![12, 2, 7, 9], None),
            (1, 1, 20, 1, vec![100, 3, 7, 8, 9], None),
            (2, 2, 12, 1, vec![widest() + 5, 1, 1], None),
            (0, 10, 32, 1, vec![70_000, 1, 1, 4, 5, 6, 7, 8], None),
            (1, 0, 16, 3, [5, 1, 0, 0].repeat(3), None),
            (2, 10, 12, 1, vec![100, 5, 3], None),
            (0, 0, 16, 2, [6, 2, 0, 0].repeat(2), Some(1)),
            (1, 2, 12, 0, vec![9, 1, 1], None),
            (0, 0, 32, 1, vec![1 << 25, 10, 1, 2, 11, 12, 13, 14], None),
        ];
        cases.extend((3..10).map(|b| (b, 10, 12, 1, vec![b as u32, 1, 1], None)));
        cases.extend([
            (0, 0, 16, 1, vec![1 << 22, 16, 0, 0], None),
            (4, 0, 16, 1, vec![1 << 22, 16, 0, 0], None),
            (5, 1, 20, 1, vec![1 << 22, 16, 0, 0, 0], None),
            (6, 0, 16, 1, vec![3, 1, 0, 0], None),
        ]);
        let at_of = |i: usize| -> u64 { 128 * i as u64 + 32 };
        let bytes = 128 * cases.len() as u64 + 128;
        let buffers: Vec<_> = (0..10)
            .map(|b| {
                let (buffer, ptr) = host.shared_buffer(bytes).unwrap();
                for (i, (from, _, _, _, words, _)) in cases.iter().enumerate() {
                    if *from == b {
                        // SAFETY: our mapped buffer, written before any list runs.
                        unsafe {
                            let to = ptr.cast::<u8>().add(at_of(i) as usize).cast::<u32>();
                            words.iter().enumerate().for_each(|(w, &word)| to.add(w).write_volatile(word));
                        }
                    }
                }
                buffer
            })
            .collect();
        let (counter, counter_ptr) = host.shared_buffer(4).unwrap();
        // SAFETY: as above.
        unsafe { counter_ptr.cast::<u32>().write_volatile(1) };
        // SAFETY: live buffers.
        let gpu = |b: &ID3D12Resource| unsafe { b.GetGPUVirtualAddress() };
        let walked = |(_, _, stride, records, _, count): &Case| {
            count.map_or(*records, |c| c.min(*records)) as usize * (*stride as usize / 4)
        };
        let read = |buffer: &ID3D12Resource, offset: u64, n: usize| -> Vec<u32> {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours; the list has run.
            unsafe {
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let base = p.cast::<u8>().add(offset as usize).cast::<u32>();
                let out = (0..n).map(|i| base.add(i).read_volatile()).collect();
                buffer.Unmap(0, None);
                out
            }
        };

        let p = host.new_list().unwrap();
        let places: Vec<_> = super::with_clamp_groups(true, || {
            cases
                .iter()
                .enumerate()
                .map(|(i, (b, kind, stride, records, _, count))| {
                    let record = (Some(&buffers[*b]), gpu(&buffers[*b]) + at_of(i));
                    let counted = count.map(|_| gpu(&counter));
                    enqueue_from(&p, record, (*kind, *stride, *records), counted).unwrap()
                })
                .collect()
        });
        p.close();
        super::with_clamp_groups(true, || host.execute(&[&p]));
        let hoisted: Vec<Vec<u32>> =
            cases.iter().zip(&places).map(|(case, (b, at))| read(b, *at, walked(case))).collect();

        let c = host.new_list().unwrap();
        let copies: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(i, (b, kind, stride, records, _, count))| {
                let counted = count.map(|_| gpu(&counter));
                let at = gpu(&buffers[*b]) + at_of(i);
                let out = bound(&c, at, (*kind, *stride, *records), counted).unwrap();
                c.note_inline_clamp();
                let bytes = u64::from(*stride) * u64::from((*records).max(1));
                let (copy, _) = host.shared_buffer(bytes).unwrap();
                // SAFETY: an open list; live buffers.
                unsafe {
                    c.list.ResourceBarrier(&[transition(
                        &out,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    c.list.CopyBufferRegion(&copy, 0, &out, 0, bytes);
                    c.list.ResourceBarrier(&[transition(
                        &out,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    )]);
                }
                done(&c, &out);
                copy
            })
            .collect();
        c.close();
        host.execute(&[&c]);
        for (i, (case, copy)) in cases.iter().zip(&copies).enumerate() {
            assert_eq!(hoisted[i], read(copy, 0, walked(case)), "clamp {i} kind {}", case.1);
        }
        let n = cases.len();
        assert_eq!((hoisted[n - 4][0], hoisted[n - 3][0], hoisted[n - 2][0]), (1 << 22, 0, 0));
        assert_eq!(hoisted[n - 1][..2], [3, 1]);
        assert_eq!(host.log_messages(), None);
    }

    /// A producer, executions, the producer again over the same records,
    /// executions again, in one list: the executions after the second
    /// write cannot join the pass before it and get one of their own after
    /// it, so every execution runs the groups written last before it (those
    /// after an unrelated write still join). Twice through the same list,
    /// reset between: its passes' outputs start common again.
    #[test]
    fn a_write_ends_the_group_of_what_it_writes() {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
cbuffer Offset : register(b0) { uint base; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store((base + g.x) * 4, 1);
}
"#;
        const PRODUCE: &str = r#"
RWByteAddressBuffer records : register(u0);
cbuffer First : register(b0) { uint first; };
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    records.Store3(id.x * 12u, uint3(first + id.x, 1u, 1u));
}
"#;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let (mark_root, mark_pso) = crate::compute::pipeline(
            MARK,
            s!("group_marks"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        let (produce_root, produce_pso) = crate::compute::pipeline(
            PRODUCE,
            s!("group_produce"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: plain command signature creation.
        unsafe {
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
                .unwrap()
        };
        let signature = signature.unwrap();
        let (marks, marks_ptr) = host.shared_buffer(256 * 4).unwrap();
        // The title's buffer of records (a registered host copy, stale: one
        // group each) and one elsewhere.
        let backing = vec![0u8; 4096];
        let at = backing.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 4096);
        let records = crate::shadow::find(at, 48).expect("registered shadow");
        let list = host.new_list().unwrap();
        for round in 0..2 {
            // SAFETY: our mapped buffers, written before the list runs.
            unsafe {
                std::ptr::write_bytes(marks_ptr, 0, 256 * 4);
                let mut p = std::ptr::null_mut();
                records.buffer.Map(0, None, Some(&mut p)).unwrap();
                let stale = p.cast::<u8>().add(records.offset(at) as usize).cast::<[u32; 3]>();
                (0..4).for_each(|i| stale.add(i).write_volatile([1, 1, 1]));
                records.buffer.Unmap(0, None);
            }
            let produce = |first: u32| {
                records.gpu_writes(&list, at, 48, false);
                // SAFETY: an open list; our pipeline and live buffers.
                unsafe {
                    let l = &list.list;
                    l.SetComputeRootSignature(&produce_root);
                    l.SetPipelineState(&produce_pso);
                    l.SetComputeRootUnorderedAccessView(0, records.gpu(at));
                    l.SetComputeRoot32BitConstants(1, 1, (&raw const first).cast(), 0);
                    l.Dispatch(1, 1, 1);
                    l.ResourceBarrier(&[uav_barrier()]);
                }
            };
            let mut opened = Vec::new();
            let mut execute = |i: u64, base: u32| {
                let record = at + 12 * (i % 4);
                assert!(list.clamp_blocker(record, 12, None).is_some(), "out of the prelude");
                let (seat, open) = list
                    .clamp_group_seat((&records.buffer, records.gpu(record)), (record, 12), Vec::new)
                    .expect("a seat");
                opened.push(open);
                let (clamped, offset) = list.clamp_group_join(&seat, records.gpu(record), (2, 12)).unwrap();
                // SAFETY: an open list; our pipeline, signature and live buffers.
                unsafe {
                    let l = &list.list;
                    l.SetComputeRootSignature(&mark_root);
                    l.SetPipelineState(&mark_pso);
                    l.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
                    l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                    l.ExecuteIndirect(&signature, 1, &clamped, offset, None, 0);
                }
            };
            produce(3);
            (0..4).for_each(|i| execute(i, 16 * i as u32));
            // A write elsewhere: the group goes on.
            list.defer_gpu_write(&records, at as usize + 1024, 16, false);
            execute(1, 64);
            produce(7);
            (4..8).for_each(|i| execute(i, 16 * i as u32 + 64));
            assert_eq!(opened, [true, false, false, false, false, true, false, false, false], "round {round}");
            list.close();
            host.execute(&[&list]);
            let ran = |base: usize| {
                // SAFETY: mapped; the list has run.
                (0..16).filter(|&k| unsafe { marks_ptr.cast::<u32>().add(base + k).read_volatile() } != 0).count()
            };
            let first: Vec<usize> = (0..4).map(|i| ran(16 * i)).collect();
            assert_eq!(first, [3, 4, 5, 6], "round {round}: the first write's groups");
            assert_eq!(ran(64), 4, "round {round}: after the write elsewhere");
            let second: Vec<usize> = (4..8).map(|i| ran(16 * i + 64)).collect();
            assert_eq!(second, [7, 8, 9, 10], "round {round}: the second write's groups");
            list.reset();
        }
        // SAFETY: a live buffer.
        let past = unsafe { marks.GetGPUVirtualAddress() } + (1 << 30);
        assert!(
            list.clamp_group_seat((&marks, past), (past, 12), Vec::new).is_none(),
            "a record outside its buffer is clamped inline"
        );
        crate::shadow::unregister(at as usize, at);
        drop(backing);
        assert_eq!(host.log_messages(), None);
    }

    /// A list that mixes a hoisted prefix (in prelude) and an inline suffix
    /// (in the main list) shares the same list budget sequentially across the
    /// prelude -> main list boundary.
    #[test]
    fn hoisted_prefix_and_inline_suffix_share_budget_in_order() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // Each record asks for 2^24 * 4 = 2^26 vertex instances (half the 2^27
        // list budget). Two hoisted records exhaust the budget completely, so
        // an inline record after them in the same list must be zeroed.
        let (src, ptr) = host.shared_buffer(32).unwrap();
        let (inline_src, inline_ptr) = host.shared_buffer(16).unwrap();
        // SAFETY: mapped buffers.
        let (src_gpu, inline_gpu) = unsafe {
            let p = ptr.cast::<u32>();
            for i in 0..2 {
                p.add(i * 4).write_volatile(1 << 24);
                p.add(i * 4 + 1).write_volatile(4);
                p.add(i * 4 + 2).write_volatile(0);
                p.add(i * 4 + 3).write_volatile(0);
            }
            let ip = inline_ptr.cast::<u32>();
            ip.add(0).write_volatile(3);
            ip.add(1).write_volatile(1);
            ip.add(2).write_volatile(0);
            ip.add(3).write_volatile(0);
            (src.GetGPUVirtualAddress(), inline_src.GetGPUVirtualAddress())
        };
        let list = host.new_list().unwrap();
        let (hoisted_buf, hoisted_off) = super::enqueue(&list, src_gpu, (0, 16, 2), None).unwrap();
        let inline_buf = super::bound(&list, inline_gpu, (0, 16, 1), None).unwrap();
        list.note_inline_clamp();
        super::done(&list, &inline_buf);
        list.close();
        host.execute(&[&list]);

        let mut p = std::ptr::null_mut();
        // SAFETY: mapped shared buffers, list has run.
        unsafe {
            hoisted_buf.Map(0, None, Some(&mut p)).unwrap();
            let base = p.cast::<u8>().add(hoisted_off as usize).cast::<u32>();
            assert_eq!(base.add(0).read_volatile(), 1 << 24, "hoisted record 0 fits");
            assert_eq!(base.add(4).read_volatile(), 1 << 24, "hoisted record 1 fits");
            hoisted_buf.Unmap(0, None);

            inline_buf.Map(0, None, Some(&mut p)).unwrap();
            assert_eq!(
                p.cast::<u32>().read_volatile(),
                0,
                "inline suffix sees budget spent by hoisted prelude"
            );
            inline_buf.Unmap(0, None);
        }
        assert_eq!(host.log_messages(), None);
    }
}
