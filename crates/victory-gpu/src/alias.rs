//! Zero-copy title buffers on Apple Silicon through D3DMetal (`mach_vm_remap`
//! aliasing, issue #204).
//!
//! D3DMetal refuses `OpenExistingHeapFromAddress*`, so VIctory normally keeps
//! host copies of title buffers (`crate::shadow`) and copies pages back and
//! forth. On Apple Silicon (unified memory), D3DMetal's own mapped write-back
//! L0 buffers are backed by the same physical RAM. Under Wine + Rosetta 2, raw
//! `mach_msg2_trap` MIG calls (`mach_vm_remap` with `copy = FALSE`,
//! `VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) point the title's `VirtualAlloc`
//! virtual address range directly at the D3DMetal buffer's physical pages.
//!
//! On by default on Rosetta + D3DMetal (never WARP); `VICTORY_ALIAS_BUFFERS=0`
//! keeps the host copies. The probe at start-up ([`supported`]) still falls
//! back to the copies where the remap does not work.

use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};

use crate::backend::HostMemory;
use crate::host::host;
use crate::watch::PAGE;

use windows::Win32::Graphics::Direct3D12::*;

#[cfg(target_arch = "x86_64")]
mod mach {
    use core::arch::asm;
    use std::sync::{Mutex, OnceLock};

    const MACH: u64 = 0x0100_0000;
    const TRAP_REPLY_PORT: u64 = 26;
    const TRAP_TASK_SELF: u64 = 28;
    /// `mach_msg2_trap` (8 arguments). `mach_msg_trap` (31) is fatal on
    /// macOS 27+ for kernel-object calls (`EXC_GUARD`).
    const TRAP_MSG2: u64 = 47;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;

    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    unsafe fn trap(n: u64, a: [u64; 6]) -> u64 {
        // SAFETY: forwarded to `trap8`.
        unsafe { trap8(n, a, 0, 0) }
    }

    unsafe fn trap8(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
        let ret: u64;
        // SAFETY: x86-64 XNU Mach trap ABI under Rosetta 2; 7th and 8th args
        // sit at `[rsp + 8]` and `[rsp + 16]`.
        unsafe {
            asm!(
                "sub rsp, 32",
                "mov qword ptr [rsp], 0",
                "mov qword ptr [rsp + 8], {a7}",
                "mov qword ptr [rsp + 16], {a8}",
                "syscall",
                "add rsp, 32",
                inlateout("rax") MACH | n => ret,
                in("rdi") a[0],
                in("rsi") a[1],
                in("rdx") a[2],
                in("r10") a[3],
                in("r8") a[4],
                in("r9") a[5],
                a7 = in(reg) a7,
                a8 = in(reg) a8,
                lateout("rcx") _,
                lateout("r11") _,
            );
        }
        ret
    }

    pub fn task_self() -> u32 {
        // SAFETY: argument-less Mach trap.
        unsafe { trap(TRAP_TASK_SELF, [0; 6]) as u32 }
    }

    fn reply_port() -> u32 {
        static PORT: OnceLock<u32> = OnceLock::new();
        *PORT.get_or_init(|| {
            // SAFETY: argument-less Mach trap.
            unsafe { trap(TRAP_REPLY_PORT, [0; 6]) as u32 }
        })
    }

    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct Header {
        bits: u32,
        size: u32,
        remote: u32,
        local: u32,
        voucher: u32,
        id: i32,
    }

    /// `mach_vm_remap` MIG request (`msgh_id = 4813`, `#pragma pack(4)`).
    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct RemapRequest {
        head: Header,
        descriptors: u32,
        src_task: u32,
        pad1: u32,
        disp_type: u32,
        ndr: [u8; 8],
        target_address: u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_address: u64,
        copy: u32,
        inheritance: u32,
    }

    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    struct RemapReply {
        head: Header,
        ndr: [u8; 8],
        ret: i32,
        target_address: u64,
        cur_protection: i32,
        max_protection: i32,
        trailer: [u8; 128],
    }

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    static REMAP_LOCK: Mutex<()> = Mutex::new(());

    /// Maps `[src, src + size)` of this task at `target`, sharing physical
    /// pages (`copy = FALSE`).
    ///
    /// # Safety
    /// Replaces whatever is mapped at `target`; both ranges must be page
    /// aligned and owned by the caller.
    pub unsafe fn remap_shared(target: u64, src: u64, size: u64, flags: i32) -> Remapped {
        let _guard = REMAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let task = task_self();
        let reply = reply_port();
        let mut msg = [0u8; size_of::<RemapReply>()];
        let req = RemapRequest {
            head: Header {
                // MACH_MSGH_BITS(COPY_SEND=19, MAKE_SEND_ONCE=21) | MACH_MSGH_BITS_COMPLEX
                bits: 19 | (21 << 8) | 0x8000_0000,
                size: size_of::<RemapRequest>() as u32,
                remote: task,
                local: reply,
                voucher: 0,
                id: 4813,
            },
            descriptors: 1,
            src_task: task,
            pad1: 0,
            disp_type: 19 << 16,
            ndr: [0, 0, 0, 0, 1, 0, 0, 0],
            target_address: target,
            size,
            mask: 0,
            flags,
            src_address: src,
            copy: 0,
            inheritance: 1,
        };
        // SAFETY: `req` and `msg` are plain-old-data structs of known size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const req).cast::<u8>(),
                msg.as_mut_ptr(),
                size_of::<RemapRequest>(),
            );
        }
        let pack = |lo: u32, hi: u32| (u64::from(hi) << 32) | u64::from(lo);
        let args = [
            msg.as_mut_ptr() as u64,
            3 | MACH64_SEND_KOBJECT_CALL,
            pack(req.head.bits, size_of::<RemapRequest>() as u32),
            pack(task, reply),
            pack(0, 4813),
            pack(1, reply),
        ];
        // SAFETY: `msg` holds the request and space for the reply and trailer.
        let r = unsafe { trap8(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
        // SAFETY: the kernel wrote at most `msg.len()` bytes into `msg`.
        let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
        Remapped {
            msg: r,
            kr: rep.ret,
            address: rep.target_address,
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod mach {
    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    pub fn task_self() -> u32 {
        0
    }

    pub unsafe fn remap_shared(_target: u64, _src: u64, _size: u64, _flags: i32) -> Remapped {
        Remapped {
            msg: 1,
            kr: -1,
            address: 0,
        }
    }
}

/// Minimum physical memory (20 GiB) required for aliasing to be enabled by default.
/// Below this (e.g. 8/16/18 GB Macs), pinned/wired aliased regions can trigger
/// Metal out-of-memory errors; `VICTORY_ALIAS_BUFFERS=1` still opts in.
const DEFAULT_MIN_RAM_BYTES: u64 = 20 << 30;

fn total_ram_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is plain-old-data with `dwLength` initialized.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is valid and `dwLength` is set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.ullTotalPhys == 0 {
        None
    } else {
        Some(status.ullTotalPhys)
    }
}

/// `asked`: `Some(on)` when `VICTORY_ALIAS_BUFFERS` is set (`0` is off, any
/// other value on), `None` when it is not: then on only with >= 20 GiB RAM.
fn decide_enabled(asked: Option<bool>, rosetta: bool, warp: bool, total_ram: Option<u64>) -> bool {
    let default_on = total_ram.is_some_and(|ram| ram >= DEFAULT_MIN_RAM_BYTES);
    asked.unwrap_or(default_on) && rosetta && !warp
}

/// Whether title buffers are aliased on this host: Rosetta + D3DMetal,
/// on by default with >= 20 GiB RAM unless `VICTORY_ALIAS_BUFFERS` overrides it.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let rosetta = crate::watch::rosetta();
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        let asked = std::env::var_os("VICTORY_ALIAS_BUFFERS")
            .map(|_| crate::switch::on("VICTORY_ALIAS_BUFFERS"));
        let ram = total_ram_bytes();
        let on = decide_enabled(asked, rosetta, warp, ram);
        if asked.is_none() && rosetta && !warp && !on {
            tracing::info!(
                ram_gib = ram.map(|b| b / (1 << 30)),
                "zero-copy title buffer aliasing off by default below 20 GiB RAM (set VICTORY_ALIAS_BUFFERS=1 to enable)"
            );
        }
        on
    })
}

/// Probes once whether `mach_vm_remap` onto a D3DMetal write-back L0 buffer
/// works and whether fresh buffers arrive zero-initialized.
fn supported() -> Option<bool> {
    static PROBE: OnceLock<Option<bool>> = OnceLock::new();
    *PROBE.get_or_init(|| {
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        if !crate::watch::rosetta() || warp || mach::task_self() == 0 {
            return None;
        }
        let h = host()?;
        let (buffer, gpu_mem) = h.shared_buffer(PAGE as u64)?;
        // SAFETY: `gpu_mem` is our `PAGE`-byte mapped buffer.
        let zeroed = unsafe {
            gpu_mem.cast::<u64>().read_volatile() == 0
                && gpu_mem.add(PAGE - 8).cast::<u64>().read_volatile() == 0
        };
        // SAFETY: standalone `PAGE`-byte allocation freed below.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        if title.is_null() {
            return None;
        }
        // SAFETY: `title` and `gpu_mem` are both `PAGE`-aligned and `PAGE` bytes.
        let r = unsafe {
            mach::remap_shared(
                title as u64,
                gpu_mem as u64,
                PAGE as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        let ok = if r.msg == 0 && r.kr == 0 && r.address == title as u64 {
            // SAFETY: both pointers point to the same `PAGE` of RAM.
            unsafe {
                title.add(128).write_volatile(0x5a);
                let seen_gpu = gpu_mem.add(128).read_volatile() == 0x5a;
                gpu_mem.add(256).write_volatile(0xa5);
                let seen_title = title.add(256).read_volatile() == 0xa5;
                seen_gpu && seen_title
            }
        } else {
            false
        };
        // SAFETY: `title` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
        drop(buffer);
        if ok {
            tracing::info!(
                zeroed,
                "zero-copy title buffer aliasing active (mach_vm_remap)"
            );
            Some(zeroed)
        } else {
            tracing::warn!(
                ?r,
                "mach_vm_remap aliasing probe failed; falling back to shadow copies"
            );
            None
        }
    })
}

/// CPU-visible backing for XMem physical pages. The placed buffer keeps a
/// CPU mapping of the heap from which reserved resources can take tiles.
pub struct PhysicalHeap {
    backing: ID3D12Resource,
    // A placed buffer must keep its heap alive.
    pub(crate) _heap: ID3D12Heap,
    address: usize,
    len: usize,
    /// Counted in [`IN_PLACE_BYTES`] ([`in_place_heap`]).
    counted: bool,
}

impl PhysicalHeap {
    pub(crate) fn new(device: &ID3D12Device, len: usize) -> windows::core::Result<Self> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: len as u64,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut heap = None;
        let mut backing = None;
        let mut address = std::ptr::null_mut();
        // SAFETY: plain D3D12 creation with output slots on this stack.
        let (heap, backing) = unsafe {
            device.CreateHeap(&desc, &mut heap)?;
            let heap: ID3D12Heap = heap.ok_or_else(windows::core::Error::empty)?;
            device.CreatePlacedResource(
                &heap,
                0,
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut backing,
            )?;
            let backing: ID3D12Resource = backing.ok_or_else(windows::core::Error::empty)?;
            backing.Map(0, None, Some(&mut address))?;
            (heap, backing)
        };
        if address.is_null() {
            return Err(windows::core::Error::empty());
        }
        Ok(Self {
            backing,
            _heap: heap,
            address: address as usize,
            len,
            counted: false,
        })
    }

    /// The GPU's address of the heap's first byte.
    pub fn gpu_address(&self) -> u64 {
        // SAFETY: a live placed buffer.
        unsafe { self.backing.GetGPUVirtualAddress() }
    }

    /// The heap's bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the heap has no bytes (never, for one that was made).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The CPU view, valid for the lifetime of this heap.
    pub fn cpu_address(&self) -> usize {
        self.address
    }

    /// Replaces a Windows file view with the corresponding heap pages.
    ///
    /// # Safety
    /// The caller owns the destination view and keeps this heap alive as
    /// long as the view is mapped.
    pub unsafe fn alias_view(&self, at: usize, offset: usize, len: usize) -> bool {
        if at == 0
            || len == 0
            || [at, offset, len].iter().any(|n| !n.is_multiple_of(PAGE))
            || offset.checked_add(len).is_none_or(|end| end > self.len)
        {
            return false;
        }
        // SAFETY: the caller owns the view; the source range is in the
        // live placed buffer and checked above.
        let remap = unsafe {
            mach::remap_shared(
                at as u64,
                (self.address + offset) as u64,
                len as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        remap.msg == 0 && remap.kr == 0 && remap.address == at as u64
    }
}

impl Drop for PhysicalHeap {
    fn drop(&mut self) {
        // SAFETY: unmapped once, while the placed buffer still owns the view.
        unsafe { self.backing.Unmap(0, None) };
        if self.counted {
            IN_PLACE_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
    }
}

/// Allocates CPU-visible physical tiles for the Rosetta diagnostics trial.
#[cfg(feature = "diagnostics")]
pub fn physical_heap(len: usize) -> Option<PhysicalHeap> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_TILES") && supported().is_some())
        || len == 0
        || !len.is_multiple_of(PAGE)
    {
        return None;
    }
    let host = host()?;
    match PhysicalHeap::new(&host.device, len) {
        Ok(heap) => Some(heap),
        Err(error) => {
            tracing::warn!(len, %error, "physical tile heap unavailable; using a file section");
            None
        }
    }
}

// Title memory in place on XMem physical pages (`VICTORY_INPLACE_SECTIONS=1`).
//
// `alias_region` covers memory the title commits with `XMemVirtualAlloc`.
// Its physical pages (`XMemAllocatePhysicalPages`, mapped with
// `XMemMapPhysicalPages`) are sections instead, and every view of them had a
// host copy: Grand Theft Auto VI's 174 MB buffer of constants among them,
// refreshed nine times a frame. With the switch, a section of at most
// `MOST_SECTION` gets a D3DMetal heap ([`PhysicalHeap`]) when it is made, and
// victory-hle points every view of it at the heap's pages
// ([`PhysicalHeap::alias_view`]): all views and the GPU then see the same
// memory. A title buffer that lies in one run of views showing consecutive
// pages of such a section is that heap's own buffer at the run's offset
// ([`in_place_shadow`]): nothing is copied, refreshed, guarded or copied
// back for it. One GPU address per physical page, whatever the addresses
// the title maps it at (#221's rule for coherence within a submission).

/// The most bytes of one section in place: its pages are the GPU's, wired.
const MOST_SECTION: usize = 512 << 20;

/// Bytes of heaps made by [`in_place_heap`] that are still alive.
static IN_PLACE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    /// Overrides `VICTORY_INPLACE_SECTIONS` on this thread.
    pub(crate) static TEST_IN_PLACE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// The resolver this thread's lookups use in place of the runtime's.
    pub(crate) static TEST_RESOLVER: std::cell::Cell<Option<InPlaceResolver>> = const { std::cell::Cell::new(None) };
}

/// Whether XMem sections may hold title memory in place: the switch, on a
/// host where views can be pointed at D3DMetal's pages.
pub fn in_place_sections() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_IN_PLACE.with(std::cell::Cell::get) {
        return on && supported().is_some();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_INPLACE_SECTIONS") && supported().is_some();
        if on {
            tracing::info!(
                budget_mb = in_place_budget() >> 20,
                "title memory in place on XMem physical pages (VICTORY_INPLACE_SECTIONS)"
            );
        }
        on
    })
}

/// The most bytes of one section in place: `VICTORY_INPLACE_SECTION_MB`,
/// [`MOST_SECTION`] (512 MiB) by default. `bench_physical_heap_sizes`
/// measures what D3DMetal accepts.
fn most_section() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("VICTORY_INPLACE_SECTION_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(MOST_SECTION, |mb| mb << 20)
    })
}

/// Whether memory outside buffer objects (the loose windows of
/// `crate::shadow`) is used in place too where it lies on a section heap:
/// `VICTORY_INPLACE_LOOSE=1`, with [`in_place_sections`].
pub(crate) fn in_place_loose() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_IN_PLACE_LOOSE.with(std::cell::Cell::get) {
        return on && in_place_sections();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_INPLACE_LOOSE")) && in_place_sections()
}

#[cfg(test)]
thread_local! {
    /// Overrides `VICTORY_INPLACE_LOOSE` on this thread.
    pub(crate) static TEST_IN_PLACE_LOOSE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The bytes all in-place heaps may take together:
/// `VICTORY_INPLACE_SECTIONS_MB`, 1024 MiB by default.
fn in_place_budget() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("VICTORY_INPLACE_SECTIONS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(1024)
            << 20
    })
}

/// A heap for a new section of `len` bytes, if its pages are to hold title
/// memory in place ([`in_place_sections`], at most `MOST_SECTION`, within
/// the budget).
pub fn in_place_heap(len: usize) -> Option<std::sync::Arc<PhysicalHeap>> {
    if !in_place_sections() || len == 0 || !len.is_multiple_of(PAGE) || len > most_section() {
        return None;
    }
    let before = IN_PLACE_BYTES.fetch_add(len, Ordering::Relaxed);
    if before + len > in_place_budget() {
        IN_PLACE_BYTES.fetch_sub(len, Ordering::Relaxed);
        static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !TOLD.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                len,
                budget_mb = in_place_budget() >> 20,
                "XMem section not in place: the budget is taken (VICTORY_INPLACE_SECTIONS_MB)"
            );
        }
        return None;
    }
    let made = host().map(|h| PhysicalHeap::new(&h.device, len));
    match made {
        Some(Ok(mut heap)) => {
            heap.counted = true;
            let total = before + len;
            if len >= 16 << 20 {
                tracing::info!(
                    len,
                    total,
                    "XMem section on a D3DMetal heap, for title memory in place"
                );
            } else {
                tracing::debug!(
                    len,
                    total,
                    "XMem section on a D3DMetal heap, for title memory in place"
                );
            }
            Some(std::sync::Arc::new(heap))
        }
        made => {
            IN_PLACE_BYTES.fetch_sub(len, Ordering::Relaxed);
            if let Some(Err(error)) = made {
                tracing::warn!(len, %error, "XMem section not in place: no heap");
            }
            None
        }
    }
}

/// Where title memory `[start, start + len)` lies in a section on a heap,
/// if it lies in one run of views showing consecutive pages of it: the heap
/// and the run's offset into it at `start`.
pub type InPlaceResolver = fn(u64, u64) -> Option<(std::sync::Arc<PhysicalHeap>, usize)>;

static RESOLVER: OnceLock<InPlaceResolver> = OnceLock::new();

/// Installs the runtime's [`InPlaceResolver`] (the first call's).
pub fn set_in_place_resolver(resolve: InPlaceResolver) {
    let _ = RESOLVER.set(resolve);
}

fn resolver() -> Option<InPlaceResolver> {
    #[cfg(test)]
    if let Some(resolve) = TEST_RESOLVER.with(std::cell::Cell::get) {
        return Some(resolve);
    }
    RESOLVER.get().copied()
}

/// Whether title memory `[start, start + len)` would be used in place: the
/// switch is on and the runtime finds it in one run of a section on a heap.
pub fn title_memory_in_place(start: u64, len: u64) -> bool {
    in_place_sections() && resolver().is_some_and(|resolve| resolve(start, len).is_some())
}

/// Title memory in place, by start: its end. Read by [`is_in_place`] for
/// every indirect execution while any is in place.
static IN_PLACE: std::sync::RwLock<std::collections::BTreeMap<u64, u64>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());
static IN_PLACE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// The longest range in [`IN_PLACE`].
static IN_PLACE_SPAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Heaps that title memory in place was found in: kept while the process
/// lives, since shadows and the page tables naming them may outlive the
/// section (a few hundred MB at most, under the budget).
static KEPT: Mutex<Vec<std::sync::Arc<PhysicalHeap>>> = Mutex::new(Vec::new());

/// The title buffer `[start, start + len)` in place, as a zero-copy shadow
/// over its section's heap, if [`in_place_sections`] and the runtime finds
/// it in one run of a section on a heap.
pub(crate) fn in_place_shadow(
    start: u64,
    len: u64,
) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
    if len == 0 || !in_place_sections() || remapped_before(start, len) {
        return None;
    }
    let (heap, offset) = resolver()?(start, len)?;
    if (offset as u64)
        .checked_add(len)
        .is_none_or(|end| end > heap.len() as u64)
    {
        return None;
    }
    crate::host::name(&heap.backing, || "XMem section in place".to_owned());
    // The heap's buffer holds the whole section: the range starts `offset`
    // into it, which views and copies of it count from.
    let shadow = crate::shadow::Shadow::new_aliased_at(
        heap.backing.clone(),
        (heap.cpu_address() + offset) as *mut u8,
        heap.gpu_address() + offset as u64,
        start,
        len,
        offset as u64,
    );
    {
        let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
        if !kept.iter().any(|k| std::sync::Arc::ptr_eq(k, &heap)) {
            kept.push(heap);
        }
    }
    let mut map = IN_PLACE.write().unwrap_or_else(|e| e.into_inner());
    // Two buffers may start at one address: keep the longer end.
    match map.entry(start) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(start + len);
            IN_PLACE_COUNT.fetch_add(1, Ordering::Release);
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let end = entry.get_mut();
            *end = (*end).max(start + len);
        }
    }
    IN_PLACE_SPAN.fetch_max(len, Ordering::Relaxed);
    drop(map);
    note_coverage(true, len);
    if len >= 16 << 20 {
        tracing::info!(
            start = format_args!("{start:#x}"),
            len,
            "title buffer in place on XMem physical pages: no host copy"
        );
    } else {
        tracing::debug!(
            start = format_args!("{start:#x}"),
            len,
            "title buffer in place on XMem physical pages"
        );
    }
    Some(shadow)
}

/// Bytes of title memory found in place (`[0]`) and copied to the host
/// (`[1]`) while sections may hold it in place, and how many ranges each.
static COVERAGE: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Counts `bytes` of title memory found in place, or copied to the host
/// although [`in_place_sections`]: logged each time either total doubles
/// (from 64 MiB), so a run shows how much of its memory the heaps cover.
pub(crate) fn note_coverage(in_place: bool, bytes: u64) {
    if !in_place && !in_place_sections() {
        return;
    }
    let k = usize::from(!in_place);
    let before = COVERAGE[k].fetch_add(bytes, Ordering::Relaxed);
    COVERAGE[k + 2].fetch_add(1, Ordering::Relaxed);
    let after = before + bytes;
    if after >= 64 << 20 && after.leading_zeros() != before.max(1).leading_zeros() {
        let [placed, copied, placed_ranges, copied_ranges] =
            [0, 1, 2, 3].map(|i| COVERAGE[i].load(Ordering::Relaxed));
        tracing::info!(
            in_place_mb = placed >> 20,
            in_place_ranges = placed_ranges,
            copied_mb = copied >> 20,
            copied_ranges,
            "title memory in place against copied to the host, so far"
        );
    }
}

/// Whether any of `[address, address + len)` is title memory in place.
pub(crate) fn is_in_place(address: u64, len: u64) -> bool {
    if IN_PLACE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let end = address.saturating_add(len.max(1));
    let map = IN_PLACE.read().unwrap_or_else(|e| e.into_inner());
    let lowest = address.saturating_sub(IN_PLACE_SPAN.load(Ordering::Relaxed));
    map.range(lowest..end)
        .any(|(&start, &stop)| start < end && stop > address)
}

/// Whether an indirect execution whose arguments are at `[address, address +
/// len)` (and its count at `count`) has them clamped inline, right before it runs:
/// in place, the card reads what earlier work wrote there only then
/// (avatarmanz/VIctory-7900XTX 631400f). `VICTORY_INPLACE_SECTIONS_HOIST=1`
/// lets them be clamped ahead with the others.
pub(crate) fn clamp_in_place_inline(address: u64, len: u64, count: Option<u64>) -> bool {
    static HOIST: OnceLock<bool> = OnceLock::new();
    (is_in_place(address, len) || count.is_some_and(|at| is_in_place(at, 4)))
        && !*HOIST.get_or_init(|| crate::switch::on("VICTORY_INPLACE_SECTIONS_HOIST"))
}

/// The runtime mapped other pages over `[at, at + len)`, or unmapped it:
/// title buffers in place there are looked up again.
pub(crate) fn forget_in_place(at: u64, len: u64) {
    if len == 0 || IN_PLACE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let gone: Vec<u64> = {
        let map = IN_PLACE.read().unwrap_or_else(|e| e.into_inner());
        let lowest = at.saturating_sub(IN_PLACE_SPAN.load(Ordering::Relaxed));
        map.range(lowest..end)
            .filter(|&(_, &stop)| stop > at)
            .map(|(&start, _)| start)
            .collect()
    };
    if gone.is_empty() {
        return;
    }
    let mut map = IN_PLACE.write().unwrap_or_else(|e| e.into_inner());
    let (mut from, mut to) = (at, end);
    for start in gone {
        if let Some(stop) = map.remove(&start) {
            IN_PLACE_COUNT.fetch_sub(1, Ordering::Release);
            from = from.min(start);
            to = to.max(stop);
        }
    }
    drop(map);
    // Never in place there again: a list recorded before the title mapped
    // other pages there and submitted after it read the old pages through
    // the addresses it was recorded with (slots are made again, below; a
    // list's own views cannot be). So that can happen once a range at most,
    // and the log below counts it.
    REMAPPED
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .entry(from)
        .and_modify(|end| *end = (*end).max(to))
        .or_insert(to);
    REMAPPED_COUNT.fetch_add(1, Ordering::Release);
    let gone = crate::shadow::forget_in_place(from, to);
    // Host view slots made from them still show the section's pages: they
    // are made again before the next submission ([`remake_let_go`]), as
    // slots made from copies show the new pages once those are refreshed.
    if !gone.is_empty() {
        let mut gone_ever = LET_GONE.lock().unwrap_or_else(|e| e.into_inner());
        gone_ever.retain(|w| w.strong_count() > 0);
        gone_ever.extend(gone.iter().map(std::sync::Arc::downgrade));
        LET_GONE_ANY.store(true, Ordering::Release);
        drop(gone_ever);
        LET_GO
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(gone);
        LET_GO_PENDING.store(true, Ordering::Release);
    }
    static EVENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        tracing::info!(
            from = format_args!("{from:#x}"),
            to = format_args!("{to:#x}"),
            n,
            "title memory in place let go: the title mapped other pages there"
        );
    }
}

/// Title memory once in place that the title mapped other pages over, by
/// start: its end. Never in place again ([`in_place_shadow`]).
static REMAPPED: std::sync::RwLock<std::collections::BTreeMap<u64, u64>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());
static REMAPPED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Forgets that `[start, end)` was remapped: tests whose memory may lie
/// where another test's was.
#[cfg(test)]
pub(crate) fn clear_remapped(start: u64, end: u64) {
    REMAPPED
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|&from, &mut to| to <= start || from >= end);
}

/// Whether any of `[start, start + len)` was in place once and remapped.
fn remapped_before(start: u64, len: u64) -> bool {
    if REMAPPED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let end = start.saturating_add(len);
    let refused = REMAPPED
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .range(..end)
        .any(|(_, &stop)| stop > start);
    if refused {
        static REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            tracing::info!(
                start = format_args!("{start:#x}"),
                len,
                n,
                "title buffer not in place: the title remapped pages there before"
            );
        }
    }
    refused
}

/// Every title buffer in place let go that something still holds: a list
/// recorded with it reads the section's old pages ([`stale_list`]).
static LET_GONE: Mutex<Vec<std::sync::Weak<crate::shadow::Shadow>>> = Mutex::new(Vec::new());
static LET_GONE_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Lists submitted that read title memory in place let go after they were
/// recorded ([`stale_list`]).
static STALE_LISTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `shadows` (what a list about to be submitted refreshes or writes)
/// hold title memory in place the title has since mapped other pages over:
/// the list was recorded with the section's old pages, which it reads; the
/// console would read the new ones. Counted and logged (it cannot be undone
/// in a recorded list; [`in_place_shadow`] keeps it to once a range).
pub(crate) fn stale_list<'a>(
    shadows: impl IntoIterator<Item = &'a std::sync::Arc<crate::shadow::Shadow>>,
) -> bool {
    if !LET_GONE_ANY.load(Ordering::Acquire) {
        return false;
    }
    let mut gone = LET_GONE.lock().unwrap_or_else(|e| e.into_inner());
    gone.retain(|w| w.strong_count() > 0);
    if gone.is_empty() {
        LET_GONE_ANY.store(false, Ordering::Release);
        return false;
    }
    let held: Vec<*const crate::shadow::Shadow> =
        gone.iter().map(std::sync::Weak::as_ptr).collect();
    drop(gone);
    let Some(shadow) = shadows
        .into_iter()
        .find(|s| held.contains(&std::sync::Arc::as_ptr(s)))
    else {
        return false;
    };
    let n = STALE_LISTS.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        let (start, len) = shadow.range();
        tracing::warn!(
            start = format_args!("{start:#x}"),
            len,
            n,
            "a list recorded before the title mapped other pages over title memory in place reads the old pages"
        );
    }
    true
}

/// [`stale_list`]'s count so far (tests).
#[cfg(test)]
pub(crate) fn stale_lists() -> u64 {
    STALE_LISTS.load(Ordering::Relaxed)
}

/// Title buffers in place let go since the last [`remake_let_go`].
static LET_GO: Mutex<Vec<std::sync::Arc<crate::shadow::Shadow>>> = Mutex::new(Vec::new());
static LET_GO_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes again every host view slot made from title memory in place that
/// was let go since the last call, from the record it was made from: the
/// view then shows what the title maps there now. Called before each
/// submission's heap sync; returns the slots made again.
pub(crate) fn remake_let_go() -> usize {
    if !LET_GO_PENDING.swap(false, Ordering::AcqRel) {
        return 0;
    }
    let gone = std::mem::take(&mut *LET_GO.lock().unwrap_or_else(|e| e.into_inner()));
    if gone.is_empty() {
        return 0;
    }
    let made = crate::descriptor::remake_holding(&gone);
    tracing::info!(
        buffers = gone.len(),
        slots = made,
        "host views of title memory in place let go made again"
    );
    made
}

/// Aliases `[base, base + len)` onto a new write-back L0 D3D12 buffer so CPU
/// and GPU share physical pages with zero copy.
pub fn alias_region(base: usize, len: usize, preserve_contents: bool) -> bool {
    static LOCK: Mutex<()> = Mutex::new(());
    let Some(zeroed) = supported() else {
        return false;
    };
    if base == 0 || len < PAGE || !base.is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
        return false;
    }
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if crate::shadow::has_aliased_overlap(base as u64, end as u64) {
        return false;
    }
    let Some(h) = host() else {
        return false;
    };
    let Some((buffer, gpu_mem)) = h.shared_buffer(len as u64) else {
        return false;
    };
    // SAFETY: `base` and `gpu_mem` are `len`-byte owned mappings.
    unsafe {
        if preserve_contents {
            std::ptr::copy_nonoverlapping(base as *const u8, gpu_mem, len);
        } else if !zeroed {
            std::ptr::write_bytes(gpu_mem, 0, len);
        }
        let r = mach::remap_shared(
            base as u64,
            gpu_mem as u64,
            len as u64,
            mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
        );
        if r.msg != 0 || r.kr != 0 || r.address != base as u64 {
            tracing::warn!(
                base = format_args!("{base:#x}"),
                len,
                ?r,
                "mach_vm_remap failed; keeping plain memory"
            );
            return false;
        }
    }
    crate::host::name(&buffer, || {
        format!("aliased title buffer {base:#x}+{len:#x}")
    });
    // SAFETY: `buffer` is alive.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = crate::shadow::Shadow::new_aliased(buffer, gpu_mem, gpu, base as u64, len as u64);
    crate::shadow::register_aliased(shadow);
    if len >= 16 << 20 {
        tracing::info!(
            base = format_args!("{base:#x}"),
            len,
            "large title buffer aliased zero-copy"
        );
    } else {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            "title buffer aliased zero-copy"
        );
    }
    true
}

/// Removes an aliased region registered at `base`.
pub fn unalias_region(base: usize) -> bool {
    crate::shadow::unregister_aliased(base as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, RemoveVectoredExceptionHandler, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
    };

    struct AliasedAlloc {
        ptr: *mut u8,
        len: usize,
        aliased: bool,
    }

    impl AliasedAlloc {
        fn new(len: usize, init: Option<&dyn Fn(*mut u8)>, preserve: bool) -> Self {
            // SAFETY: allocating `len` bytes of RW memory for the test.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            if let Some(f) = init {
                f(ptr);
            }
            let aliased = alias_region(ptr as usize, len, preserve);
            assert!(aliased, "alias_region failed");
            Self { ptr, len, aliased }
        }
    }

    impl Drop for AliasedAlloc {
        fn drop(&mut self) {
            if self.aliased {
                unalias_region(self.ptr as usize);
            }
            // As the runtime does when the title unmaps memory: the guards
            // let go of its pages. A tracker some test or copy still holds
            // left them in the guard set otherwise, and a test running at
            // the same time that was given the address had them skipped as
            // guarded already, so they stayed writable (#31:
            // `takes_guard_one_view_per_call` after
            // `aliased_gpu_writes_notify_overlapping_texture_tracker`).
            crate::watch::title_memory_remapped(self.ptr as usize, self.len);
            // SAFETY: `self.ptr` was allocated by `VirtualAlloc`.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        }
    }

    fn committed(
        device: &ID3D12Device,
        heap_type: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            ..Default::default()
        };
        let mut res = None;
        // SAFETY: valid committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(size),
                    state,
                    None,
                    &mut res,
                )
                .unwrap();
        }
        res.unwrap()
    }

    fn run_copy(device: &ID3D12Device, f: impl FnOnce(&ID3D12GraphicsCommandList)) {
        // SAFETY: standalone queue/list/fence for test verification.
        unsafe {
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
            f(&list);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = windows::core::Interface::cast(&list).unwrap();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            if fence.GetCompletedValue() < 1 {
                fence.SetEventOnCompletion(1, event).unwrap();
                WaitForSingleObject(event, 5000);
            }
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
    }

    fn gpu_read(device: &ID3D12Device, src: &ID3D12Resource, offset: u64, len: usize) -> Vec<u8> {
        let back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            len as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: `back` and `src` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(&back, 0, src, offset, len as u64)
        });
        let mut out = vec![0u8; len];
        // SAFETY: `back` is a readback buffer of `len` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            back.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), len);
            back.Unmap(0, None);
        }
        out
    }

    fn gpu_write(device: &ID3D12Device, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) {
        let up = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: `up` is an upload buffer of `bytes.len()` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: `dst` and `up` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(dst, offset, &up, 0, bytes.len() as u64)
        });
    }

    #[test]
    fn switch_respects_rosetta_and_warp() {
        assert!(!decide_enabled(Some(false), true, false, Some(32 << 30)));
        assert!(!decide_enabled(Some(true), false, false, Some(32 << 30)));
        assert!(!decide_enabled(Some(true), true, true, Some(32 << 30)));
        assert!(decide_enabled(Some(true), true, false, Some(32 << 30)));
        assert!(decide_enabled(Some(true), true, false, Some(16 << 30)));
    }

    #[test]
    fn unset_is_on_under_rosetta_only() {
        assert!(decide_enabled(None, true, false, Some(24 << 30)));
        assert!(!decide_enabled(None, false, false, Some(24 << 30)));
        assert!(!decide_enabled(None, true, true, Some(24 << 30)));
    }

    #[test]
    fn unset_requires_at_least_20_gib_ram() {
        assert!(!decide_enabled(None, true, false, None));
        assert!(!decide_enabled(None, true, false, Some(8 << 30)));
        assert!(!decide_enabled(None, true, false, Some(16 << 30)));
        assert!(!decide_enabled(None, true, false, Some(18 << 30)));
        assert!(decide_enabled(None, true, false, Some(20 << 30)));
        assert!(decide_enabled(None, true, false, Some(32 << 30)));
        assert!(total_ram_bytes().is_some());
    }

    #[test]
    fn aliased_buffer_is_coherent_with_gpu_copies() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 4 * PAGE;
        let alloc = AliasedAlloc::new(
            len,
            Some(&|p| {
                for i in (0..len).step_by(4096) {
                    // SAFETY: `p` has `len` bytes.
                    unsafe { p.add(i).write((i >> 12) as u8) };
                }
            }),
            true,
        );
        // Initial contents survived remap.
        for i in (0..len).step_by(4096) {
            // SAFETY: `alloc.ptr` has `len` bytes.
            assert_eq!(unsafe { alloc.ptr.add(i).read_volatile() }, (i >> 12) as u8);
        }

        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(shadow.clean());
        assert_eq!(shadow.update(alloc.ptr as usize, len), 0);
        assert_eq!(shadow.copy_back(alloc.ptr as usize, len), 0);

        for round in 0..8u8 {
            let off = len - 2 * PAGE;
            for i in 0..PAGE {
                let b = (i as u8).wrapping_mul(31) ^ round;
                // SAFETY: inside `alloc`.
                unsafe { alloc.ptr.add(off + i).write_volatile(b) };
            }
            let got = gpu_read(&h.device, &shadow.buffer, off as u64, PAGE);
            for (i, &b) in got.iter().enumerate() {
                assert_eq!(b, (i as u8).wrapping_mul(31) ^ round);
            }

            let new: Vec<u8> = (0..PAGE).map(|i| (i as u8) ^ round ^ 0xc3).collect();
            gpu_write(&h.device, &shadow.buffer, (len - PAGE) as u64, &new);
            // SAFETY: inside `alloc`.
            let cpu = unsafe { std::slice::from_raw_parts(alloc.ptr.add(len - PAGE), PAGE) };
            assert_eq!(cpu, &new[..]);
        }
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn aliased_gpu_writes_notify_overlapping_texture_tracker() {
        if !crate::watch::rosetta() || !crate::watch::fault_tracking() {
            return;
        }
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 2 * PAGE;
        // SAFETY: allocating `len` bytes with `MEM_WRITE_WATCH` for tracker test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let alloc = AliasedAlloc {
            ptr,
            len,
            aliased: true,
        };
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let base = alloc.ptr as usize;

        let tex = crate::watch::track(base, len, false).expect("texture tracker");
        let _ = tex.take(base, base + len);
        assert!(tex.clean());

        assert_eq!(shadow.copy_back(base, PAGE), 0);
        assert!(tex.changed(base / PAGE));
        assert!(!tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert!(tex.clean());
        assert_eq!(shadow.copy_back_marked(&[base + PAGE]), 0);
        assert!(!tex.changed(base / PAGE));
        assert!(tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert_eq!(shadow.copy_back_page(base), 0);
        assert!(tex.changed(base / PAGE));
    }

    #[test]
    fn aliased_shadow_refreshes_widened_rgb32_view() {
        if !crate::watch::rosetta() {
            return;
        }
        let len = PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let widened = shadow
            .widened(alloc.ptr as u64, 48, 0x3f80_0000)
            .expect("widened view");
        assert!(!shadow.clean());

        let words = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // SAFETY: writing 48 bytes at the start of `alloc.ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), alloc.ptr.cast::<u32>(), words.len());
        }
        assert_eq!(shadow.update(alloc.ptr as usize, 48), 0);
        assert_eq!(
            widened.read_host(16),
            vec![
                1,
                2,
                3,
                0x3f80_0000,
                4,
                5,
                6,
                0x3f80_0000,
                7,
                8,
                9,
                0x3f80_0000,
                10,
                11,
                12,
                0x3f80_0000,
            ]
        );
    }

    #[test]
    fn aliased_memory_supports_virtual_query_and_write_guard() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        static GUARD_AT: AtomicUsize = AtomicUsize::new(0);
        static GUARD_HIT: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: `info` is provided by Windows/Wine exception dispatch.
            unsafe {
                let rec = &*(*info).ExceptionRecord;
                let page = GUARD_AT.load(Ordering::SeqCst);
                if rec.ExceptionCode as u32 == 0xC000_0005 && page != 0 {
                    let addr = rec.ExceptionInformation[1];
                    if addr >= page && addr < page + PAGE {
                        let mut old = 0;
                        VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old);
                        GUARD_HIT.store(true, Ordering::SeqCst);
                        return -1;
                    }
                }
                0
            }
        }

        let len = 2 * PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        // SAFETY: querying our own allocation.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let q = VirtualQuery(alloc.ptr.add(PAGE).cast(), &mut info, size_of_val(&info));
            assert_ne!(q, 0);
            assert_eq!(info.State, MEM_COMMIT);
            assert_eq!(info.Protect, PAGE_READWRITE);

            let handler = AddVectoredExceptionHandler(1, Some(on_fault));
            let page = alloc.ptr.add(PAGE);
            GUARD_AT.store(page as usize, Ordering::SeqCst);
            GUARD_HIT.store(false, Ordering::SeqCst);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), PAGE, PAGE_READONLY, &mut old),
                0
            );
            page.add(100).write_volatile(0x77);
            let hit = GUARD_HIT.load(Ordering::SeqCst);
            GUARD_AT.store(0, Ordering::SeqCst);
            RemoveVectoredExceptionHandler(handler);
            assert!(hit, "VirtualProtect write guard did not fault");
        }
        assert_eq!(
            gpu_read(&h.device, &shadow.buffer, (PAGE + 100) as u64, 1),
            vec![0x77]
        );
    }

    #[test]
    fn unalias_region_removes_shadow_and_allows_virtual_free() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: allocating `len` bytes for explicit unalias + free test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let shadow = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(unalias_region(ptr as usize));
        assert!(crate::shadow::find(ptr as u64, len as u64).is_none_or(|s| !s.is_aliased()));
        // SAFETY: `ptr` was allocated by `VirtualAlloc`.
        assert_ne!(unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) }, 0);
        drop(shadow);
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The view the test resolver below answers for, and its heap.
    static TEST_VIEW: AtomicUsize = AtomicUsize::new(0);
    static TEST_HEAP: Mutex<Option<std::sync::Arc<PhysicalHeap>>> = Mutex::new(None);
    const TEST_SECTION: usize = 4 * PAGE;

    /// As victory-hle's resolver would for a section of `TEST_SECTION` bytes
    /// shown whole, in order, at `TEST_VIEW`.
    fn test_resolve(start: u64, len: u64) -> Option<(std::sync::Arc<PhysicalHeap>, usize)> {
        let base = TEST_VIEW.load(Ordering::Relaxed) as u64;
        let heap = TEST_HEAP.lock().unwrap().clone()?;
        (start >= base && start + len <= base + TEST_SECTION as u64)
            .then(|| (heap, (start - base) as usize))
    }

    /// A title buffer on XMem physical pages whose section is on a heap is
    /// that heap's own memory: the GPU reads what the title wrote through
    /// any view of the section, the title reads what the GPU wrote, and
    /// nothing is copied either way. When the title maps other pages there,
    /// the buffer is looked up again.
    #[test]
    fn title_buffers_on_section_heaps_are_used_in_place() {
        use crate::backend::HostGpu;
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        if !crate::watch::rosetta() || supported().is_none() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        TEST_IN_PLACE.with(|c| c.set(Some(true)));
        let heap = in_place_heap(TEST_SECTION).expect("a heap for the section");
        // The section and two views of it, both shown on the heap's pages,
        // as victory-hle maps the title's.
        // SAFETY: a fresh section and views of it, ours, unmapped below.
        let (section, views) = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                TEST_SECTION as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views = [0, 1].map(|_| {
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, TEST_SECTION)
                    .Value
                    .cast::<u8>()
            });
            for view in views {
                assert!(!view.is_null());
                assert!(heap.alias_view(view as usize, 0, TEST_SECTION));
            }
            (section, views)
        };
        clear_remapped(views[0] as u64, views[0] as u64 + TEST_SECTION as u64);
        TEST_VIEW.store(views[0] as usize, Ordering::Relaxed);
        *TEST_HEAP.lock().unwrap() = Some(heap.clone());
        TEST_RESOLVER.with(|c| c.set(Some(test_resolve)));
        let start = views[0] as u64 + PAGE as u64;
        let len = 2 * PAGE;
        // A resource of our own as the title buffer's.
        let resource = 0x7e57_1000usize;
        crate::shadow::register(resource, start, len as u64);

        let shadow = crate::shadow::find(start + 16, 64).expect("a shadow in place");
        assert!(shadow.is_aliased());
        assert_eq!(
            shadow.gpu(start + 16),
            heap.gpu_address() + PAGE as u64 + 16
        );
        // Views and copies count from where the range is in the heap's
        // buffer, which holds the whole section.
        assert_eq!(shadow.offset(start + 16), PAGE as u64 + 16);
        let (viewed, exact) = crate::shadow::for_view(start + 16, 64, 4).expect("a view's copy");
        assert!(!exact && std::sync::Arc::ptr_eq(&viewed, &shadow));
        // Elements of 12 bytes off the heap's (the range is 64 KiB into it):
        // a copy of the view's own, not a view at the wrong place.
        let (_, exact) = crate::shadow::for_view(start + 16, 48, 12).expect("a view's copy");
        assert!(exact);
        assert!(is_in_place(start + 8, 20));
        assert!(!is_in_place(start - 64, 20));
        assert!(clamp_in_place_inline(start + 8, 20, None));
        assert!(clamp_in_place_inline(start - 64, 20, Some(start + 4)));
        // A shorter buffer from the same start found in place too leaves the
        // longer one's end in place.
        assert!(in_place_shadow(start, PAGE as u64).is_some());
        assert!(is_in_place(start + PAGE as u64 + 8, 20));
        // The title writes through the other view; the GPU reads it there.
        let bytes: Vec<u8> = (0..256).map(|i| (i * 7) as u8).collect();
        // SAFETY: inside the second view.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), views[1].add(PAGE + 256), 256) };
        assert_eq!(
            gpu_read(&h.device, &shadow.buffer, (PAGE + 256) as u64, 256),
            bytes
        );
        // The GPU writes; the title reads it through the first view.
        let new: Vec<u8> = (0..256).map(|i| (i as u8) ^ 0x5a).collect();
        gpu_write(&h.device, &shadow.buffer, (PAGE + 512) as u64, &new);
        // SAFETY: inside the first view.
        let seen = unsafe { std::slice::from_raw_parts(views[0].add(PAGE + 512), 256) };
        assert_eq!(seen, &new[..]);
        assert_eq!(shadow.update(start as usize, len), 0);
        assert_eq!(shadow.copy_back(start as usize, len), 0);

        // A list recorded now reads these pages; submitted after the title
        // maps other pages there, it is counted as reading the old ones.
        let before = h.new_list().expect("a list");
        shadow.refresh(&before, start, len as u64);
        before.close();
        let stale = stale_lists();

        // The title maps other pages over part of it: looked up again, and
        // (with no section there any more) copied.
        TEST_RESOLVER.with(|c| c.set(None));
        crate::watch::title_memory_remapped(start as usize + PAGE, PAGE);
        assert!(!is_in_place(start, len as u64));
        let again = crate::shadow::find(start + 16, 64).expect("a copy");
        assert!(!again.is_aliased());
        // Never in place there again, even where a section on a heap shows.
        TEST_RESOLVER.with(|c| c.set(Some(test_resolve)));
        assert!(
            in_place_shadow(start, len as u64).is_none(),
            "remapped before"
        );
        TEST_RESOLVER.with(|c| c.set(None));
        h.execute(&[&before]);
        assert!(
            stale_lists() > stale,
            "the list recorded before the remap is counted"
        );
        let after = h.new_list().expect("a list");
        again.refresh(&after, start, len as u64);
        after.close();
        let stale = stale_lists();
        h.execute(&[&after]);
        assert_eq!(stale_lists(), stale, "a list recorded after it is not");

        drop((again, shadow));
        crate::shadow::unregister(resource, start);
        TEST_IN_PLACE.with(|c| c.set(None));
        *TEST_HEAP.lock().unwrap() = None;
        // SAFETY: our views and section, no longer used.
        unsafe {
            for view in views {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.cast() });
            }
            CloseHandle(section);
        }
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Bench (`--ignored --nocapture`): how large a `PhysicalHeap` D3DMetal
    /// accepts and what it costs, for title memory in place at full
    /// coverage (Grand Theft Auto VI asks for 6.1 GiB of physical pages in
    /// one call). Per size: creation, first touch of every 16 KiB page,
    /// pointing views at its first and last 64 MiB, and a GPU copy out of
    /// and into the last page (offsets past 4 GiB).
    #[test]
    #[ignore = "bench: allocates up to VICTORY_HEAP_BENCH_GIB (12) GiB of GPU memory"]
    fn bench_physical_heap_sizes() {
        if !crate::watch::rosetta() || supported().is_none() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let most: usize = std::env::var("VICTORY_HEAP_BENCH_GIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        const VIEW: usize = 64 << 20;
        let mut held: Vec<PhysicalHeap> = Vec::new();
        let mut total = 0usize;
        for mib in [256usize, 1024, 2048, 4096, 6272, 8192, 12288] {
            let len = mib << 20;
            if mib > most << 10 {
                break;
            }
            let t = std::time::Instant::now();
            let heap = match PhysicalHeap::new(&h.device, len) {
                Ok(heap) => heap,
                Err(error) => {
                    println!("heap {mib} MiB: refused: {error}");
                    break;
                }
            };
            let made = t.elapsed();
            let base = heap.cpu_address() as *mut u8;
            let t = std::time::Instant::now();
            let mut sum = 0u64;
            for at in (0..len).step_by(16 << 10) {
                // SAFETY: inside the heap's mapping.
                unsafe {
                    sum += u64::from(base.add(at).read_volatile());
                    base.add(at).write_volatile(at as u8);
                }
            }
            let touched = t.elapsed();
            // Views of the first and last 64 MiB, as the runtime's.
            let t = std::time::Instant::now();
            let mut ok = true;
            let mut views = Vec::new();
            for offset in [0, len - VIEW] {
                // SAFETY: a fresh allocation of ours.
                let view = unsafe {
                    VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
                }
                .cast::<u8>();
                assert!(!view.is_null());
                // SAFETY: our view; the heap outlives it.
                ok &= unsafe { heap.alias_view(view as usize, offset, VIEW) };
                views.push(view);
            }
            let aliased = t.elapsed();
            // The title writes the last page through its view; the GPU
            // reads it and writes the page before it.
            let pattern: Vec<u8> = (0..4096).map(|i| (i * 13 + mib) as u8).collect();
            // SAFETY: inside the last view.
            unsafe {
                std::ptr::copy_nonoverlapping(pattern.as_ptr(), views[1].add(VIEW - 4096), 4096)
            };
            let t = std::time::Instant::now();
            let read = gpu_read(&h.device, &heap.backing, (len - 4096) as u64, 4096);
            let copied = t.elapsed();
            let t = std::time::Instant::now();
            let _ = gpu_read(&h.device, &heap.backing, (len / 2) as u64, 4096);
            let again = t.elapsed();
            gpu_write(&h.device, &heap.backing, (len - 8192) as u64, &pattern);
            // SAFETY: inside the last view.
            let seen = unsafe { std::slice::from_raw_parts(views[1].add(VIEW - 8192), 4096) };
            // SAFETY: a live placed buffer.
            let gpu = unsafe { heap.backing.GetGPUVirtualAddress() };
            // SAFETY: querying device health.
            let alive = unsafe { h.device.GetDeviceRemovedReason() }.is_ok();
            total += len;
            println!(
                "heap {mib} MiB (total held {} MiB): made {:.1} ms, first touch {:.1} ms ({:.2} us/16K page, zero {}), 2 views {:.2} ms ok {ok}, gpu read ok {} ({:.2} ms, a second {:.2} ms), gpu write seen {}, gpu va {gpu:#x}, device alive {alive}",
                total >> 20,
                made.as_secs_f64() * 1e3,
                touched.as_secs_f64() * 1e3,
                touched.as_secs_f64() * 1e6 / (len >> 14) as f64,
                sum == 0,
                aliased.as_secs_f64() * 1e3,
                read == pattern,
                copied.as_secs_f64() * 1e3,
                again.as_secs_f64() * 1e3,
                seen == &pattern[..],
            );
            for view in views {
                // SAFETY: our allocation.
                unsafe { VirtualFree(view.cast(), 0, MEM_RELEASE) };
            }
            if !alive {
                break;
            }
            let t = std::time::Instant::now();
            drop(heap);
            total -= len;
            println!("  released in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        // Many heaps held together, all touched: the title's whole pool.
        let together: usize = std::env::var("VICTORY_HEAP_BENCH_TOTAL_GIB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let t = std::time::Instant::now();
        while total < together << 30 {
            let len = 2usize << 30;
            let Ok(heap) = PhysicalHeap::new(&h.device, len) else {
                println!("held {} MiB, then refused", total >> 20);
                break;
            };
            let base = heap.cpu_address() as *mut u8;
            for at in (0..len).step_by(16 << 10) {
                // SAFETY: inside the heap's mapping.
                unsafe { base.add(at).write_volatile(1) };
            }
            total += len;
            held.push(heap);
        }
        let last = held.last().expect("a heap");
        let read = gpu_read(&h.device, &last.backing, (2u64 << 30) - 16384, 4096);
        // SAFETY: querying device health.
        let alive = unsafe { h.device.GetDeviceRemovedReason() }.is_ok();
        println!(
            "held {} MiB in {} heaps, touched, in {:.1} ms; gpu read of the last ok {}; device alive {alive}",
            total >> 20,
            held.len(),
            t.elapsed().as_secs_f64() * 1e3,
            read[0] == 1,
        );
        let t = std::time::Instant::now();
        drop(held);
        println!("released in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }

    /// The view and heap [`loose_resolve`] answers for.
    static LOOSE_VIEW: AtomicUsize = AtomicUsize::new(0);
    static LOOSE_HEAP: Mutex<Option<std::sync::Arc<PhysicalHeap>>> = Mutex::new(None);

    fn loose_resolve(start: u64, len: u64) -> Option<(std::sync::Arc<PhysicalHeap>, usize)> {
        let base = LOOSE_VIEW.load(Ordering::Relaxed) as u64;
        let heap = LOOSE_HEAP.lock().unwrap().clone()?;
        (start >= base && start + len <= base + TEST_SECTION as u64)
            .then(|| (heap, (start - base) as usize))
    }

    /// Memory outside any buffer object that lies on a section heap is used
    /// in place too with `VICTORY_INPLACE_LOOSE`: the pages a range touches
    /// are the heap's own (no 4 MiB copy, nothing refreshed or copied
    /// back), ranges inside share them, and when the title maps other
    /// pages there the memory is looked up again.
    #[test]
    fn loose_memory_on_section_heaps_is_used_in_place() {
        if !crate::watch::rosetta() || supported().is_none() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        TEST_IN_PLACE.with(|c| c.set(Some(true)));
        let heap = in_place_heap(TEST_SECTION).expect("a heap for the section");
        // SAFETY: a fresh allocation of ours, pointed at the heap's pages.
        let view = unsafe {
            let view = VirtualAlloc(
                std::ptr::null(),
                TEST_SECTION,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            assert!(!view.is_null());
            assert!(heap.alias_view(view as usize, 0, TEST_SECTION));
            view
        };
        LOOSE_VIEW.store(view as usize, Ordering::Relaxed);
        *LOOSE_HEAP.lock().unwrap() = Some(heap.clone());
        TEST_RESOLVER.with(|c| c.set(Some(loose_resolve)));

        // Without the switch: a copy, as before.
        TEST_IN_PLACE_LOOSE.with(|c| c.set(Some(false)));
        let far = view as u64 + 3 * PAGE as u64 + 64;
        let copy = crate::shadow::find(far, 64).expect("a window's copy");
        assert!(!copy.is_aliased());
        drop(copy);
        crate::watch::title_memory_remapped(view as usize, TEST_SECTION);

        // With it: the heap's own pages.
        TEST_IN_PLACE_LOOSE.with(|c| c.set(Some(true)));
        let at = view as u64 + PAGE as u64 + 100;
        let shadow = crate::shadow::find(at, 64).expect("memory in place");
        assert!(shadow.is_aliased());
        assert_eq!(shadow.gpu(at), heap.gpu_address() + PAGE as u64 + 100);
        assert_eq!(shadow.offset(at), PAGE as u64 + 100);
        assert!(is_in_place(at, 64));
        let again = crate::shadow::find(at + 1000, 16).expect("the same pages");
        assert!(std::sync::Arc::ptr_eq(&shadow, &again));
        let bytes: Vec<u8> = (0..64).map(|i| (i * 11 + 3) as u8).collect();
        // SAFETY: inside the view.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), view.add(PAGE + 100), 64) };
        assert_eq!(gpu_read(&h.device, &shadow.buffer, (PAGE + 100) as u64, 64), bytes);
        let new: Vec<u8> = (0..64).map(|i| (i as u8) ^ 0xc3).collect();
        gpu_write(&h.device, &shadow.buffer, (PAGE + 512) as u64, &new);
        // SAFETY: inside the view.
        assert_eq!(unsafe { std::slice::from_raw_parts(view.add(PAGE + 512), 64) }, &new[..]);
        assert_eq!(shadow.update(at as usize, 64), 0);
        assert_eq!(shadow.copy_back(at as usize, 64), 0);

        // The title maps other pages there: looked up again, now copied.
        TEST_RESOLVER.with(|c| c.set(None));
        crate::watch::title_memory_remapped(view as usize + PAGE, PAGE);
        assert!(!is_in_place(at, 64));
        let copied = crate::shadow::find(at, 64).expect("a copy");
        assert!(!copied.is_aliased());

        drop((copied, again, shadow));
        crate::watch::title_memory_remapped(view as usize, TEST_SECTION);
        TEST_IN_PLACE.with(|c| c.set(None));
        TEST_IN_PLACE_LOOSE.with(|c| c.set(None));
        *LOOSE_HEAP.lock().unwrap() = None;
        // SAFETY: our allocation, no longer used.
        unsafe { VirtualFree(view.cast(), 0, MEM_RELEASE) };
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }
}
