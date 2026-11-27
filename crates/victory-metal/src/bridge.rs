//! The Windows side of the bridge: loading the anchor, getting the
//! unixlib's call table from Wine, and the calls.

use std::ffi::c_void;
use std::fmt;
use std::time::Duration;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress, LoadLibraryW};

use crate::abi;

/// `NTSTATUS (WINAPI *)(unixlib_handle_t, unsigned int, void *)`.
type Dispatcher = unsafe extern "system" fn(u64, u32, *mut c_void) -> i32;
/// `NtQueryVirtualMemory`.
type QueryVirtualMemory =
    unsafe extern "system" fn(isize, *const c_void, u32, *mut c_void, usize, *mut usize) -> i32;

/// Wine's `MemoryWineUnixFuncs` information class: the call table of the
/// unixlib paired with the module at the address asked about.
const MEMORY_WINE_UNIX_FUNCS: u32 = 1000;
/// `GetCurrentProcess()`.
const CURRENT_PROCESS: isize = -1;
const STATUS_INVALID_PARAMETER: i32 = 0xC000_000D_u32 as i32;

/// Why the bridge could not be opened.
#[derive(Debug)]
pub enum OpenError {
    /// Not running under Wine: there is no unixlib to call.
    NotWine,
    /// [`crate::BRIDGE_VAR`] is not set.
    NotAsked,
    /// The anchor DLL could not be loaded (the Windows error code).
    Load(String, u32),
    /// Wine loaded the anchor without a unixlib (the status Wine gave):
    /// `WINEDLLPATH` did not name the folder that holds the pair when Wine
    /// started, or the anchor is not marked as a builtin.
    NoUnixlib(i32),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotWine => write!(f, "not running under Wine"),
            Self::NotAsked => write!(f, "{} is not set", crate::BRIDGE_VAR),
            Self::Load(path, code) => write!(f, "could not load {path} (error {code})"),
            Self::NoUnixlib(status) => write!(
                f,
                "Wine has no unixlib for the anchor (status {:#x}): WINEDLLPATH must name the \
                 folder holding x86_64-windows/victory_metal.dll and x86_64-unix/victory_metal.so",
                *status as u32
            ),
        }
    }
}

impl std::error::Error for OpenError {}

/// A call that failed on the macOS side.
#[derive(Debug)]
pub struct CallError {
    /// The status the unixlib returned.
    pub status: i32,
    /// What it said went wrong.
    pub message: String,
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "status {:#x}: {}", self.status as u32, self.message)
    }
}

impl std::error::Error for CallError {}

/// The Metal device behind the bridge.
#[derive(Debug, Clone)]
pub struct Device {
    pub name: String,
    pub unified_memory: bool,
    pub registry_id: u64,
    pub max_buffer_length: u64,
    pub recommended_working_set: u64,
    /// The CPU's page size on the macOS side of this process.
    pub page_size: u64,
}

/// Where the time of a [`Bridge::fill`] went, by the unixlib's clock.
#[derive(Debug, Clone, Copy)]
pub struct FillTimes {
    /// Making the no-copy buffer over the memory.
    pub wrap: Duration,
    /// Recording the dispatches.
    pub encode: Duration,
    /// From the commit to the end of the wait for completion.
    pub wait: Duration,
    /// What the GPU reports between the command buffer's start and end.
    pub gpu: Duration,
}

/// The unixlib's call table, from the anchor Wine paired it with.
pub struct Bridge {
    handle: u64,
    dispatcher: Dispatcher,
    _anchor: HMODULE,
}

// SAFETY: the handle and the dispatcher are process-wide values Wine hands
// out once, and the unixlib serializes its own calls.
unsafe impl Send for Bridge {}
// SAFETY: as above.
unsafe impl Sync for Bridge {}

impl Bridge {
    /// Opens the bridge [`crate::BRIDGE_VAR`] names.
    pub fn open() -> Result<Self, OpenError> {
        let path = std::env::var_os(crate::BRIDGE_VAR).ok_or(OpenError::NotAsked)?;
        Self::open_at(&path.to_string_lossy())
    }

    /// Opens the bridge whose anchor DLL is at `path`, as the process sees
    /// it.
    pub fn open_at(path: &str) -> Result<Self, OpenError> {
        // SAFETY: ntdll is always loaded; the names are NUL-terminated.
        let (query, dispatcher) = unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            if GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_none() {
                return Err(OpenError::NotWine);
            }
            let query = GetProcAddress(ntdll, c"NtQueryVirtualMemory".as_ptr().cast());
            // An exported variable: the address of the dispatcher's pointer.
            let dispatcher = GetProcAddress(ntdll, c"__wine_unix_call_dispatcher".as_ptr().cast());
            match (query, dispatcher) {
                (Some(q), Some(d)) => (
                    std::mem::transmute::<unsafe extern "system" fn() -> isize, QueryVirtualMemory>(
                        q,
                    ),
                    *(d as *const Dispatcher),
                ),
                _ => return Err(OpenError::NotWine),
            }
        };
        let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated path.
        let anchor = unsafe { LoadLibraryW(wide.as_ptr()) };
        if anchor.is_null() {
            // SAFETY: no other call in between.
            let code = unsafe { windows_sys::Win32::Foundation::GetLastError() };
            return Err(OpenError::Load(path.to_owned(), code));
        }
        let mut handle = 0u64;
        // SAFETY: the module is loaded; the out block is a handle's size.
        let status = unsafe {
            query(
                CURRENT_PROCESS,
                anchor.cast_const(),
                MEMORY_WINE_UNIX_FUNCS,
                (&raw mut handle).cast(),
                size_of::<u64>(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 || handle == 0 {
            return Err(OpenError::NoUnixlib(status));
        }
        Ok(Self {
            handle,
            dispatcher,
            _anchor: anchor,
        })
    }

    /// # Safety
    /// `args` is the block `code` reads and writes, or null for
    /// [`abi::NOOP`].
    unsafe fn call(&self, code: u32, args: *mut c_void) -> i32 {
        // SAFETY: the handle is the unixlib's table, `code` one of its
        // entries, and the caller vouches for the block.
        unsafe { (self.dispatcher)(self.handle, code, args) }
    }

    /// A call that does nothing: what the way into macOS code and back
    /// costs.
    pub fn noop(&self) {
        // SAFETY: the no-op reads nothing.
        unsafe { self.call(abi::NOOP, std::ptr::null_mut()) };
    }

    /// Makes the Metal device, its queue and the fill pipeline (the first
    /// time) and describes the device.
    pub fn device(&self) -> Result<Device, CallError> {
        let mut d: abi::Device = abi::block(|d: &mut abi::Device, n| d.size = n);
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::OPEN, (&raw mut d).cast()) };
        if status != 0 {
            return Err(failed(status, &d.error));
        }
        Ok(Device {
            name: abi::text(&d.name),
            unified_memory: d.unified_memory != 0,
            registry_id: d.registry_id,
            max_buffer_length: d.max_buffer_length,
            recommended_working_set: d.recommended_working_set,
            page_size: d.page_size,
        })
    }

    /// Lets the GPU write `memory` in place, with no copy: `dispatches`
    /// compute dispatches in one command buffer, each writing its equal
    /// share, so that word `k` becomes `value + k`, and waits for them.
    /// `memory` has to start and end on page boundaries (4 KiB), and its
    /// words have to split evenly into `dispatches`.
    pub fn fill(
        &self,
        memory: &mut [u32],
        dispatches: u32,
        value: u32,
    ) -> Result<FillTimes, CallError> {
        let mut f: abi::Fill = abi::block(|f: &mut abi::Fill, n| f.size = n);
        f.address = memory.as_mut_ptr() as u64;
        f.length = size_of_val(memory) as u64;
        f.dispatches = dispatches;
        f.value = value;
        f.threads_per_group = 64;
        // SAFETY: a block of the call's own layout; the memory it names is
        // borrowed mutably for the call, which waits for the GPU.
        let status = unsafe { self.call(abi::FILL, (&raw mut f).cast()) };
        if status != 0 {
            return Err(failed(status, &f.error));
        }
        Ok(FillTimes {
            wrap: Duration::from_nanos(f.wrap_ns),
            encode: Duration::from_nanos(f.encode_ns),
            wait: Duration::from_nanos(f.wait_ns),
            gpu: Duration::from_nanos(f.gpu_ns),
        })
    }

    /// `count` empty command buffers, each committed and waited for alone;
    /// the time they took in all, by the unixlib's clock.
    pub fn submit_empty(&self, count: u32) -> Result<Duration, CallError> {
        let mut s: abi::Submit = abi::block(|s: &mut abi::Submit, n| s.size = n);
        s.count = count;
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::SUBMIT_EMPTY, (&raw mut s).cast()) };
        if status != 0 {
            return Err(failed(status, &s.error));
        }
        Ok(Duration::from_nanos(s.total_ns))
    }
}

fn failed(status: i32, error: &[u8]) -> CallError {
    let mut message = abi::text(error);
    if message.is_empty() && status == STATUS_INVALID_PARAMETER {
        message = "the unixlib does not take this block (built from another abi.h?)".into();
    }
    CallError { status, message }
}
