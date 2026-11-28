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

/// Where the time of a [`Bridge::turns`] went, by the unixlib's clock.
#[derive(Debug, Clone, Copy)]
pub struct TurnsTimes {
    /// Recording the command buffer.
    pub encode: Duration,
    /// From the commit to the end of the wait for completion.
    pub wait: Duration,
    /// What the GPU reports between the command buffer's start and end.
    pub gpu: Duration,
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

    /// `pairs` times a dispatch of one thread that writes an indirect
    /// draw's 20 bytes of arguments, then that draw into a 64 x 64 target,
    /// in one command buffer, waited for. With `batched` false each
    /// dispatch and each draw get an encoder of their own, as compute and
    /// render work taking turns do; with `true`, one compute encoder holds
    /// all dispatches and one render encoder all draws. The unixlib checks
    /// every record and the target.
    pub fn turns(&self, pairs: u32, batched: bool) -> Result<TurnsTimes, CallError> {
        let mut t: abi::Turns = abi::block(|t: &mut abi::Turns, n| t.size = n);
        t.pairs = pairs;
        t.batched = batched.into();
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::TURNS, (&raw mut t).cast()) };
        if status != 0 {
            return Err(failed(status, &t.error));
        }
        Ok(TurnsTimes {
            encode: Duration::from_nanos(t.encode_ns),
            wait: Duration::from_nanos(t.wait_ns),
            gpu: Duration::from_nanos(t.gpu_ns),
        })
    }
}

/// A no-copy buffer over memory the Windows side owns, kept by the unixlib
/// until [`Bridge::release`].
#[derive(Debug, Clone, Copy)]
pub struct Wrapped {
    /// The unixlib's handle for it.
    pub handle: u64,
    /// The buffer's GPU address (what a shader's pointer to it holds).
    pub gpu_address: u64,
    /// How long `newBufferWithBytesNoCopy` took.
    pub wrap: Duration,
}

/// Where the time of a [`Bridge::run`] went, by the unixlib's clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunTimes {
    /// Recording and committing the command buffer.
    pub encode: Duration,
    /// From the commit (or the [`Bridge::wait`] call) to the end of the
    /// wait for completion.
    pub wait: Duration,
    /// What the GPU reports between the command buffer's start and end.
    pub gpu: Duration,
}

/// One step of a [`Bridge::run`]: compute work over wrapped buffers, or an
/// event the command buffer waits for or signals. Offsets are bytes, 4-byte
/// aligned; `words` is at most 2^31.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Op(abi::Op);

impl Op {
    fn new(kind: u32) -> Self {
        Self(abi::Op {
            kind,
            first_bad: u32::MAX,
            ..Default::default()
        })
    }

    /// Word `k` of `words` words at `offset` of `buffer` becomes `value + k`.
    pub fn write(buffer: u64, offset: u64, words: u32, value: u32) -> Self {
        let mut op = Self::new(abi::OP_WRITE);
        (op.0.buffer, op.0.offset, op.0.words, op.0.value) = (buffer, offset, words, value);
        op
    }

    /// Counts the words of that range that are not `value + k`
    /// ([`Op::mismatches`], [`Op::first_bad`]).
    pub fn check(buffer: u64, offset: u64, words: u32, value: u32) -> Self {
        let mut op = Self::write(buffer, offset, words, value);
        op.0.kind = abi::OP_CHECK;
        op
    }

    /// Copies `words` words from `source` at `from` to `buffer` at `offset`.
    pub fn copy(buffer: u64, offset: u64, source: u64, from: u64, words: u32) -> Self {
        let mut op = Self::write(buffer, offset, words, 0);
        op.0.kind = abi::OP_COPY;
        (op.0.other, op.0.other_value) = (source, from);
        op
    }

    /// The GPU waits until `event` is at least `value`.
    pub fn wait_event(event: u64, value: u64) -> Self {
        let mut op = Self::new(abi::OP_WAIT_EVENT);
        (op.0.other, op.0.other_value) = (event, value);
        op
    }

    /// The GPU sets `event` to `value` once the ops before are done.
    pub fn signal_event(event: u64, value: u64) -> Self {
        let mut op = Self::new(abi::OP_SIGNAL_EVENT);
        (op.0.other, op.0.other_value) = (event, value);
        op
    }

    /// Ends the compute encoder; the next op opens a new one.
    pub fn split() -> Self {
        Self::new(abi::OP_SPLIT)
    }

    /// For a check that ran: how many words differed.
    pub fn mismatches(&self) -> u32 {
        self.0.mismatches
    }

    /// For a check that ran: the lowest word index that differed, if any.
    pub fn first_bad(&self) -> Option<u32> {
        (self.0.first_bad != u32::MAX).then_some(self.0.first_bad)
    }
}

/// A run committed and not waited for yet ([`Bridge::start`]). Its ops get
/// their results when it is waited for; dropping it waits too.
pub struct Started<'a> {
    bridge: &'a Bridge,
    handle: u64,
    encode: Duration,
    _ops: std::marker::PhantomData<&'a mut [Op]>,
}

impl Started<'_> {
    /// Waits for the GPU and writes the results back into the ops.
    pub fn wait(self) -> Result<RunTimes, CallError> {
        let encode = self.encode;
        let handle = self.handle;
        let bridge = self.bridge;
        std::mem::forget(self);
        let mut times = bridge.wait(handle)?;
        times.encode = encode;
        Ok(times)
    }
}

impl Drop for Started<'_> {
    fn drop(&mut self) {
        let _ = self.bridge.wait(self.handle);
    }
}

impl Bridge {
    /// Wraps `[address, address + length)` in one no-copy buffer, kept until
    /// [`Bridge::release`]. Both have to be on 4 KiB page boundaries, and
    /// the memory has to stay committed while the buffer lives. With
    /// `untracked` Metal does not order work on it across encoders.
    pub fn wrap(
        &self,
        address: *mut u8,
        length: usize,
        untracked: bool,
    ) -> Result<Wrapped, CallError> {
        let mut w: abi::Wrap = abi::block(|w: &mut abi::Wrap, n| w.size = n);
        w.address = address as u64;
        w.length = length as u64;
        w.flags = if untracked { abi::WRAP_UNTRACKED } else { 0 };
        // SAFETY: a block of the call's own layout, with its size set. The
        // memory is the caller's; the unixlib only records it.
        let status = unsafe { self.call(abi::WRAP, (&raw mut w).cast()) };
        if status != 0 {
            return Err(failed(status, &w.error));
        }
        Ok(Wrapped {
            handle: w.handle,
            gpu_address: w.gpu_address,
            wrap: Duration::from_nanos(w.wrap_ns),
        })
    }

    /// Lets a buffer or event handle go.
    pub fn release(&self, handle: u64) -> Result<(), CallError> {
        let mut r: abi::Release = abi::block(|r: &mut abi::Release, n| r.size = n);
        r.handle = handle;
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::RELEASE, (&raw mut r).cast()) };
        if status != 0 {
            return Err(failed(status, &r.error));
        }
        Ok(())
    }

    fn event_call(
        &self,
        op: u32,
        handle: u64,
        value: u64,
        timeout: Duration,
    ) -> Result<abi::Event, CallError> {
        let mut e: abi::Event = abi::block(|e: &mut abi::Event, n| e.size = n);
        (e.op, e.handle, e.value) = (op, handle, value);
        e.timeout_ns = timeout.as_nanos().min(u64::MAX.into()) as u64;
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::EVENT, (&raw mut e).cast()) };
        if status != 0 {
            return Err(failed(status, &e.error));
        }
        Ok(e)
    }

    /// A new shared event (`MTLSharedEvent`) at value 0.
    pub fn event(&self) -> Result<u64, CallError> {
        Ok(self
            .event_call(abi::EVENT_NEW, 0, 0, Duration::ZERO)?
            .handle)
    }

    /// Sets `event` to `value` from the CPU.
    pub fn signal(&self, event: u64, value: u64) -> Result<(), CallError> {
        self.event_call(abi::EVENT_SIGNAL, event, value, Duration::ZERO)
            .map(drop)
    }

    /// Waits on the CPU until `event` is at least `value`, at most
    /// `timeout`; how long it took.
    pub fn wait_for(
        &self,
        event: u64,
        value: u64,
        timeout: Duration,
    ) -> Result<Duration, CallError> {
        Ok(Duration::from_nanos(
            self.event_call(abi::EVENT_WAIT, event, value, timeout)?
                .waited_ns,
        ))
    }

    /// The value `event` is at now.
    pub fn value(&self, event: u64) -> Result<u64, CallError> {
        Ok(self
            .event_call(abi::EVENT_VALUE, event, 0, Duration::ZERO)?
            .value)
    }

    fn run_call(&self, ops: &mut [Op], wait: bool) -> Result<abi::Run, CallError> {
        let mut r: abi::Run = abi::block(|r: &mut abi::Run, n| r.size = n);
        r.count = u32::try_from(ops.len()).expect("ops");
        r.ops = ops.as_mut_ptr() as u64;
        r.wait = wait.into();
        // SAFETY: a block of the call's own layout; `Op` is `vm_op`
        // (transparent), and the ops stay borrowed until the run is waited
        // for (here, or through `Started`).
        let status = unsafe { self.call(abi::RUN, (&raw mut r).cast()) };
        if status != 0 {
            return Err(failed(status, &r.error));
        }
        Ok(r)
    }

    /// Records `ops` into one command buffer, commits it and waits for it;
    /// checks write their results into their ops.
    pub fn run(&self, ops: &mut [Op]) -> Result<RunTimes, CallError> {
        let r = self.run_call(ops, true)?;
        Ok(RunTimes {
            encode: Duration::from_nanos(r.encode_ns),
            wait: Duration::from_nanos(r.wait_ns),
            gpu: Duration::from_nanos(r.gpu_ns),
        })
    }

    /// As [`Bridge::run`], without waiting: the run goes on while the
    /// caller works (and, say, signals an event it waits for).
    pub fn start<'a>(&'a self, ops: &'a mut [Op]) -> Result<Started<'a>, CallError> {
        let r = self.run_call(ops, false)?;
        Ok(Started {
            bridge: self,
            handle: r.handle,
            encode: Duration::from_nanos(r.encode_ns),
            _ops: std::marker::PhantomData,
        })
    }

    fn wait(&self, run: u64) -> Result<RunTimes, CallError> {
        let mut w: abi::Wait = abi::block(|w: &mut abi::Wait, n| w.size = n);
        w.handle = run;
        // SAFETY: a block of the call's own layout, with its size set.
        let status = unsafe { self.call(abi::WAIT, (&raw mut w).cast()) };
        if status != 0 {
            return Err(failed(status, &w.error));
        }
        Ok(RunTimes {
            wait: Duration::from_nanos(w.wait_ns),
            gpu: Duration::from_nanos(w.gpu_ns),
            ..Default::default()
        })
    }
}

fn failed(status: i32, error: &[u8]) -> CallError {
    let mut message = abi::text(error);
    if message.is_empty() && status == STATUS_INVALID_PARAMETER {
        message = "the unixlib does not take this block (built from another abi.h?)".into();
    }
    CallError { status, message }
}
