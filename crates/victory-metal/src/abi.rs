//! The calls into the unixlib and their argument blocks, laid out as
//! `unix/abi.h` lays them out. Each block starts with its own size, which
//! the unixlib checks before it reads further.

/// Nothing: what a call through Wine's dispatcher costs.
pub(crate) const NOOP: u32 = 0;
/// Makes the device, a queue and the fill pipeline once; [`Device`].
pub(crate) const OPEN: u32 = 1;
/// Compute dispatches over the caller's memory, in place; [`Fill`].
pub(crate) const FILL: u32 = 2;
/// Empty command buffers, each committed and waited for; [`Submit`].
pub(crate) const SUBMIT_EMPTY: u32 = 3;
/// Compute and render work taking turns, or batched; [`Turns`].
pub(crate) const TURNS: u32 = 4;

pub(crate) const ERROR_LEN: usize = 192;

#[repr(C)]
pub(crate) struct Device {
    pub size: u32,
    pub unified_memory: u32,
    pub registry_id: u64,
    pub max_buffer_length: u64,
    pub recommended_working_set: u64,
    pub page_size: u64,
    pub name: [u8; 64],
    pub error: [u8; ERROR_LEN],
}

#[repr(C)]
pub(crate) struct Fill {
    pub size: u32,
    pub dispatches: u32,
    pub address: u64,
    pub length: u64,
    pub value: u32,
    pub threads_per_group: u32,
    pub wrap_ns: u64,
    pub encode_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub error: [u8; ERROR_LEN],
}

#[repr(C)]
pub(crate) struct Submit {
    pub size: u32,
    pub count: u32,
    pub total_ns: u64,
    pub error: [u8; ERROR_LEN],
}

#[repr(C)]
pub(crate) struct Turns {
    pub size: u32,
    pub pairs: u32,
    pub batched: u32,
    pub reserved: u32,
    pub encode_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub error: [u8; ERROR_LEN],
}

/// Wraps the caller's memory in a no-copy buffer kept under a handle; [`Wrap`].
pub(crate) const WRAP: u32 = 5;
/// Lets a handle go; [`Release`].
pub(crate) const RELEASE: u32 = 6;
/// A shared event from the CPU side; [`Event`].
pub(crate) const EVENT: u32 = 7;
/// Ops in one command buffer, committed; [`Run`], [`Op`].
pub(crate) const RUN: u32 = 8;
/// Waits for a run committed without waiting; [`Wait`].
pub(crate) const WAIT: u32 = 9;

/// [`Wrap::flags`]: the buffer's hazards are left untracked.
pub(crate) const WRAP_UNTRACKED: u32 = 1;

#[repr(C)]
pub(crate) struct Wrap {
    pub size: u32,
    pub flags: u32,
    pub address: u64,
    pub length: u64,
    pub handle: u64,
    pub gpu_address: u64,
    pub wrap_ns: u64,
    pub error: [u8; ERROR_LEN],
}

#[repr(C)]
pub(crate) struct Release {
    pub size: u32,
    pub reserved: u32,
    pub handle: u64,
    pub error: [u8; ERROR_LEN],
}

pub(crate) const EVENT_NEW: u32 = 0;
pub(crate) const EVENT_SIGNAL: u32 = 1;
pub(crate) const EVENT_WAIT: u32 = 2;
pub(crate) const EVENT_VALUE: u32 = 3;

#[repr(C)]
pub(crate) struct Event {
    pub size: u32,
    pub op: u32,
    pub handle: u64,
    pub value: u64,
    pub timeout_ns: u64,
    pub waited_ns: u64,
    pub error: [u8; ERROR_LEN],
}

pub(crate) const OP_WRITE: u32 = 0;
pub(crate) const OP_CHECK: u32 = 1;
pub(crate) const OP_WAIT_EVENT: u32 = 2;
pub(crate) const OP_SIGNAL_EVENT: u32 = 3;
pub(crate) const OP_SPLIT: u32 = 4;
pub(crate) const OP_COPY: u32 = 5;

/// One op of a run (`struct vm_op`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Op {
    pub kind: u32,
    pub words: u32,
    pub buffer: u64,
    pub offset: u64,
    pub value: u32,
    pub mismatches: u32,
    pub other: u64,
    pub other_value: u64,
    pub first_bad: u32,
    pub reserved: u32,
}

#[repr(C)]
pub(crate) struct Run {
    pub size: u32,
    pub count: u32,
    pub ops: u64,
    pub wait: u32,
    pub reserved: u32,
    pub handle: u64,
    pub encode_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub error: [u8; ERROR_LEN],
}

#[repr(C)]
pub(crate) struct Wait {
    pub size: u32,
    pub reserved: u32,
    pub handle: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub error: [u8; ERROR_LEN],
}

// The same sizes as abi.h's static assertions.
const _: () = assert!(size_of::<Device>() == 296);
const _: () = assert!(size_of::<Fill>() == 256);
const _: () = assert!(size_of::<Submit>() == 208);
const _: () = assert!(size_of::<Turns>() == 232);
const _: () = assert!(size_of::<Wrap>() == 240);
const _: () = assert!(size_of::<Release>() == 208);
const _: () = assert!(size_of::<Event>() == 232);
const _: () = assert!(size_of::<Op>() == 56);
const _: () = assert!(size_of::<Run>() == 248);
const _: () = assert!(size_of::<Wait>() == 224);

/// A zeroed block of type `T` with its size field set. Every block is
/// plain integers and bytes, so all zeroes is a valid value.
pub(crate) fn block<T: Sized>(set_size: impl FnOnce(&mut T, u32)) -> T {
    // SAFETY: the blocks above hold only integers and byte arrays.
    let mut b: T = unsafe { std::mem::zeroed() };
    set_size(&mut b, size_of::<T>() as u32);
    b
}

/// The text up to the first NUL of a fixed-size field.
pub(crate) fn text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_match_the_header() {
        // The offsets abi.h's readers depend on, beyond the sizes checked
        // at compile time.
        assert_eq!(std::mem::offset_of!(Device, name), 40);
        assert_eq!(std::mem::offset_of!(Device, error), 104);
        assert_eq!(std::mem::offset_of!(Fill, address), 8);
        assert_eq!(std::mem::offset_of!(Fill, wrap_ns), 32);
        assert_eq!(std::mem::offset_of!(Fill, error), 64);
        assert_eq!(std::mem::offset_of!(Submit, total_ns), 8);
        assert_eq!(std::mem::offset_of!(Turns, encode_ns), 16);
        assert_eq!(std::mem::offset_of!(Turns, error), 40);
        let header = include_str!("../unix/abi.h");
        assert_eq!(std::mem::offset_of!(Wrap, handle), 24);
        assert_eq!(std::mem::offset_of!(Wrap, error), 48);
        assert_eq!(std::mem::offset_of!(Event, waited_ns), 32);
        assert_eq!(std::mem::offset_of!(Op, value), 24);
        assert_eq!(std::mem::offset_of!(Op, other), 32);
        assert_eq!(std::mem::offset_of!(Op, first_bad), 48);
        assert_eq!(std::mem::offset_of!(Run, handle), 24);
        assert_eq!(std::mem::offset_of!(Run, error), 56);
        assert_eq!(std::mem::offset_of!(Wait, error), 32);
        for (name, size) in [
            ("vm_device", size_of::<Device>()),
            ("vm_fill", size_of::<Fill>()),
            ("vm_submit", size_of::<Submit>()),
            ("vm_turns", size_of::<Turns>()),
            ("vm_wrap", size_of::<Wrap>()),
            ("vm_release", size_of::<Release>()),
            ("vm_event", size_of::<Event>()),
            ("vm_op", size_of::<Op>()),
            ("vm_run", size_of::<Run>()),
            ("vm_wait", size_of::<Wait>()),
        ] {
            let assertion = format!("sizeof(struct {name}) == {size}");
            assert!(header.contains(&assertion), "abi.h lacks `{assertion}`");
        }
        for (name, code) in [
            ("VM_NOOP", NOOP),
            ("VM_OPEN", OPEN),
            ("VM_FILL", FILL),
            ("VM_SUBMIT_EMPTY", SUBMIT_EMPTY),
            ("VM_TURNS", TURNS),
            ("VM_WRAP", WRAP),
            ("VM_RELEASE", RELEASE),
            ("VM_EVENT", EVENT),
            ("VM_RUN", RUN),
            ("VM_WAIT", WAIT),
            ("VM_EVENT_NEW", EVENT_NEW),
            ("VM_EVENT_SIGNAL", EVENT_SIGNAL),
            ("VM_EVENT_WAIT", EVENT_WAIT),
            ("VM_EVENT_VALUE", EVENT_VALUE),
            ("VM_OP_WRITE", OP_WRITE),
            ("VM_OP_CHECK", OP_CHECK),
            ("VM_OP_WAIT_EVENT", OP_WAIT_EVENT),
            ("VM_OP_SIGNAL_EVENT", OP_SIGNAL_EVENT),
            ("VM_OP_SPLIT", OP_SPLIT),
            ("VM_OP_COPY", OP_COPY),
        ] {
            let line = format!("{name} = {code},");
            assert!(header.contains(&line), "abi.h lacks `{line}`");
        }
    }

    #[test]
    fn block_sets_its_size_and_text_stops_at_nul() {
        let d: Device = block(|d: &mut Device, n| d.size = n);
        assert_eq!(d.size, 296);
        assert_eq!(d.registry_id, 0);
        assert_eq!(text(b"Apple M5 Pro\0junk"), "Apple M5 Pro");
        assert_eq!(text(b"no nul"), "no nul");
    }
}
