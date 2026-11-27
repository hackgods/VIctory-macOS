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

// The same sizes as abi.h's static assertions.
const _: () = assert!(size_of::<Device>() == 296);
const _: () = assert!(size_of::<Fill>() == 256);
const _: () = assert!(size_of::<Submit>() == 208);
const _: () = assert!(size_of::<Turns>() == 232);

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
        for (name, size) in [
            ("vm_device", size_of::<Device>()),
            ("vm_fill", size_of::<Fill>()),
            ("vm_submit", size_of::<Submit>()),
            ("vm_turns", size_of::<Turns>()),
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
