//! macOS port: the time a frame the title's streaming request list may
//! spend issuing its recorded requests, made larger.
//!
//! Grand Theft Auto VI 1.0.2.1 (b99f876b...) plays a cutscene's recorded
//! streaming requests from a request list (state `Prestream` in the title's
//! crash log). The list's update (+0x227c000, once a frame for each list in
//! the table at +0xab442c0) issues an entry's steps in a loop that goes on
//! while steps are left and either under 500 microseconds were spent in
//! this update or the entry is not time-sliced. An entry is time-sliced
//! while its byte +0x7f is set and the title's check +0x4f862f0 says no;
//! the update writes that back to +0x7f. The 500.0 is loaded once into
//! xmm6 (+0x227c081) and compared at +0x227c28e, +0x227c4da and, for
//! entries whose byte +0x82 is 1, +0x227c5d8.
//!
//! The 500 microseconds are a fixed value, not a share of the frame: at two
//! frames a second a time-sliced list gets 1 ms a second, against 15 ms a
//! second on a console (notes/streaming-budgets.md). Whether any entry is
//! time-sliced in the opening sequence is not known; the watch's
//! `request lists` line is there to show it.
//!
//! `VICTORY_SRL_SLICE=<k>` (2 to 200) makes that one load read a larger
//! constant. The 500.0 itself is shared by hundreds of instructions and is
//! left alone: the load's displacement is pointed at one of four floats the
//! title's read-only data already holds, 5,000, 10,000, 20,000 or 50,000,
//! whichever is nearest to 500 times `k`. What it changes in the title: a
//! time-sliced list may issue requests for up to that long in a frame, so
//! such frames get longer by up to the larger slice and more requests are
//! in flight at once. Nothing else is touched. Not tried in the game yet.
//!
//! Only offsets and short byte patterns of the build are stored here
//! (docs/asset-policy.md).

/// The switch.
pub const SWITCH: &str = "VICTORY_SRL_SLICE";

/// The load: `vmovss xmm6, [rip + 0x3ede24b]`, which is the float at
/// [`STOCK`].
const SITE: u32 = 0x0227_c081;
const SITE_BYTES: [u8; 8] = [0xc5, 0xfa, 0x10, 0x35, 0x4b, 0xe2, 0xed, 0x03];
/// Where the displacement is in the site; it counts from the site's end.
const DISP_AT: usize = 4;
/// The constant the title loads, and its value in microseconds.
const STOCK: (u32, f32) = (0x0615_a2d4, 500.0);
/// Larger floats in the title's read-only data, and the values they must
/// hold.
const LARGER: [(u32, f32); 4] =
    [(0x0614_f628, 5_000.0), (0x0615_9c54, 10_000.0), (0x0615_b8e4, 20_000.0), (0x0615_a414, 50_000.0)];

/// The factor the switch's value asks for; `None` leaves the title as it is
/// (unset, 0, 1 or not a number from 2 to 200).
pub fn wanted(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok().filter(|k| (2..=200).contains(k))
}

/// The larger constant nearest to 500 times `k`; of two equally near, the
/// larger.
fn nearest(k: u32) -> (u32, f32) {
    let want = STOCK.1 * k as f32;
    LARGER.into_iter().rev().min_by(|a, b| (a.1 - want).abs().total_cmp(&(b.1 - want).abs())).unwrap()
}

/// The slice in microseconds that `k` gives.
pub fn slice_for(k: u32) -> u32 {
    nearest(k).1 as u32
}

/// The site's bytes with the load pointed at `target`, given the bytes now
/// at the site, at the stock constant and at the target. Says why if any
/// of them is not what this was written for.
fn retargeted(site: &[u8; 8], stock: [u8; 4], target: (u32, f32), at_target: [u8; 4]) -> Result<[u8; 8], &'static str> {
    if *site != SITE_BYTES {
        return Err("not the build it was written for (the load's bytes differ)");
    }
    if stock != STOCK.1.to_le_bytes() {
        return Err("not the build it was written for (the load's constant is not 500.0)");
    }
    if at_target != target.1.to_le_bytes() {
        return Err("not the build it was written for (the larger constant is not where it was read)");
    }
    let mut bytes = *site;
    let disp = target.0.wrapping_sub(SITE + SITE_BYTES.len() as u32);
    bytes[DISP_AT..].copy_from_slice(&disp.to_le_bytes());
    Ok(bytes)
}

/// Points the load of the image mapped at `base` at the constant for `k`.
/// Before any title thread runs. Gives the new slice in microseconds, or
/// says why nothing was changed.
#[cfg(windows)]
pub fn patch(base: u64, k: u32) -> Result<u32, &'static str> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let target = nearest(k);
    let site = (base + u64::from(SITE)) as *mut u8;
    let float_at = |rva: u32| (base + u64::from(rva)) as *const [u8; 4];
    // SAFETY: the image is mapped; the site is inside its code and the two
    // constants are inside its read-only data.
    let bytes = unsafe { retargeted(&*(site as *const [u8; 8]), *float_at(STOCK.0), target, *float_at(target.0))? };
    let mut old = 0;
    // SAFETY: the site is the title's code, made writable for the write,
    // before any title thread runs; only the displacement changes.
    unsafe {
        if VirtualProtect(site as *const _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("the title's code could not be made writable");
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), site, bytes.len());
        VirtualProtect(site as *const _, bytes.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    Ok(target.1 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_two_to_two_hundred() {
        assert_eq!(wanted(Some("40")), Some(40));
        assert_eq!(wanted(Some(" 2 ")), Some(2));
        assert_eq!(wanted(Some("200")), Some(200));
        for off in [None, Some(""), Some("0"), Some("1"), Some("201"), Some("-4"), Some("on")] {
            assert_eq!(wanted(off), None, "{off:?}");
        }
    }

    /// The site's displacement leads to the constant these offsets name.
    #[test]
    fn the_load_reads_the_stock_constant() {
        assert_eq!(SITE_BYTES[..DISP_AT], [0xc5, 0xfa, 0x10, 0x35], "vmovss xmm6, [rip + disp32]");
        let disp = u32::from_le_bytes(SITE_BYTES[DISP_AT..].try_into().unwrap());
        assert_eq!(SITE + 8 + disp, STOCK.0);
    }

    #[test]
    fn the_nearest_larger_constant_is_taken() {
        for (k, us) in [(2, 5_000), (10, 5_000), (14, 5_000), (15, 10_000), (20, 10_000), (29, 10_000)] {
            assert_eq!(slice_for(k), us, "{k}");
        }
        for (k, us) in [(30, 20_000), (40, 20_000), (69, 20_000), (70, 50_000), (100, 50_000), (200, 50_000)] {
            assert_eq!(slice_for(k), us, "{k}");
        }
    }

    #[test]
    fn only_the_displacement_changes() {
        for target in LARGER {
            let bytes = retargeted(&SITE_BYTES, 500f32.to_le_bytes(), target, target.1.to_le_bytes()).unwrap();
            assert_eq!(bytes[..DISP_AT], SITE_BYTES[..DISP_AT]);
            let disp = u32::from_le_bytes(bytes[DISP_AT..].try_into().unwrap());
            assert_eq!(SITE + 8 + disp, target.0);
        }
        let x40 = retargeted(&SITE_BYTES, 500f32.to_le_bytes(), LARGER[2], 20_000f32.to_le_bytes()).unwrap();
        assert_eq!(x40, [0xc5, 0xfa, 0x10, 0x35, 0x5b, 0xf8, 0xed, 0x03]);
    }

    #[test]
    fn anything_unexpected_is_refused() {
        let (stock, target) = (500f32.to_le_bytes(), LARGER[2]);
        let good = target.1.to_le_bytes();
        let mut other = SITE_BYTES;
        other[3] = 0x3d; // another register
        assert!(retargeted(&other, stock, target, good).unwrap_err().contains("load's bytes"));
        let mut moved = SITE_BYTES;
        moved[4] ^= 4; // another constant
        assert!(retargeted(&moved, stock, target, good).unwrap_err().contains("load's bytes"));
        assert!(retargeted(&SITE_BYTES, 250f32.to_le_bytes(), target, good).unwrap_err().contains("not 500.0"));
        assert!(retargeted(&SITE_BYTES, stock, target, 20_001f32.to_le_bytes()).unwrap_err().contains("larger constant"));
        assert!(retargeted(&SITE_BYTES, stock, target, [0; 4]).is_err());
    }
}
