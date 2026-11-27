//! A bridge from VIctory's Windows code to Metal, through a Wine unixlib:
//! the first piece of a native Metal host (VIctory-macOS #221).
//!
//! VIctory runs as a Windows program under Wine, and its GPU work goes
//! D3D12 → D3DMetal → Metal today. A Metal host skips D3DMetal and, since
//! Apple Silicon's memory is unified, can let the GPU use title memory in
//! place (`newBufferWithBytesNoCopy`). Windows code cannot call Metal
//! itself: it reaches macOS code the way Wine's own builtin DLLs do, and the
//! way D3DMetal's `d3d12.dll` does. Wine pairs a builtin PE module with a
//! Mach-O library of the same name in the `x86_64-unix` folder beside it
//! and hands the PE side that library's call table; a call through
//! `__wine_unix_call_dispatcher` then runs one entry of it on the calling
//! thread. Both sides share the process, so pointers to Windows memory are
//! good on the macOS side as they are.
//!
//! The pieces:
//!
//! - `unix/victory_metal.m`, the unixlib (Objective-C, built for x86_64 as
//!   the Wine process runs under Rosetta), and `unix/abi.h`, the calls and
//!   their argument blocks, mirrored by `abi.rs`.
//! - `unix/anchor.rs`, an empty DLL marked as a Wine builtin: the module
//!   Wine pairs the unixlib with.
//! - [`Bridge`] (Windows only): loads the anchor, gets the call table and
//!   makes the calls.
//!
//! `tools/macos/build-metal-bridge.sh` builds the anchor and the unixlib
//! into `target/metal-bridge/{x86_64-windows,x86_64-unix}`. The bridge is
//! off unless asked for: Wine has to find the pair, so `WINEDLLPATH` names
//! `target/metal-bridge` when Wine starts, and `VICTORY_METAL_BRIDGE` names
//! the anchor as the process sees it (a `Z:` path).
//! `tools/macos/test-wine.sh` does all three. Nothing else in VIctory uses
//! the bridge yet; `docs/backend.md` says where it goes.

// Only the Windows side makes calls; the layout tests run everywhere.
#[cfg_attr(not(windows), allow(dead_code))]
mod abi;
#[cfg(windows)]
mod bridge;

#[cfg(windows)]
pub use bridge::{Bridge, CallError, Device, FillTimes, OpenError, TurnsTimes};

/// The variable that names the anchor DLL, as the process sees it.
pub const BRIDGE_VAR: &str = "VICTORY_METAL_BRIDGE";
