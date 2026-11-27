//! Keeps the host GPU's clocks up while the title submits
//! (`VICTORY_CARD_AWAKE=1`, or `turns` for 20 s on and 20 s off in one run;
//! off by default).
//!
//! The host GPU stands idle most of a world frame while the CPU prepares the
//! next submission. Its driver takes the gaps for a light load and lowers the
//! clocks, and the next submission starts slow. A Metal System Trace of the
//! prison cell on an M5 Pro (15 frames): half the GPU's busy time ran below
//! its maximum performance state, 31% at medium and 19% at minimum, where the
//! same one-record argument clamp took 2.4, 2.8 and 3.9 µs. Its GPU sat idle
//! in eleven gaps of more than 10 ms a frame.
//!
//! One group of 64 threads loops on a compute queue of its own, over and
//! over, two submissions on the way at most: a few of the GPU's thread slots,
//! and the GPU is never idle. Nothing is sent once no title submission has
//! gone out for a while (a menu, a minimized window). On the same M5 Pro
//! under D3DMetal, a compute job of 4.7 ms at full clocks took a median 7.4
//! to 7.9 ms after 30 ms idle and 5.8 ms back to back; with the loop running,
//! 4.8 to 5.0 and 4.7 to 4.8 ms.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

/// About a millisecond a submission on an M5 Pro at full clocks.
const LOOP: &str = r#"
RWByteAddressBuffer sum : register(u0);
cbuffer Count : register(b0) { uint count; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < count; i++) x = x * 1664525u + 1013904223u;
    sum.Store((id.x & 63u) * 4u, x);
}
"#;

/// Iterations of [`LOOP`] a submission.
const COUNT: u32 = 200_000;

/// Quiet this long after the last title submission.
const QUIET_MS: u64 = 250;

/// With `turns`: this long on, this long off.
const TURN_MS: u64 = 20_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    /// On and off by turns, to compare the two in one run (the log says when).
    ByTurns,
}

/// The mode a `VICTORY_CARD_AWAKE` value asks for.
pub(crate) fn mode_of(value: Option<&str>) -> Mode {
    match value {
        Some("1") => Mode::On,
        Some("turns") => Mode::ByTurns,
        _ => Mode::Off,
    }
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| mode_of(std::env::var("VICTORY_CARD_AWAKE").ok().as_deref()))
}

fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// When a title submission last went out ([`now_ms`] plus one; 0: never).
static LAST: AtomicU64 = AtomicU64::new(0);

/// Submissions of the loop so far.
static LOOPS: AtomicU64 = AtomicU64::new(0);

/// Whether to keep the GPU busy at `now` (ms) in `mode`, the last title
/// submission having gone out at `last` ([`LAST`]'s encoding).
pub(crate) fn wanted_at(mode: Mode, last: u64, now: u64) -> bool {
    let recent = last != 0 && now < last - 1 + QUIET_MS;
    match mode {
        Mode::Off => false,
        Mode::On => recent,
        Mode::ByTurns => recent && (now / TURN_MS) % 2 == 1,
    }
}

fn wanted() -> bool {
    wanted_at(mode(), LAST.load(Ordering::Relaxed), now_ms())
}

/// A title submission went to the host queue of `device`: the loop starts
/// with the first, and runs as long as the process does.
pub(crate) fn sent(device: &ID3D12Device) {
    let mode = mode();
    if mode == Mode::Off {
        return;
    }
    let now = now_ms();
    let before = LAST.swap(now + 1, Ordering::Relaxed);
    if mode == Mode::ByTurns && before.saturating_sub(1) / TURN_MS != now / TURN_MS {
        tracing::info!(
            on = (now / TURN_MS) % 2 == 1,
            loops = LOOPS.load(Ordering::Relaxed),
            "the GPU is kept awake, by turns (VICTORY_CARD_AWAKE)"
        );
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        tracing::info!(
            ?mode,
            "the GPU is kept awake between submissions (VICTORY_CARD_AWAKE)"
        );
        std::mem::forget(start(device, wanted));
    });
}

/// The thread that keeps the GPU busy; it stops when this is dropped.
pub(crate) struct Awake {
    stop: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// D3D12 objects are free-threaded.
struct Device(ID3D12Device);
// SAFETY: as above.
unsafe impl Send for Device {}

/// Keeps the GPU of `device` busy while `wanted`.
pub(crate) fn start(
    device: &ID3D12Device,
    wanted: impl Fn() -> bool + Send + 'static,
) -> Option<Awake> {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let (ended, device) = (stop.clone(), Device(device.clone()));
    let thread = std::thread::Builder::new()
        .name("victory-gpu-awake".into())
        .spawn(move || {
            let device = device;
            if let Err(e) = run(&device.0, &wanted, &ended) {
                tracing::warn!(error = %e, "the GPU is not kept awake");
            }
        })
        .ok()?;
    Some(Awake {
        stop,
        thread: Some(thread),
    })
}

fn run(
    device: &ID3D12Device,
    wanted: &dyn Fn() -> bool,
    stop: &AtomicBool,
) -> windows::core::Result<()> {
    use crate::compute::Param;
    let (root, pso) = crate::compute::pipeline(
        LOOP,
        s!("awake"),
        0,
        &[
            Param::Uav { register: 0 },
            Param::Constants {
                register: 0,
                count: 1,
            },
        ],
    )?;
    // SAFETY: plain object creation and recording; one closed list, sent
    // over and over, whose objects live until what was sent last has run.
    unsafe {
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            ..Default::default()
        })?;
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
        list.SetComputeRootSignature(&root);
        list.SetComputeRootUnorderedAccessView(0, sum.GetGPUVirtualAddress());
        let count = COUNT;
        list.SetComputeRoot32BitConstants(1, 1, (&raw const count).cast(), 0);
        list.Dispatch(1, 1, 1);
        list.Close()?;
        let list: ID3D12CommandList = list.cast()?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let mut value = 0u64;
        let mut result = Ok(());
        while !stop.load(Ordering::Relaxed) && result.is_ok() {
            if !wanted() {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            // Never more than two on the way.
            if fence.GetCompletedValue() + 2 <= value {
                std::thread::sleep(Duration::from_micros(250));
                continue;
            }
            queue.ExecuteCommandLists(&[Some(list.clone())]);
            value += 1;
            LOOPS.fetch_add(1, Ordering::Relaxed);
            result = queue.Signal(&fence, value);
        }
        while result.is_ok() && fence.GetCompletedValue() < value {
            std::thread::sleep(Duration::from_millis(1));
            result = device.GetDeviceRemovedReason();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{LOOPS, Mode, QUIET_MS, TURN_MS, mode_of, wanted_at};
    use std::sync::atomic::Ordering;

    #[test]
    fn only_one_and_turns_switch_it_on() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("0")), Mode::Off);
        assert_eq!(mode_of(Some("")), Mode::Off);
        assert_eq!(mode_of(Some("1")), Mode::On);
        assert_eq!(mode_of(Some("turns")), Mode::ByTurns);
    }

    #[test]
    fn the_loop_runs_only_while_the_title_submits() {
        // Never submitted: off in every mode.
        for mode in [Mode::Off, Mode::On, Mode::ByTurns] {
            assert!(!wanted_at(mode, 0, 10));
        }
        // Last submission at 1000 ms (stored as 1001).
        let last = 1001;
        assert!(wanted_at(Mode::On, last, 1000));
        assert!(wanted_at(Mode::On, last, 1000 + QUIET_MS - 1));
        assert!(!wanted_at(Mode::On, last, 1000 + QUIET_MS));
        assert!(!wanted_at(Mode::Off, last, 1000));
    }

    #[test]
    fn turns_are_off_then_on() {
        let at = |now: u64| wanted_at(Mode::ByTurns, now + 1, now);
        assert!(!at(TURN_MS / 2));
        assert!(at(TURN_MS + TURN_MS / 2));
        assert!(!at(2 * TURN_MS + 1));
    }

    /// The loop submits to the host GPU while wanted, stops submitting when
    /// not, and its thread ends when dropped.
    #[test]
    fn the_loop_submits_while_wanted_and_stops() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let on = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = on.clone();
        let before = LOOPS.load(Ordering::Relaxed);
        let awake = super::start(&host.device, move || flag.load(Ordering::Relaxed)).unwrap();
        let sent = || LOOPS.load(Ordering::Relaxed) - before;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while sent() < 4 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(sent() >= 4, "the loop sent {} submissions", sent());
        on.store(false, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let stopped = sent();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(sent(), stopped, "no submissions while not wanted");
        drop(awake);
        assert_eq!(host.log_messages(), None);
    }
}
