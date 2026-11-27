//! macOS port: input switches for play at a few frames a second, all off by
//! default. A child of `gameinput`, so that file keeps its upstream shape.
//!
//! `VICTORY_INPUT_LATCH=1`: the title takes a pad's current reading once a
//! frame, 250 to 500 ms apart here, and a press and release between two of
//! them left no reading it saw (sprint is tapped, so is jump). With the latch
//! each button change the title has not seen yet waits for it: a button that
//! was up in the last reading handed out shows down in the next one if it went
//! down at any time since, and one that was down shows up if it was let go at
//! any time since (pressed again as well: down in the reading after). Triggers
//! show their highest value since the last reading; sticks the latest value.
//! Readings handed out within `LATCH_HOLD` of each other agree.

use super::{FALLBACK, PADS, PadState};

/// Devices with a latch: the host pads and the keyboard's stand-in pad.
const LATCHED: usize = FALLBACK + 1;
const _: () = assert!(FALLBACK == PADS);

/// Reads of one device this close together (µs) see the same buttons.
const LATCH_HOLD: u64 = 15_000;
/// With no reading handed out for this long (µs) the latch lets go, so that a
/// title that stops asking for a pad never finds a button stuck down.
const LATCH_MAX: u64 = 2_000_000;

#[derive(Clone, Copy, Debug, Default)]
struct Latch {
    /// Buttons the device's readings show.
    shown: u32,
    /// `shown` when the last reading was handed out: what the title saw.
    seen: u32,
    /// Since that hand-out: buttons down in some host sample, up in some,
    /// and down in a sample after one where they were up.
    down: u32,
    up: u32,
    again: u32,
    /// Pressed again after a release the title is about to see: shown down
    /// in the reading after that one.
    queued: u32,
    /// Trigger peaks since that hand-out, and what readings show.
    peak: [f32; 2],
    triggers: [f32; 2],
    /// While readings are held as handed out: since when (µs).
    holding: Option<u64>,
    /// The last hand-out (µs).
    handed: u64,
}

impl Latch {
    fn hand_out(&mut self, now: u64) {
        if self.holding.is_some() {
            return;
        }
        self.queued &= !self.shown;
        self.seen = self.shown;
        (self.down, self.up, self.again) = (0, 0, 0);
        self.peak = [0.0; 2];
        self.holding = Some(now);
        self.handed = now;
    }

    fn apply(&mut self, pad: PadState, now: u64, hold: u64) -> PadState {
        if self.holding.is_none() && now.saturating_sub(self.handed) >= LATCH_MAX {
            self.hand_out(now);
        }
        let host = pad.buttons;
        self.again |= self.up & host;
        self.down |= host;
        self.up |= !host;
        self.peak[0] = self.peak[0].max(pad.left_trigger);
        self.peak[1] = self.peak[1].max(pad.right_trigger);
        if self
            .holding
            .is_some_and(|at| now.saturating_sub(at) >= hold)
        {
            self.holding = None;
        }
        if self.holding.is_none() {
            let seen = self.seen;
            self.queued |= seen & self.up & self.again;
            self.shown = (seen & !self.up) | (!seen & (self.down | self.queued));
            self.triggers = self.peak;
        }
        PadState {
            buttons: self.shown,
            left_trigger: self.triggers[0],
            right_trigger: self.triggers[1],
            ..pad
        }
    }
}

/// The switches and what they keep; `Default` is all off.
#[derive(Debug, Default)]
pub(super) struct Mac {
    /// `Some(hold)` with the latch on.
    latch: Option<u64>,
    latches: [Latch; LATCHED],
}

impl Mac {
    /// The switches from the environment, one log line for each one on.
    pub(super) fn from_env() -> Mac {
        let on =
            |name: &str| std::env::var(name).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
        let mut mac = Mac::default();
        if on("VICTORY_INPUT_LATCH") {
            tracing::info!(
                hold_ms = LATCH_HOLD / 1000,
                "input latch: presses and releases between two readings show in the next (VICTORY_INPUT_LATCH)"
            );
            mac.latch = Some(LATCH_HOLD);
        }
        mac
    }

    /// Device `index`'s gamepad state as its next reading shows it.
    pub(super) fn latch(
        &mut self,
        index: usize,
        pad: Option<PadState>,
        now: u64,
    ) -> Option<PadState> {
        let (Some(hold), Some(latch)) = (self.latch, self.latches.get_mut(index)) else {
            return pad;
        };
        let Some(pad) = pad else {
            *latch = Latch {
                handed: now,
                ..Latch::default()
            };
            return None;
        };
        Some(latch.apply(pad, now, hold))
    }

    /// A reading of device `index` went to the title at `now`.
    pub(super) fn handed(&mut self, index: usize, now: u64) {
        if self.latch.is_some()
            && let Some(latch) = self.latches.get_mut(index)
        {
            latch.hand_out(now);
        }
    }
}

#[cfg(test)]
pub(super) fn latch_for_tests(hold: u64) -> Mac {
    Mac {
        latch: Some(hold),
        ..Mac::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u32 = 4;

    fn pad(buttons: u32, left_trigger: f32) -> Option<PadState> {
        Some(PadState {
            buttons,
            left_trigger,
            left_x: 0.25,
            ..Default::default()
        })
    }

    fn buttons(mac: &mut Mac, host: u32, now: u64) -> u32 {
        mac.latch(0, pad(host, 0.0), now).unwrap().buttons
    }

    #[test]
    fn a_second_press_before_the_title_saw_the_release_comes_after_it() {
        let mut mac = latch_for_tests(0);
        assert_eq!(buttons(&mut mac, A, 10), A);
        mac.handed(0, 11);
        // Let go, pressed again, let go, all before the next reading.
        for (host, t) in [(0, 20), (A, 30), (0, 40)] {
            assert_eq!(buttons(&mut mac, host, t), 0);
        }
        mac.handed(0, 41);
        assert_eq!(buttons(&mut mac, 0, 50), A, "the second press");
        mac.handed(0, 51);
        assert_eq!(buttons(&mut mac, 0, 60), 0);
    }

    #[test]
    fn reads_close_together_agree() {
        let mut mac = latch_for_tests(LATCH_HOLD);
        assert_eq!(buttons(&mut mac, A, 10), A);
        mac.handed(0, 1_000);
        assert_eq!(buttons(&mut mac, 0, 9_000), A, "held as handed out");
        mac.handed(0, 9_500);
        assert_eq!(buttons(&mut mac, 0, 17_000), 0, "then the release");
    }

    #[test]
    fn a_latch_nobody_reads_lets_go() {
        let mut mac = latch_for_tests(0);
        assert_eq!(buttons(&mut mac, A, 10), A);
        assert_eq!(buttons(&mut mac, 0, 20), A);
        // Never handed out: shown as the host has it after `LATCH_MAX`.
        assert_eq!(buttons(&mut mac, 0, LATCH_MAX + 1), 0);
    }

    #[test]
    fn triggers_keep_their_peak_and_sticks_their_latest() {
        let mut mac = latch_for_tests(0);
        mac.latch(1, pad(0, 0.75), 10);
        let shown = mac.latch(1, pad(0, 0.25), 20).unwrap();
        assert_eq!((shown.left_trigger, shown.left_x), (0.75, 0.25));
        mac.handed(1, 25);
        assert_eq!(mac.latch(1, pad(0, 0.25), 30).unwrap().left_trigger, 0.25);
        // A disconnect forgets the latch.
        mac.latch(1, pad(A, 0.0), 40);
        assert_eq!(mac.latch(1, None, 50), None);
        assert_eq!(mac.latch(1, pad(0, 0.0), 60).unwrap().buttons, 0);
    }
}
