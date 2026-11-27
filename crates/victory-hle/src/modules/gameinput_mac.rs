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
//!
//! `VICTORY_STICK_DEADZONE=<share>` (for example 0.15): host pad sticks within
//! that share of full travel of the centre read as centred, and the rest of
//! the travel is spread over 0 to 1 again (a round dead zone per stick).
//! XInput values are raw: a stick at rest reads up to several per cent off
//! centre, more on a worn pad, and every wobble of it was a new reading. The
//! title has a dead zone of its own; this one comes before it.
//!
//! `VICTORY_INPUT_REFOCUS=1`: keys already down when the title's window gets
//! the focus back are left out until they are seen up. Wine's macOS driver
//! posts a key's release only while our app is active (`cocoa_app.m`,
//! `NSEventTypeKeyUp` goes through only for keys it saw go down, and only to
//! our windows), so a key let go in another app, Cmd among them after
//! Cmd-Tab, stays down in `GetAsyncKeyState` until it is pressed again.
//!
//! `VICTORY_INPUT_WATCH=1`: once a second, while a host pad is off rest or a
//! key is down, one log line (`input watch`) with what the host reports: each
//! such pad's sticks, triggers and buttons, how long its value has been
//! exactly the same and how long XInput's packet number has (a pad whose
//! reports stopped keeps both still), and the keys down. Changes nothing.
//!
//! `VICTORY_PAD_REARM=1`: a watchdog for a host pad whose reading stopped
//! while off rest (27 November: right trigger at 1.00 and packet 78 for 56 s
//! while the player pressed other things). Wine's XInput (`xinput1_3/main.c`)
//! keeps a pad's last state and its "connected" until its read thread
//! (`wine_xinput_hid_update`) takes the next HID report; if that thread stops
//! issuing reads (a read that ended in an error it does not expect, or was
//! cancelled because the thread that issued it ended) or stops altogether (a
//! failed wait ends it), the state never changes again. Three steps:
//! - A second XInput module (`xinput1_3.dll`, its own handles and read thread
//!   in Wine) reads the same pads all along. Its packet number moving while
//!   ours stays still means our read is behind: the slot shows its reading.
//! - A pad off rest with the same packet number for `REARM_STILL`, or behind
//!   the second module, gets its read re-armed: `XInputEnable(FALSE)` then
//!   `TRUE`, which cancels each pad's read and issues a new one and keeps the
//!   state, so a trigger really held keeps reading held. It runs on a thread
//!   of its own that never ends (Wine cancels a thread's reads when it ends),
//!   and XInput is not asked while it runs (it holds XInput's pad locks).
//! - No new packet `REARM_WAIT` after the re-arm, and the second module finds
//!   no pad in that slot (or the re-arm never returned: Wine's input stack
//!   does not answer): the slot reads at rest until a reading moves again.
//!
//! Reports that stop below XInput (Wine's HID bus, SDL, the pad's radio)
//! reach neither module, look like a pad held still, and are kept. One
//! `pad frozen:` line each time a step fires.
//!
//! `VICTORY_PAD_RUMBLE=0`: the title's rumble is not sent to XInput. On the
//! SDL path (read in SDL 2.32.10) a rumble report goes out on SDL's rumble
//! thread holding the pad's device lock (`SDL_hidapi_rumble.c`), and SDL's
//! update skips a pad whose lock is taken (`SDL_hidapijoystick.c:1347`): a
//! write that blocks (`IOHIDDeviceSetReport` has no timeout, `hidapi/mac`)
//! stops the pad's input with no disconnect. Wine turns PS4 rumble over
//! Bluetooth on (`bus_sdl.c`, "CW HACK 19629"). Without our rumble SDL still
//! writes at open and, in its enhanced mode, a keep-alive after 0.5 s without
//! a report; `SDL_JOYSTICK_HIDAPI_PS4_RUMBLE=0` in the environment beats
//! Wine's hint (`SDL_hints.c:58`) and keeps a pad switched on afresh out of
//! that mode (a pad already in it stays there until switched off).

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

/// Keys held across a focus change; see the module notes.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
pub(super) struct Refocus {
    /// `None` until the environment was read.
    on: Option<bool>,
    focused: bool,
    /// Down when the focus came back, and not seen up since.
    ignored: Vec<u8>,
}

#[cfg(any(windows, test))]
impl Refocus {
    /// Takes out of `held` (virtual keys the host reports down while
    /// `focused`, none without it) the keys held across a focus change.
    pub(super) fn filter(&mut self, focused: bool, held: &mut Vec<u8>) {
        let on = *self.on.get_or_insert_with(|| {
            let on = std::env::var("VICTORY_INPUT_REFOCUS")
                .is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
            if on {
                tracing::info!(
                    "input refocus: keys down when the focus comes back wait for their release (VICTORY_INPUT_REFOCUS)"
                );
            }
            on
        });
        if !on {
            return;
        }
        if focused && !self.focused && !held.is_empty() {
            tracing::info!(keys = ?held, "input refocus: keys left out until released");
            self.ignored = held.clone();
        }
        self.focused = focused;
        self.ignored.retain(|k| held.contains(k));
        held.retain(|k| !self.ignored.contains(k));
    }
}

/// What `VICTORY_INPUT_WATCH` keeps of one host pad.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug)]
struct Watched {
    pad: PadState,
    packet: u32,
    /// When the value and the packet number last changed (µs).
    changed: u64,
    packet_changed: u64,
}

/// The input watch; see the module notes.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
pub(super) struct Watch {
    /// `None` until the environment was read.
    on: Option<bool>,
    pads: [Option<Watched>; PADS],
    /// The last line (µs).
    logged: u64,
}

#[cfg(any(windows, test))]
impl Watch {
    fn on(&mut self) -> bool {
        *self.on.get_or_insert_with(|| {
            let on = std::env::var("VICTORY_INPUT_WATCH")
                .is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
            if on {
                tracing::info!(
                    "input watch: a line a second while input is off rest (VICTORY_INPUT_WATCH)"
                );
            }
            on
        })
    }

    /// Host slot `slot` reported `state` (with XInput's packet number).
    pub(super) fn pad(&mut self, slot: usize, state: Option<(PadState, u32)>, now: u64) {
        if !self.on() {
            return;
        }
        let Some(watched) = self.pads.get_mut(slot) else {
            return;
        };
        *watched = state.map(|(pad, packet)| match *watched {
            Some(w) => Watched {
                pad,
                packet,
                changed: if w.pad == pad { w.changed } else { now },
                packet_changed: if w.packet == packet {
                    w.packet_changed
                } else {
                    now
                },
            },
            None => Watched {
                pad,
                packet,
                changed: now,
                packet_changed: now,
            },
        });
    }

    /// The keys down (virtual keys) after the pads of one poll; the line
    /// logged, if one is due.
    pub(super) fn keys(&mut self, keys: &[u8], now: u64) -> Option<String> {
        use std::fmt::Write;
        if !self.on() || now.saturating_sub(self.logged) < 1_000_000 {
            return None;
        }
        let off_rest = |p: &PadState| {
            p.buttons != 0
                || [
                    p.left_trigger,
                    p.right_trigger,
                    p.left_x,
                    p.left_y,
                    p.right_x,
                    p.right_y,
                ]
                .iter()
                .any(|v| v.abs() > 0.1)
        };
        let secs = |since: u64| now.saturating_sub(since) as f64 / 1e6;
        let mut line = String::new();
        for (slot, w) in self.pads.iter().enumerate() {
            let Some(w) = w.filter(|w| off_rest(&w.pad)) else {
                continue;
            };
            let p = w.pad;
            let _ = write!(
                line,
                "pad {slot}: left {:+.2} {:+.2} right {:+.2} {:+.2} triggers {:.2} {:.2} buttons {:#x}, same for {:.1} s, packet {} same for {:.1} s; ",
                p.left_x,
                p.left_y,
                p.right_x,
                p.right_y,
                p.left_trigger,
                p.right_trigger,
                p.buttons,
                secs(w.changed),
                w.packet,
                secs(w.packet_changed),
            );
        }
        if !keys.is_empty() {
            let _ = write!(line, "keys {keys:02x?}");
        }
        if line.is_empty() {
            return None;
        }
        self.logged = now;
        tracing::info!("input watch: {line}");
        Some(line)
    }
}

/// `VICTORY_PAD_REARM`: a pad off rest whose packet number stayed the same
/// this long (µs) gets its read re-armed. XInput's packet number moves with
/// each HID report, and Wine's SDL bus sends one only when a value changes: a
/// thumb on a stick changes it many times a second, but a trigger held down
/// reads 255 and a button 1 in each report, so a pad held still can keep one
/// packet for seconds (4 s with the trigger held, 27 November 16:58:23). The
/// re-arm keeps the reading, so this only sets how soon a still pad is tried.
const REARM_STILL: u64 = 3_000_000;
/// After a re-arm, this long (µs) without a new packet before the second
/// module decides.
const REARM_WAIT: u64 = 2_000_000;
/// The second module's packet moved this long ago (µs) and ours has not
/// since: our read is behind. Both modules get each report within
/// milliseconds of each other.
const REARM_BEHIND: u64 = 500_000;
/// A re-arm still running after this long (µs) never returns. It cancels and
/// reissues up to four reads, a few milliseconds.
const REARM_HUNG: u64 = 2_000_000;

/// What the second XInput module says about a slot.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Probe {
    /// No second module (not loaded, or not asked).
    Off,
    /// Its `XInputGetState` failed (1167: no pad in that slot).
    Error(u32),
    /// Its reading and packet number.
    Reading(PadState, u32),
}

#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Stage {
    /// Our reading.
    #[default]
    Watching,
    /// Still while off rest, re-armed: waiting for a new packet.
    Rearmed,
    /// Nothing shows the pad gone: our reading until its packet moves.
    Kept,
    /// The second module's reading: ours is behind.
    Second,
    /// At rest: the pad is gone, or Wine's input stack does not answer.
    Rest,
}

#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, Default)]
struct RearmSlot {
    /// Our module's last reading, and when its packet number moved (µs).
    main: Option<(PadState, u32)>,
    moved: u64,
    /// The second module's last reading, and when its packet number moved
    /// (not counting the first one seen).
    second: Option<(PadState, u32)>,
    second_moved: Option<u64>,
    /// The second module's packet number when ours last moved: the same
    /// report reaches the two a poll apart at times.
    second_at_move: Option<u32>,
    /// The second module read this slot since our pad connected; the error it
    /// gives now.
    second_seen: bool,
    second_error: Option<u32>,
    stage: Stage,
    /// When this still spell's re-arm was asked (one per spell).
    rearmed: Option<u64>,
    /// Rumble last set: when (µs), low and high speed.
    rumble: Option<(u64, u16, u16)>,
}

/// The `VICTORY_PAD_REARM` decisions, apart from XInput; see the module notes.
#[cfg(any(windows, test))]
#[derive(Debug, Default)]
pub(super) struct Rearm {
    slots: [RearmSlot; PADS],
    /// Re-arms asked of the helper thread, and when the last one was (µs).
    asked: u64,
    asked_at: u64,
    /// A slot wants a re-arm.
    wanted: bool,
    /// A re-arm never returned: our module is not asked again.
    hung: bool,
}

/// A pad is off rest: a button, a trigger, or a stick past drift.
#[cfg(any(windows, test))]
fn off_rest(p: &PadState) -> bool {
    p.buttons != 0
        || p.left_trigger.max(p.right_trigger) > 0.1
        || p.left_x.hypot(p.left_y).max(p.right_x.hypot(p.right_y)) > 0.25
}

#[cfg(any(windows, test))]
fn describe(p: &PadState) -> String {
    format!(
        "left {:+.2} {:+.2} right {:+.2} {:+.2} triggers {:.2} {:.2} buttons {:#x}",
        p.left_x, p.left_y, p.right_x, p.right_y, p.left_trigger, p.right_trigger, p.buttons
    )
}

#[cfg(any(windows, test))]
fn secs(us: u64) -> f64 {
    us as f64 / 1e6
}

#[cfg(any(windows, test))]
impl Rearm {
    /// Whether our module may be asked about pads: not while a re-arm runs
    /// (`finished` of `asked` done), and never again after one did not return.
    pub(super) fn may_ask(&mut self, finished: u64, now: u64) -> bool {
        if !self.hung && finished < self.asked && now.saturating_sub(self.asked_at) >= REARM_HUNG {
            self.hung = true;
            tracing::warn!(
                "pad frozen: the re-arm (XInputEnable off and on) has not returned after {:.1} s, so Wine's input stack does not answer; XInput is not asked again, pads read through the second module or at rest",
                secs(now - self.asked_at)
            );
        }
        !self.hung && finished >= self.asked
    }

    /// A re-arm wanted since the last call, counted as asked at `now`.
    pub(super) fn take_wanted(&mut self, now: u64) -> bool {
        if !std::mem::take(&mut self.wanted) || self.hung {
            return false;
        }
        self.asked += 1;
        self.asked_at = now;
        true
    }

    /// The title set slot `slot`'s rumble (for the log lines).
    pub(super) fn rumble(&mut self, slot: usize, low: u16, high: u16, now: u64) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.rumble = Some((now, low, high));
        }
    }

    /// One poll of host slot `slot`. `main`: our module's answer when it was
    /// asked (`Some(None)`: no pad), `None` when it was not; `second`: the
    /// second module's. What the slot reports.
    pub(super) fn slot(
        &mut self,
        slot: usize,
        main: Option<Option<(PadState, u32)>>,
        second: Probe,
        now: u64,
    ) -> Option<PadState> {
        let hung = self.hung;
        let s = self.slots.get_mut(slot)?;
        match main {
            Some(None) => {
                *s = RearmSlot {
                    rumble: s.rumble,
                    ..RearmSlot::default()
                };
                return None;
            }
            Some(Some((pad, packet))) => {
                match s.main {
                    Some((_, old)) if old == packet => {}
                    Some((_, old)) => {
                        if s.stage != Stage::Watching || s.rearmed.is_some() {
                            let after = s.rearmed.map_or(String::new(), |t| {
                                format!(", {:.1} s after the re-arm", secs(now - t))
                            });
                            tracing::info!(
                                "pad frozen: pad {slot} reads again (packet {old} -> {packet}{after})"
                            );
                        }
                        s.stage = Stage::Watching;
                        s.rearmed = None;
                        s.moved = now;
                        s.second_at_move = s.second.map(|(_, k)| k);
                    }
                    None => {
                        s.moved = now;
                        s.second_at_move = s.second.map(|(_, k)| k);
                    }
                }
                s.main = Some((pad, packet));
            }
            None => {}
        }
        let (pad, packet) = s.main?;
        match second {
            Probe::Reading(p, k) => {
                if s.second.is_some_and(|(_, old)| old != k) {
                    s.second_moved = Some(now);
                }
                s.second = Some((p, k));
                s.second_at_move.get_or_insert(k);
                s.second_seen = true;
                s.second_error = None;
            }
            Probe::Error(e) => {
                s.second = None;
                s.second_at_move = None;
                s.second_error = Some(e);
            }
            Probe::Off => {}
        }
        // Decisions need a fresh answer from our module, or know it is dead.
        if main.is_some() || hung {
            let gone = s.second_error.filter(|_| s.second_seen);
            // Newer input in the second module: it moved after ours, a while
            // ago, to another reading or by two reports or more.
            let ahead = s.second.is_some_and(|(p, k)| {
                s.second_moved
                    .is_some_and(|t| t > s.moved && now.saturating_sub(t) >= REARM_BEHIND)
                    && (p != pad || s.second_at_move.is_some_and(|m| k.wrapping_sub(m) >= 2))
            });
            let waited = s
                .rearmed
                .is_some_and(|t| now.saturating_sub(t) >= REARM_WAIT);
            let still = secs(now.saturating_sub(s.moved));
            let next = match s.stage {
                Stage::Watching | Stage::Rearmed | Stage::Kept | Stage::Rest if ahead => {
                    let k = s.second.map_or(0, |(_, k)| k);
                    tracing::info!(
                        "pad frozen: pad {slot} packet {packet} unchanged for {still:.1} s while the second XInput module read newer input (its packet {k}); its reading is shown until ours moves"
                    );
                    Some(Stage::Second)
                }
                Stage::Kept | Stage::Second if gone.is_some() => {
                    tracing::warn!(
                        "pad frozen: pad {slot} still packet {packet}, and the second XInput module finds no pad there (error {}); reported at rest until a reading moves",
                        gone.unwrap_or_default()
                    );
                    Some(Stage::Rest)
                }
                Stage::Watching | Stage::Rearmed | Stage::Kept if hung => {
                    tracing::warn!(
                        "pad frozen: pad {slot} at packet {packet}: XInput does not answer; reported at rest until the second XInput module reads newer input"
                    );
                    Some(Stage::Rest)
                }
                Stage::Watching if off_rest(&pad) && still >= secs(REARM_STILL) => {
                    let rumble = match s.rumble {
                        Some((at, low, high)) => format!(
                            "rumble {:.2} {:.2} set {:.1} s ago",
                            f32::from(low) / 65535.0,
                            f32::from(high) / 65535.0,
                            secs(now.saturating_sub(at))
                        ),
                        None => "no rumble set".into(),
                    };
                    tracing::info!(
                        "pad frozen: pad {slot} packet {packet} unchanged for {still:.1} s while off rest ({}; {rumble}); re-arming the read (XInputEnable off and on)",
                        describe(&pad)
                    );
                    Some(Stage::Rearmed)
                }
                Stage::Rearmed if waited => {
                    if let Some(e) = gone {
                        tracing::warn!(
                            "pad frozen: pad {slot} still packet {packet} {:.1} s after the re-arm, and the second XInput module finds no pad there (error {e}); reported at rest until a reading moves",
                            secs(REARM_WAIT)
                        );
                        Some(Stage::Rest)
                    } else {
                        let second = match (s.second, s.second_error) {
                            (Some((p, k)), _) if p == pad => {
                                format!("the second XInput module reads the same (packet {k})")
                            }
                            (Some((_, k)), _) => format!(
                                "the second XInput module reads otherwise but nothing newer (packet {k})"
                            ),
                            (None, Some(e)) => {
                                format!("the second XInput module never read this slot (error {e})")
                            }
                            (None, None) => "there is no second XInput module".into(),
                        };
                        tracing::info!(
                            "pad frozen: pad {slot} still packet {packet} {:.1} s after the re-arm; {second}: kept as it reads (held still, or reports stopped below XInput)",
                            secs(REARM_WAIT)
                        );
                        Some(Stage::Kept)
                    }
                }
                _ => None,
            };
            if let Some(next) = next {
                if matches!(next, Stage::Rearmed | Stage::Second) && s.rearmed.is_none() && !hung {
                    s.rearmed = Some(now);
                    self.wanted = true;
                }
                s.stage = next;
            }
        }
        match s.stage {
            Stage::Second => s.second.map(|(p, _)| p).or(Some(PadState::default())),
            Stage::Rest => Some(PadState::default()),
            _ => Some(pad),
        }
    }
}

/// `VICTORY_PAD_RUMBLE` (its value): `0` or `off` drops the title's rumble.
#[cfg(any(windows, test))]
fn rumble_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| matches!(v.trim(), "0" | "off"))
}

/// Whether the title's rumble is dropped (`VICTORY_PAD_RUMBLE=0`); see the
/// module notes.
#[cfg(windows)]
pub(super) fn rumble_dropped() -> bool {
    static DROPPED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DROPPED.get_or_init(|| {
        let dropped = rumble_off(std::env::var("VICTORY_PAD_RUMBLE").ok().as_deref());
        if dropped {
            tracing::info!(
                "pad rumble: the title's rumble is not sent to XInput (VICTORY_PAD_RUMBLE=0)"
            );
        }
        dropped
    })
}

/// `VICTORY_PAD_REARM` on the host: the switch, the decisions, and the helper
/// thread that re-arms XInput and holds the second module.
#[cfg(windows)]
#[derive(Default)]
pub(super) struct PadRearm {
    /// `None` until the environment was read.
    on: Option<bool>,
    rearm: Rearm,
    helper: Option<helper::Helper>,
}

#[cfg(windows)]
impl PadRearm {
    fn on(&mut self) -> bool {
        if let Some(on) = self.on {
            return on;
        }
        let mut on =
            std::env::var("VICTORY_PAD_REARM").is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off"));
        if on {
            match helper::Helper::start() {
                Ok(helper) => {
                    tracing::info!(
                        still_s = secs(REARM_STILL),
                        wait_s = secs(REARM_WAIT),
                        "pad rearm: a host pad off rest with the same packet number re-arms its read; a pad still frozen that the second XInput module finds gone reads at rest (VICTORY_PAD_REARM)"
                    );
                    self.helper = Some(helper);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "pad rearm: no helper thread; VICTORY_PAD_REARM is off");
                    on = false;
                }
            }
        }
        self.on = Some(on);
        on
    }

    /// Whether XInput may be asked about pads, or set their rumble, now.
    pub(super) fn may_ask(&mut self, now: u64) -> bool {
        if !self.on() {
            return true;
        }
        let finished = self.helper.as_ref().map_or(0, helper::Helper::finished);
        self.rearm.may_ask(finished, now)
    }

    /// The title set slot `slot`'s rumble; whether XInput may be told now.
    pub(super) fn rumble(&mut self, slot: usize, low: u16, high: u16, now: u64) -> bool {
        if self.on() {
            self.rearm.rumble(slot, low, high, now);
        }
        self.may_ask(now)
    }

    /// What host slot `slot` reports, from XInput's answer (`Some(None)`: no
    /// pad; `None`: not asked).
    pub(super) fn pad(
        &mut self,
        slot: usize,
        main: Option<Option<(PadState, u32)>>,
        now: u64,
    ) -> Option<PadState> {
        if !self.on() {
            return main.flatten().map(|(pad, _)| pad);
        }
        let second = match self.helper.as_ref().and_then(helper::Helper::second) {
            Some(get) if main != Some(None) => {
                let mut state =
                    windows_sys::Win32::UI::Input::XboxController::XINPUT_STATE::default();
                // SAFETY: the second module's `XInputGetState`, room for the state.
                match unsafe { get(slot as u32, &mut state) } {
                    0 => Probe::Reading(super::host::convert(&state.Gamepad), state.dwPacketNumber),
                    e => Probe::Error(e),
                }
            }
            _ => Probe::Off,
        };
        let shown = self.rearm.slot(slot, main, second, now);
        if self.rearm.take_wanted(now)
            && let Some(helper) = &self.helper
        {
            helper.rearm();
        }
        shown
    }
}

/// The thread that re-arms XInput's reads and holds the second module.
#[cfg(windows)]
mod helper {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock, mpsc};
    use std::time::Instant;

    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    use windows_sys::Win32::UI::Input::XboxController::{XINPUT_STATE, XInputEnable};

    pub(super) type GetState = unsafe extern "system" fn(u32, *mut XINPUT_STATE) -> u32;

    /// Modules with an XInput of their own in Wine (`PARENTSRC` of
    /// `xinput1_3`); ours is `xinput1_4.dll`.
    const SECOND: [&str; 3] = ["xinput1_3.dll", "xinput1_2.dll", "xinput1_1.dll"];

    pub(super) struct Helper {
        requests: mpsc::Sender<()>,
        finished: Arc<AtomicU64>,
        second: Arc<OnceLock<Option<GetState>>>,
    }

    /// The second module's `XInputGetState`; its first call starts its read
    /// thread, which opens the pads there.
    fn load_second() -> Option<GetState> {
        for name in SECOND {
            let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
            // SAFETY: a nul-terminated name; the module stays loaded.
            let module = unsafe { LoadLibraryW(wide.as_ptr()) };
            if module.is_null() {
                continue;
            }
            // SAFETY: a loaded module and a nul-terminated name.
            let Some(proc) = (unsafe { GetProcAddress(module, c"XInputGetState".as_ptr().cast()) })
            else {
                continue;
            };
            // SAFETY: `XInputGetState`'s documented signature.
            let get: GetState = unsafe { std::mem::transmute(proc) };
            let slots: Vec<u32> = (0..4)
                .filter(|&slot| {
                    let mut state = XINPUT_STATE::default();
                    // SAFETY: room for the state.
                    unsafe { get(slot, &mut state) == 0 }
                })
                .collect();
            tracing::info!(module = name, pads = ?slots, "pad rearm: second XInput module ready");
            return Some(get);
        }
        tracing::warn!(
            "pad rearm: no second XInput module; frozen pads are re-armed but never put at rest"
        );
        None
    }

    impl Helper {
        pub(super) fn start() -> std::io::Result<Helper> {
            let (requests, asked) = mpsc::channel::<()>();
            let finished = Arc::new(AtomicU64::new(0));
            let second = Arc::new(OnceLock::new());
            let (done, module) = (finished.clone(), second.clone());
            std::thread::Builder::new()
                .name("pad rearm".into())
                .spawn(move || {
                    let _ = module.set(load_second());
                    // `XInputEnable(TRUE)` issues each pad's next read from
                    // this thread, and Wine cancels the reads of a thread
                    // that ends: this one never does.
                    loop {
                        if asked.recv().is_err() {
                            loop {
                                std::thread::park();
                            }
                        }
                        let start = Instant::now();
                        // SAFETY: plain calls.
                        unsafe {
                            XInputEnable(0);
                            XInputEnable(1);
                        }
                        done.fetch_add(1, Ordering::Release);
                        tracing::info!(
                            ms = start.elapsed().as_secs_f64() * 1e3,
                            "pad frozen: reads re-armed (XInputEnable off and on)"
                        );
                    }
                })?;
            Ok(Helper {
                requests,
                finished,
                second,
            })
        }

        pub(super) fn finished(&self) -> u64 {
            self.finished.load(Ordering::Acquire)
        }

        pub(super) fn second(&self) -> Option<GetState> {
            self.second.get().copied().flatten()
        }

        pub(super) fn rearm(&self) {
            let _ = self.requests.send(());
        }
    }
}

/// The switches and what they keep; `Default` is all off.
#[derive(Debug, Default)]
pub(super) struct Mac {
    /// `Some(hold)` with the latch on.
    latch: Option<u64>,
    latches: [Latch; LATCHED],
    /// Share of stick travel read as centred; 0 is off.
    dead_zone: f32,
}

/// A round dead zone of `zone` on one stick.
fn dead_zone(x: f32, y: f32, zone: f32) -> (f32, f32) {
    let length = x.hypot(y);
    if length <= zone {
        return (0.0, 0.0);
    }
    let scale = ((length - zone) / (1.0 - zone)).min(1.0) / length;
    (x * scale, y * scale)
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
        let zone = std::env::var("VICTORY_STICK_DEADZONE").ok();
        match zone.as_deref().map(str::trim).filter(|z| !z.is_empty()) {
            None | Some("0") | Some("off") => {}
            Some(z) => match z.parse::<f32>() {
                Ok(share) if (0.0..0.9).contains(&share) => {
                    tracing::info!(share, "stick dead zone (VICTORY_STICK_DEADZONE)");
                    mac.dead_zone = share;
                }
                _ => tracing::warn!(
                    value = z,
                    "VICTORY_STICK_DEADZONE wants a share from 0 to 0.9"
                ),
            },
        }
        mac
    }

    /// Host pads' sticks through the dead zone.
    pub(super) fn dead_zones(&self, pads: &mut [Option<PadState>; PADS]) {
        if self.dead_zone <= 0.0 {
            return;
        }
        for pad in pads.iter_mut().flatten() {
            (pad.left_x, pad.left_y) = dead_zone(pad.left_x, pad.left_y, self.dead_zone);
            (pad.right_x, pad.right_y) = dead_zone(pad.right_x, pad.right_y, self.dead_zone);
        }
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
    fn keys_held_across_a_focus_change_wait_for_their_release() {
        let mut refocus = Refocus {
            on: Some(true),
            ..Refocus::default()
        };
        let mut keys = |focused, mut held: Vec<u8>| {
            refocus.filter(focused, &mut held);
            held
        };
        assert_eq!(keys(true, vec![]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W']), [b'W']);
        // Away: nothing; back with Wine still holding W (its release went
        // to another app) and Cmd (0x5B).
        assert_eq!(keys(false, vec![]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B, b'A']), [b'A']);
        // W seen up once: pressed again, it counts.
        assert_eq!(keys(true, vec![0x5B]), [0u8; 0]);
        assert_eq!(keys(true, vec![b'W', 0x5B]), [b'W']);
        // Off: unchanged.
        let mut off = Refocus {
            on: Some(false),
            ..Refocus::default()
        };
        let mut held = vec![b'W'];
        off.filter(true, &mut held);
        assert_eq!(held, [b'W']);
    }

    #[test]
    fn the_watch_says_how_long_a_pad_has_been_still() {
        let mut watch = Watch {
            on: Some(true),
            ..Watch::default()
        };
        let held = pad(0, 0.0).unwrap();
        watch.pad(0, Some((PadState::default(), 7)), 1_000_000);
        assert_eq!(watch.keys(&[], 1_000_000), None, "at rest: nothing");
        for t in [2_000_000, 3_000_000] {
            watch.pad(0, Some((held, 8)), t);
        }
        watch.pad(0, Some((held, 8)), 4_500_000);
        let line = watch.keys(&[0x57], 4_500_000).unwrap();
        assert!(line.contains("pad 0: left +0.25 +0.00"), "{line}");
        assert!(
            line.contains("same for 2.5 s, packet 8 same for 2.5 s"),
            "{line}"
        );
        assert!(line.ends_with("keys [57]"), "{line}");
        // At most a line a second.
        assert_eq!(watch.keys(&[0x57], 5_000_000), None);
    }

    #[test]
    fn sticks_near_the_centre_read_centred() {
        let mut pads = [pad(0, 0.0), None, None, None];
        pads[0].as_mut().unwrap().right_y = -1.0;
        let mac = Mac {
            dead_zone: 0.25,
            ..Mac::default()
        };
        mac.dead_zones(&mut pads);
        let p = pads[0].unwrap();
        // 0.25 from centre is inside; full travel stays full.
        assert_eq!(
            (p.left_x, p.left_y, p.right_x, p.right_y),
            (0.0, 0.0, 0.0, -1.0)
        );
        let (x, y) = dead_zone(0.0, 0.625, 0.25);
        assert!(x == 0.0 && (y - 0.5).abs() < 1e-6);
        let (x, y) = dead_zone(0.8, 0.8, 0.25);
        assert!(
            (x - y).abs() < 1e-6 && (x.hypot(y) - 1.0).abs() < 1e-6,
            "direction kept"
        );
        // Off by default.
        let mut pads = [pad(0, 0.0), None, None, None];
        Mac::default().dead_zones(&mut pads);
        assert_eq!(pads[0].unwrap().left_x, 0.25);
    }

    const S: u64 = 1_000_000;

    fn held() -> PadState {
        PadState {
            right_trigger: 1.0,
            left_x: -0.04,
            ..Default::default()
        }
    }

    /// Polls slot 0 every 8 ms from `from` to `to` (µs) with our module at
    /// `main` and the second at `second`, re-arms finishing at once; the
    /// re-arms asked and the last reading shown.
    fn run(
        rearm: &mut Rearm,
        from: u64,
        to: u64,
        main: Option<(PadState, u32)>,
        second: Probe,
    ) -> (u32, Option<PadState>) {
        let (mut asked, mut shown) = (0, None);
        let mut t = from;
        while t < to {
            let main = rearm.may_ask(rearm.asked, t).then_some(main);
            shown = rearm.slot(0, main, second, t);
            if rearm.take_wanted(t) {
                asked += 1;
            }
            t += 8_000;
        }
        (asked, shown)
    }

    #[test]
    fn a_trigger_held_still_is_rearmed_once_and_keeps_reading_held() {
        let mut rearm = Rearm::default();
        let main = Some((held(), 73));
        let second = Probe::Reading(held(), 9);
        let (asked, shown) = run(&mut rearm, 0, REARM_STILL - 10_000, main, second);
        assert_eq!((asked, shown), (0, Some(held())), "not before REARM_STILL");
        let (asked, shown) = run(&mut rearm, REARM_STILL - 10_000, 60 * S, main, second);
        assert_eq!(asked, 1, "one re-arm per still spell");
        assert_eq!(shown, Some(held()), "the second module agrees: kept");
        assert_eq!(rearm.slots[0].stage, Stage::Kept);
        // Let go: a new packet, back to watching.
        let (asked, shown) = run(
            &mut rearm,
            60 * S,
            61 * S,
            Some((PadState::default(), 74)),
            second,
        );
        assert_eq!((asked, shown), (0, Some(PadState::default())));
        assert_eq!(rearm.slots[0].stage, Stage::Watching);
    }

    #[test]
    fn a_pad_at_rest_or_drifting_is_never_rearmed() {
        let mut rearm = Rearm::default();
        let drift = PadState {
            left_x: 0.2,
            right_y: -0.1,
            ..Default::default()
        };
        for pad in [PadState::default(), drift] {
            let (asked, shown) = run(&mut rearm, 0, 30 * S, Some((pad, 5)), Probe::Off);
            assert_eq!((asked, shown), (0, Some(pad)));
        }
    }

    #[test]
    fn a_read_behind_the_second_module_shows_its_reading_and_is_rearmed() {
        let mut rearm = Rearm::default();
        let main = Some((held(), 78));
        run(&mut rearm, 0, S, main, Probe::Reading(held(), 9));
        // Let go: only the second module sees it.
        let let_go = Probe::Reading(PadState::default(), 10);
        let (asked, shown) = run(&mut rearm, S, S + REARM_BEHIND - 10_000, main, let_go);
        assert_eq!((asked, shown), (0, Some(held())), "not before REARM_BEHIND");
        let (asked, shown) = run(&mut rearm, S + REARM_BEHIND - 10_000, 10 * S, main, let_go);
        assert_eq!((asked, shown), (1, Some(PadState::default())));
        // Ours moves again: ours is shown.
        let (_, shown) = run(&mut rearm, 10 * S, 11 * S, Some((held(), 79)), let_go);
        assert_eq!(shown, Some(held()));
        assert_eq!(rearm.slots[0].stage, Stage::Watching);
    }

    #[test]
    fn a_report_the_second_module_takes_a_poll_later_is_not_newer() {
        let mut rearm = Rearm::default();
        // A stick nudged, inside the drift bound: no re-arm of its own.
        let a = PadState {
            left_x: 0.2,
            ..Default::default()
        };
        run(
            &mut rearm,
            0,
            S,
            Some((PadState::default(), 79)),
            Probe::Reading(PadState::default(), 9),
        );
        // Ours takes the report at 1 s, the second module 8 ms later.
        rearm.slot(
            0,
            Some(Some((a, 80))),
            Probe::Reading(PadState::default(), 9),
            S,
        );
        let (asked, shown) = run(
            &mut rearm,
            S + 8_000,
            10 * S,
            Some((a, 80)),
            Probe::Reading(a, 10),
        );
        assert_eq!(
            (asked, shown, rearm.slots[0].stage),
            (0, Some(a), Stage::Watching)
        );
        // A tap our module missed (down and up, the same reading again) is newer.
        let (_, _) = run(
            &mut rearm,
            10 * S,
            11 * S,
            Some((a, 80)),
            Probe::Reading(a, 12),
        );
        assert_eq!(rearm.slots[0].stage, Stage::Second);
    }

    #[test]
    fn a_frozen_pad_the_second_module_lost_reads_at_rest_until_it_moves() {
        let mut rearm = Rearm::default();
        let main = Some((held(), 78));
        run(&mut rearm, 0, S, main, Probe::Reading(held(), 9));
        // From 1 s the second module finds no pad; ours has been still since 0.
        let gone = Probe::Error(1167);
        let waiting = REARM_STILL + REARM_WAIT - 20_000;
        let (asked, shown) = run(&mut rearm, S, waiting, main, gone);
        assert_eq!((asked, shown), (1, Some(held())), "re-armed, still waiting");
        let (_, shown) = run(&mut rearm, waiting, 30 * S, main, gone);
        assert_eq!(shown, Some(PadState::default()), "at rest");
        // The pad comes back in the second module and moves: its reading.
        run(
            &mut rearm,
            30 * S,
            31 * S,
            main,
            Probe::Reading(PadState::default(), 0),
        );
        let a = PadState {
            buttons: A,
            ..Default::default()
        };
        let (_, shown) = run(&mut rearm, 31 * S, 32 * S, main, Probe::Reading(a, 1));
        assert_eq!(shown, Some(a));
        // Ours moves: ours.
        let (_, shown) = run(
            &mut rearm,
            32 * S,
            33 * S,
            Some((a, 79)),
            Probe::Reading(a, 1),
        );
        assert_eq!((shown, rearm.slots[0].stage), (Some(a), Stage::Watching));
    }

    #[test]
    fn no_pad_in_the_second_module_counts_only_after_it_read_one() {
        let mut rearm = Rearm::default();
        let (asked, shown) = run(
            &mut rearm,
            0,
            30 * S,
            Some((held(), 78)),
            Probe::Error(1167),
        );
        assert_eq!((asked, shown), (1, Some(held())));
        assert_eq!(rearm.slots[0].stage, Stage::Kept);
        // Without a second module at all: the same.
        let mut rearm = Rearm::default();
        let (asked, shown) = run(&mut rearm, 0, 30 * S, Some((held(), 78)), Probe::Off);
        assert_eq!((asked, shown), (1, Some(held())));
    }

    #[test]
    fn a_rearm_that_never_returns_stops_asking_and_rests_the_pad() {
        let mut rearm = Rearm::default();
        let main = Some((held(), 78));
        let second = Probe::Reading(held(), 9);
        let mut t = 0;
        while !rearm.take_wanted(t) {
            assert!(rearm.may_ask(rearm.asked, t));
            rearm.slot(0, Some(main), second, t);
            t += 8_000;
        }
        // The helper never finishes.
        let mut shown = None;
        while t < 20 * S {
            assert!(rearm.asked == 1 && !rearm.take_wanted(t));
            let main = rearm.may_ask(0, t).then_some(main);
            assert!(
                main.is_none() || t < rearm.asked_at + REARM_HUNG,
                "not asked while it runs"
            );
            shown = rearm.slot(0, main, second, t);
            t += 8_000;
        }
        assert!(rearm.hung);
        assert_eq!(shown, Some(PadState::default()));
        // The second module still reads: once it moves, its reading.
        let a = PadState {
            buttons: A,
            ..Default::default()
        };
        let mut last = None;
        while t < 21 * S {
            assert!(!rearm.may_ask(0, t));
            last = rearm.slot(0, None, Probe::Reading(a, 10), t);
            t += 8_000;
        }
        assert_eq!(last, Some(a));
    }

    #[test]
    fn a_disconnect_forgets_the_spell() {
        let mut rearm = Rearm::default();
        run(
            &mut rearm,
            0,
            10 * S,
            Some((held(), 78)),
            Probe::Error(1167),
        );
        assert_eq!(rearm.slot(0, Some(None), Probe::Off, 10 * S), None);
        assert_eq!(rearm.slots[0].stage, Stage::Watching);
        let (asked, _) = run(
            &mut rearm,
            10 * S,
            10 * S + REARM_STILL - 10_000,
            Some((held(), 1)),
            Probe::Off,
        );
        assert_eq!(asked, 0, "a new connection starts a new spell");
    }

    #[test]
    fn rumble_is_dropped_only_when_asked() {
        assert!(rumble_off(Some("0")) && rumble_off(Some(" off ")));
        assert!(!rumble_off(None) && !rumble_off(Some("1")) && !rumble_off(Some("")));
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
