//! macOS port: `VICTORY_KEYS=menu:<key>[:<every>[:<draws>]]`, a key pressed
//! at the start screens by what the frames show, not at set times. A child of
//! `gameinput`; the host side (Windows) feeds it, the rule here is plain Rust.
//!
//! Timed entries (`20:Enter`) failed both ways (27 November): a slow boot
//! brought the screens up after the last press and the run sat on a start
//! screen for 12 minutes (runs 21:37, 21:50), and on a normal boot each of the
//! three screens waited for its press at input + 20, 30 and 40 s. The rule,
//! read off the frame lines (`notes/keys-menu.md` has the numbers):
//! - No press before the first start screen: `SCREEN_FRAMES` frames in a row
//!   with `SCREEN_DRAWS` draws or more. Boot frames draw 3 (1 in one setup);
//!   the start screens 6 to 13, their first frame line 6 or more in every run.
//! - Then a press (held `HOLD`) every `every` seconds, 3 by default.
//! - Never again, from the first of:
//!   - a frame with more than `limit` draws (50 by default): the world, 500
//!     and more; no start screen or loading frame line averaged above 13;
//!   - `grace` after the pipelines being made rose `RISE` or more above their
//!     lowest since the second press: in 266 runs that rise never came on the
//!     first two screens and always came with the last one's preload (input +
//!     37.5 to 40) or the loading after its press. Loading itself draws 10 to
//!     12 a frame like that screen, so the draws cannot tell them apart; the
//!     grace (three presses, at least 10 s) lets the last screen take its
//!     press first;
//!   - `MAX` seconds after the first start screen (not after input came up:
//!     in run 22:41 the first screen came 165 s after input).
//!
//! What the frames are comes from victory-gpu (`queue::last_presented`,
//! `pipeline::waiting`): atomic loads, no lock on the frame path.

/// How long a press holds the key down (a timed entry's default too).
pub(super) const HOLD: f64 = 0.2;
const EVERY: f64 = 3.0;
const LIMIT: u64 = 50;
/// A start screen: this many frames in a row with this many draws or more.
const SCREEN_DRAWS: u64 = 6;
const SCREEN_FRAMES: u64 = 30;
/// Pipelines being made above their lowest since the second press.
const RISE: u64 = 100;
/// Presses after that rise before the last one, and the least time for them.
const GRACE_PRESSES: f64 = 3.0;
const GRACE_LEAST: f64 = 10.0;
/// Seconds after the first start screen.
const MAX: f64 = 180.0;

/// One `menu:` entry.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Menu {
    pub(super) name: String,
    pub(super) key: u8,
    every: f64,
    limit: u64,
}

/// `menu:<key>[:<every>[:<draws>]]`; `key` names a key (the timed entries'
/// names). None if `entry` is not one or does not parse.
pub(super) fn parse(entry: &str, key: impl Fn(&str) -> Option<u8>) -> Option<Menu> {
    let mut parts = entry.trim().strip_prefix("menu:")?.split(':').map(str::trim);
    let name = parts.next()?;
    let vk = key(name)?;
    let every = match parts.next() {
        Some(s) => s.parse::<f64>().ok().filter(|e| e.is_finite() && *e >= 2.0 * HOLD)?,
        None => EVERY,
    };
    let limit = match parts.next() {
        Some(s) => s.parse::<u64>().ok().filter(|&l| l > 0)?,
        None => LIMIT,
    };
    if parts.next().is_some() {
        return None;
    }
    Some(Menu {
        name: name.to_owned(),
        key: vk,
        every,
        limit,
    })
}

/// What victory-gpu says of the frames.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Sample {
    /// Frames presented so far.
    pub(super) frames: u64,
    /// Draws the last of them ran.
    pub(super) draws: u64,
    /// Host pipelines being made.
    pub(super) waiting: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Why {
    World { frame: u64, draws: u64 },
    Loading { at: f64, from: u64, to: u64 },
    Time,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Event {
    Pressed { at: f64, n: u32, frame: u64, draws: u64, waiting: u64 },
    Closed { at: f64, presses: u32, why: Why },
}

#[derive(Clone, Copy, Debug)]
struct Rise {
    at: f64,
    from: u64,
    to: u64,
}

/// One `menu:` entry's state; see the module notes.
#[derive(Debug)]
pub(super) struct State {
    pub(super) menu: Menu,
    /// The last frame seen and its draws.
    frame: u64,
    draws: u64,
    /// Frames in a row with `SCREEN_DRAWS` or more.
    screen: u64,
    /// When the first start screen was seen.
    armed: Option<f64>,
    presses: u32,
    pressed: Option<f64>,
    lowest: Option<u64>,
    rise: Option<Rise>,
    closed: bool,
}

impl State {
    pub(super) fn new(menu: Menu) -> State {
        State {
            menu,
            frame: 0,
            draws: 0,
            screen: 0,
            armed: None,
            presses: 0,
            pressed: None,
            lowest: None,
            rise: None,
            closed: false,
        }
    }

    /// A look at the frames `now` seconds after input came up: whether the
    /// key is down, and what to log. Asked again at the same moment (the
    /// diagnostics build asks twice a poll) it answers the same.
    pub(super) fn step(&mut self, now: f64, s: Sample) -> (bool, Option<Event>) {
        if self.closed {
            return (false, None);
        }
        if s.frames > self.frame {
            if s.draws > self.menu.limit {
                let why = Why::World {
                    frame: s.frames,
                    draws: s.draws,
                };
                return self.close(now, why);
            }
            if s.draws >= SCREEN_DRAWS {
                self.screen += s.frames - self.frame;
            } else {
                self.screen = 0;
            }
            (self.frame, self.draws) = (s.frames, s.draws);
        }
        if self.armed.is_some_and(|at| now >= at + MAX) {
            return self.close(now, Why::Time);
        }
        if self.presses >= 2 {
            let lowest = self.lowest.map_or(s.waiting, |l| l.min(s.waiting));
            self.lowest = Some(lowest);
            if self.rise.is_none() && s.waiting >= lowest + RISE {
                self.rise = Some(Rise {
                    at: now,
                    from: lowest,
                    to: s.waiting,
                });
            }
        }
        let grace = (GRACE_PRESSES * self.menu.every).max(GRACE_LEAST);
        if let Some(r) = self.rise
            && now >= r.at + grace
        {
            let why = Why::Loading {
                at: r.at,
                from: r.from,
                to: r.to,
            };
            return self.close(now, why);
        }
        if self.armed.is_none() && self.screen >= SCREEN_FRAMES {
            self.armed = Some(now);
        }
        let mut event = None;
        if self.armed.is_some() && self.pressed.is_none_or(|t| now >= t + self.menu.every) {
            self.pressed = Some(now);
            self.presses += 1;
            event = Some(Event::Pressed {
                at: now,
                n: self.presses,
                frame: self.frame,
                draws: self.draws,
                waiting: s.waiting,
            });
        }
        (self.pressed.is_some_and(|t| now < t + HOLD), event)
    }

    fn close(&mut self, now: f64, why: Why) -> (bool, Option<Event>) {
        self.closed = true;
        let presses = self.presses;
        (false, Some(Event::Closed { at: now, presses, why }))
    }

    /// The log line for `event`.
    pub(super) fn line(&self, event: &Event) -> String {
        let name = &self.menu.name;
        match *event {
            Event::Pressed {
                at,
                n,
                frame,
                draws,
                waiting,
            } => format!(
                "VICTORY_KEYS menu: {name} press {n} at {at:.1} s after input came up (frame {frame}: {draws} \
                 draws; {waiting} pipelines being made)"
            ),
            Event::Closed { at, presses, why } => {
                let why = match why {
                    Why::World { frame, draws } => {
                        format!("frame {frame} ran {draws} draws, over {}: the world", self.menu.limit)
                    }
                    Why::Loading { at, from, to } => format!(
                        "pipelines being made rose from {from} to {to} at {at:.1} s: the last start screen's \
                         preload or the loading"
                    ),
                    Why::Time => format!("{MAX:.0} s after the first start screen"),
                };
                format!(
                    "VICTORY_KEYS menu: {name} not pressed again in this run, from {at:.1} s after input came up \
                     ({presses} presses): {why}"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> Option<u8> {
        super::super::virtual_key(name)
    }

    fn menu() -> Menu {
        parse("menu:Enter", key).unwrap()
    }

    #[test]
    fn menu_entries_parse() {
        assert_eq!(
            menu(),
            Menu {
                name: "Enter".into(),
                key: 0x0D,
                every: 3.0,
                limit: 50
            }
        );
        let m = parse(" menu:Enter:5:80 ", key).unwrap();
        assert_eq!((m.every, m.limit), (5.0, 80));
        assert_eq!(parse("menu:Enter:1.5", key).unwrap().limit, 50);
        for bad in [
            "20:Enter",
            "menu:",
            "menu:Space",
            "menu:Enter:0.3",
            "menu:Enter:x",
            "menu:Enter:3:0",
            "menu:Enter:3:50:1",
        ] {
            assert_eq!(parse(bad, key), None, "{bad}");
        }
    }

    /// A frame line: seconds after input came up, fps, draws a frame, and
    /// pipelines being made (the frame line's numbers).
    type Line = (f64, f64, u64, u64);

    /// Plays `lines` as frames (each line's frames spread evenly over the
    /// five seconds before it, all with its draws and pipelines) to the rule,
    /// looked at every 8 ms as the poll thread does: the presses' times, and
    /// the close with its time.
    fn replay(lines: &[Line], until: f64) -> (Vec<f64>, Option<(f64, Why)>, Vec<f64>) {
        let mut frames: Vec<(f64, u64, u64)> = Vec::new();
        let mut before = lines[0].0 - 5.0;
        for &(t, fps, draws, waiting) in lines {
            let n = (fps * (t - before)).round().max(1.0) as usize;
            for k in 1..=n {
                frames.push((before + (t - before) * k as f64 / n as f64, draws, waiting));
            }
            before = t;
        }
        let mut state = State::new(menu());
        let (mut presses, mut closed, mut down) = (Vec::new(), None, Vec::new());
        let mut shown = 0;
        let mut now = 0.0;
        while now < until {
            while shown < frames.len() && frames[shown].0 <= now {
                shown += 1;
            }
            let s = match shown {
                0 => Sample::default(),
                n => Sample {
                    frames: n as u64,
                    draws: frames[n - 1].1,
                    waiting: frames[n - 1].2,
                },
            };
            let (held, event) = state.step(now, s);
            // Asked twice, as in the diagnostics build: the same answer.
            assert_eq!(state.step(now, s), (held, None));
            if held {
                down.push(now);
            }
            match event {
                Some(Event::Pressed { at, .. }) => presses.push(at),
                Some(Event::Closed { at, why, .. }) => closed = Some((at, why)),
                None => {}
            }
            now += 0.008;
        }
        (presses, closed, down)
    }

    fn any_in(presses: &[f64], from: f64, to: f64) -> bool {
        presses.iter().any(|&p| p > from && p <= to)
    }

    /// `2026-11-27_21-31-22_main-mac.log` (input up 12.5 s after the start):
    /// a normal boot. Timed presses at 20, 30, 40 s; loading began after the
    /// third, the world at the 82.1 line.
    const NORMAL: &[Line] = &[
        (12.0, 10.1, 3, 0), (17.1, 30.1, 12, 0), (22.2, 30.0, 10, 0), (27.3, 30.0, 6, 0),
        (32.4, 30.0, 6, 0), (37.5, 29.8, 8, 0), (42.6, 30.0, 11, 251), (47.7, 29.6, 11, 212),
        (53.1, 24.2, 11, 178), (58.8, 23.1, 10, 178), (63.9, 19.2, 11, 178), (82.1, 3.0, 88, 182),
        (87.8, 0.7, 2436, 142),
    ];

    /// `2026-11-27_21-37-13_main-mac.log` (input up 27.4 s after the start):
    /// boot frames until 54.6, the first screen came after four of the five
    /// timed presses (20 to 60 s); the screen at 161.5 got none.
    const SLOW_2137: &[Line] = &[
        (10.3, 26.9, 3, 0), (15.4, 30.0, 3, 0), (20.5, 30.0, 3, 0), (25.6, 30.0, 3, 0),
        (30.7, 30.0, 3, 0), (35.9, 30.2, 3, 0), (40.9, 30.0, 3, 0), (46.4, 24.9, 3, 0),
        (54.6, 4.9, 3, 0), (59.6, 30.6, 11, 0), (64.8, 30.0, 7, 0), (69.9, 30.0, 6, 0),
        (75.0, 30.0, 6, 0), (80.1, 30.0, 3, 0), (85.2, 30.2, 3, 0), (90.3, 30.0, 3, 0),
        (95.4, 30.2, 3, 0), (100.5, 30.0, 3, 0), (105.6, 30.2, 3, 0), (110.7, 30.2, 3, 0),
        (115.8, 30.0, 3, 0), (120.9, 30.2, 3, 0), (126.0, 30.0, 3, 0), (131.1, 30.0, 3, 0),
        (136.1, 30.4, 3, 0), (141.2, 30.4, 3, 0), (146.3, 30.4, 3, 0), (151.4, 30.0, 3, 0),
        (156.5, 29.8, 3, 0), (161.5, 30.0, 8, 0), (166.7, 30.0, 11, 0), (171.8, 30.0, 11, 0),
        (176.9, 30.0, 11, 0), (181.9, 30.0, 10, 0), (187.0, 30.0, 11, 0), (192.1, 29.8, 11, 0),
    ];

    /// `2026-11-27_21-50-37_main-mac.log` (input up 12.5 s after the start):
    /// boot frames (one of 6 draws at 0.2 fps, 302 pipelines) until 44.7; the
    /// last start screen from about 75 waited for a press until the 12-minute
    /// stop.
    const SLOW_2150: &[Line] = &[
        (8.5, 26.5, 3, 0), (14.1, 24.6, 3, 0), (19.4, 0.4, 3, 0), (24.4, 0.2, 6, 0), (29.5, 30.3, 3, 0),
        (34.6, 30.4, 3, 0), (39.7, 28.1, 3, 302), (44.7, 1.2, 3, 0), (49.8, 30.0, 12, 0),
        (54.9, 30.0, 7, 0), (60.1, 30.0, 6, 0), (65.1, 30.0, 6, 0), (70.2, 29.8, 3, 0),
        (75.3, 30.0, 10, 0), (80.4, 30.0, 11, 0), (85.5, 29.8, 11, 0), (90.5, 30.0, 11, 722),
        (95.6, 30.0, 11, 225), (100.8, 30.0, 11, 177), (105.8, 30.0, 11, 177), (110.9, 30.0, 10, 177),
        (116.0, 30.0, 11, 177), (121.1, 30.0, 11, 176), (126.2, 30.0, 10, 176),
    ];

    /// `2026-11-27_22-09-41_main-mac.log` (input up 19.9 s after the start;
    /// main-mac b0ef5e7a): boot until 30.3, two screens, then 3 draws a frame
    /// to the stop.
    const SLOW_2209: &[Line] = &[
        (8.3, 26.8, 3, 0), (13.3, 30.0, 3, 0), (18.4, 30.0, 3, 0), (24.0, 19.2, 3, 0),
        (30.3, 3.8, 3, 0), (35.3, 30.1, 11, 0), (40.4, 30.0, 13, 0), (45.6, 30.0, 6, 0),
        (50.7, 30.0, 6, 0), (55.7, 30.0, 6, 0), (60.8, 30.2, 3, 0), (65.9, 30.0, 3, 0),
        (71.0, 30.2, 3, 0), (76.2, 30.0, 3, 0), (81.2, 30.0, 3, 0), (86.3, 30.0, 3, 0),
        (91.4, 29.8, 3, 0), (96.5, 30.2, 3, 0), (101.6, 30.2, 3, 0), (106.7, 30.0, 3, 0),
        (111.8, 30.0, 3, 0), (116.8, 30.0, 3, 0), (121.9, 30.0, 3, 0), (127.0, 30.0, 3, 0),
        (132.1, 30.0, 3, 0), (137.2, 30.0, 3, 0), (142.2, 30.0, 3, 0), (147.3, 30.2, 3, 0),
        (152.4, 30.0, 3, 0), (157.4, 30.0, 3, 0), (162.5, 30.0, 3, 0), (167.6, 30.0, 3, 0),
        (172.7, 30.0, 3, 0), (177.7, 30.2, 3, 0), (182.8, 30.0, 3, 0), (187.9, 30.0, 3, 0),
    ];

    /// `2026-11-27_20-25-43_main-mac.log` (input up 24.8 s after the start):
    /// a cold pipeline cache, about 1,600 being made from the boot on.
    const COLD_2025: &[Line] = &[
        (13.3, 9.1, 3, 1616), (18.4, 30.1, 12, 1577), (23.5, 27.6, 9, 1536), (28.6, 30.0, 6, 1297),
        (33.6, 30.0, 6, 1190), (38.8, 29.8, 8, 1506), (43.9, 30.0, 10, 2447), (49.7, 25.5, 11, 2406),
        (54.8, 29.0, 11, 2376), (59.9, 26.0, 10, 2338), (65.3, 27.3, 12, 2316), (73.0, 8.2, 60, 1922),
        (78.2, 1.3, 1313, 1597),
    ];

    #[test]
    fn normal_boot_presses_from_the_first_screen_and_stops_before_the_world() {
        let (presses, closed, _) = replay(NORMAL, 120.0);
        // None in the boot; the first one second into the first screen.
        assert!(presses[0] > 12.0 && presses[0] < 13.2, "{presses:?}");
        for w in NORMAL.windows(2).take(5) {
            assert!(any_in(&presses, w[0].0, w[1].0), "no press in {w:?}: {presses:?}");
        }
        // The pipelines rose in the 37.5 to 42.6 window (its frames all
        // carry the 42.6 line's 251): closed 10 s after its first frame.
        let (at, why) = closed.unwrap();
        assert!(matches!(why, Why::Loading { from: 0, to: 251, .. }), "{why:?}");
        assert!((at - 47.53).abs() < 0.05, "{at}");
        assert!(presses.iter().all(|&p| p < at));
        assert_eq!(presses.len(), 12, "{presses:?}");
    }

    #[test]
    fn without_a_pipeline_rise_the_world_closes_it() {
        // 39 of 266 runs showed no pipelines being made at the loading.
        let flat: Vec<Line> = NORMAL.iter().map(|&(t, f, d, _)| (t, f, d, 0)).collect();
        let (presses, closed, _) = replay(&flat, 120.0);
        let (at, why) = closed.unwrap();
        // The first frame of the 88-draw window, just after 63.9.
        assert!(matches!(why, Why::World { draws: 88, .. }), "{why:?}");
        assert!(at > 63.9 && at < 64.3, "{at}");
        assert!(presses.iter().all(|&p| p < at));
    }

    #[test]
    fn slow_boot_2137_gets_presses_on_its_screens() {
        let (presses, closed, _) = replay(SLOW_2137, 250.0);
        // Nothing in 54.6 s of boot frames; the timed presses' four there
        // did nothing either (the first screen still took the fifth's).
        let first = presses[0];
        assert!(first > 54.6 && first < 56.0, "{presses:?}");
        // The first screens, and the screen from 161.5 the timed entries
        // (60 s the last) left without a press.
        assert!(any_in(&presses, 54.6, 59.6));
        assert!(any_in(&presses, 59.6, 75.0));
        assert!(presses.iter().filter(|&&p| p > 161.5 && p <= 192.1).count() >= 9);
        // 180 s after the first screen.
        let (at, why) = closed.unwrap();
        assert_eq!(why, Why::Time);
        assert!((at - (first + 180.0)).abs() < 0.01, "{at}");
        assert!(presses.iter().all(|&p| p < at));
    }

    #[test]
    fn slow_boot_2150_gets_presses_on_the_screen_it_sat_on() {
        let (presses, closed, _) = replay(SLOW_2150, 200.0);
        // The single 6-draw boot frame and the 302 pipelines of the boot
        // neither start presses nor count as the rise.
        assert!(presses[0] > 44.7 && presses[0] < 46.0, "{presses:?}");
        assert!(any_in(&presses, 44.7, 49.8));
        assert!(any_in(&presses, 49.8, 65.1));
        // The screen it waited on from about 75 (timed presses ended at 60).
        assert!(presses.iter().filter(|&&p| p > 72.0 && p <= 85.5).count() >= 4);
        let (at, why) = closed.unwrap();
        assert!(matches!(why, Why::Loading { from: 0, to: 722, .. }), "{why:?}");
        assert!((at - 95.53).abs() < 0.05, "{at}");
    }

    #[test]
    fn slow_boot_2209_presses_on_to_the_time_limit() {
        let (presses, closed, _) = replay(SLOW_2209, 250.0);
        let first = presses[0];
        assert!(first > 30.3 && first < 31.5, "{presses:?}");
        assert!(any_in(&presses, 40.4, 55.7));
        assert!(any_in(&presses, 60.8, 187.9));
        let (at, why) = closed.unwrap();
        assert_eq!(why, Why::Time);
        assert!((at - (first + 180.0)).abs() < 0.01, "{at}");
    }

    /// `2026-11-27_22-24-36_main-mac.log` (input up 25.8 s after the start;
    /// main-mac b7528797): first screen at 41, boot-like 3 draws from 71.6
    /// to 148, the last screen from about 153 with no press left.
    const SLOW_2224: &[Line] = &[
        (8.6, 26.4, 3, 0), (13.7, 30.0, 3, 0), (18.8, 30.0, 3, 0), (23.9, 30.0, 3, 0),
        (29.0, 30.0, 3, 0), (34.6, 21.4, 3, 0), (41.0, 3.6, 3, 0), (46.1, 30.4, 10, 0),
        (51.2, 30.4, 12, 0), (56.3, 30.0, 6, 0), (61.4, 30.0, 6, 0), (66.5, 30.0, 5, 0),
        (71.6, 30.4, 3, 0), (76.7, 30.0, 3, 0), (81.8, 30.4, 3, 0), (86.9, 30.0, 3, 0),
        (92.0, 30.2, 3, 0), (97.1, 30.0, 3, 0), (102.2, 30.0, 3, 0), (107.3, 30.4, 3, 0),
        (112.4, 30.2, 3, 0), (117.5, 30.2, 3, 0), (122.5, 30.0, 3, 0), (127.6, 30.2, 3, 0),
        (132.7, 30.0, 3, 0), (137.8, 30.0, 3, 0), (142.9, 30.0, 3, 0), (148.0, 30.0, 3, 0),
        (153.1, 29.8, 5, 0), (158.1, 30.0, 10, 0), (163.2, 30.0, 11, 0), (168.3, 29.8, 11, 0),
        (173.4, 30.0, 10, 243), (178.5, 30.0, 11, 178), (183.6, 30.0, 11, 177), (188.7, 30.0, 11, 177),
    ];

    /// `2026-11-27_22-41-32_main-mac.log` (input up 12.4 s after the start;
    /// main-mac b7528797): boot frames for 160 s (4 pipelines at 136.7, a
    /// 2-draw line at 0.9 fps), then the first screen to the 12-minute stop.
    const SLOW_2241: &[Line] = &[
        (9.4, 25.8, 3, 0), (14.5, 30.0, 3, 0), (19.6, 30.0, 3, 0), (24.7, 30.0, 3, 0),
        (29.8, 30.0, 3, 0), (34.8, 30.0, 3, 0), (39.9, 30.0, 3, 0), (45.0, 30.0, 3, 0),
        (50.1, 30.0, 3, 0), (55.2, 30.2, 3, 0), (60.3, 30.2, 3, 0), (65.3, 30.4, 3, 0),
        (70.4, 30.2, 3, 0), (75.5, 30.2, 3, 0), (80.6, 30.2, 3, 0), (85.6, 30.8, 3, 0),
        (90.7, 30.2, 3, 0), (95.8, 30.0, 3, 0), (101.0, 30.2, 3, 0), (106.1, 30.0, 3, 0),
        (111.2, 30.0, 3, 0), (116.3, 30.8, 3, 0), (121.4, 30.8, 3, 0), (126.5, 30.4, 3, 0),
        (131.6, 30.0, 3, 0), (136.7, 16.5, 3, 4), (143.5, 0.9, 2, 0), (149.1, 22.3, 3, 0),
        (154.6, 7.2, 3, 0), (159.6, 2.4, 3, 0), (164.7, 30.4, 8, 0), (169.8, 30.6, 13, 0),
        (174.9, 30.0, 13, 0), (180.1, 30.8, 13, 0), (185.2, 30.2, 13, 0), (190.2, 31.2, 13, 0),
        (195.3, 30.2, 13, 0), (200.4, 30.2, 13, 0), (205.5, 30.2, 13, 0), (210.6, 30.2, 13, 0),
    ];

    /// `2026-11-27_22-53-35_main-mac.log` (input up 5.6 s after the start;
    /// main-mac b7528797): boot frames until 62.3, the first screen 2 s after
    /// the last timed press, sat on to the stop.
    const SLOW_2253: &[Line] = &[
        (8.1, 24.8, 3, 0), (13.2, 30.2, 3, 0), (18.3, 30.2, 3, 0), (23.4, 30.0, 3, 0),
        (28.4, 30.2, 3, 0), (33.5, 31.0, 3, 0), (38.6, 31.0, 3, 0), (43.7, 30.0, 3, 0),
        (48.8, 30.8, 3, 0), (54.0, 16.1, 3, 0), (62.3, 7.9, 3, 0), (67.3, 30.5, 11, 0),
        (72.5, 30.6, 13, 0), (77.5, 30.8, 13, 0), (82.6, 30.8, 13, 0), (87.7, 30.4, 13, 0),
        (92.8, 30.0, 13, 0), (97.9, 30.0, 13, 0),
    ];

    #[test]
    fn slow_boot_2224_gets_presses_on_its_last_screen() {
        let (presses, closed, _) = replay(SLOW_2224, 250.0);
        assert!(presses[0] > 41.0 && presses[0] < 42.5, "{presses:?}");
        assert!(any_in(&presses, 46.1, 66.5));
        assert!(presses.iter().filter(|&&p| p > 153.1 && p <= 168.3).count() >= 4);
        // 243 being made in the 168.3 to 173.4 window: closed 10 s on.
        let (at, why) = closed.unwrap();
        assert!(matches!(why, Why::Loading { from: 0, to: 243, .. }), "{why:?}");
        assert!((at - 178.33).abs() < 0.05, "{at}");
    }

    #[test]
    fn slow_boot_2241_still_presses_past_180_s_after_input() {
        let (presses, closed, _) = replay(SLOW_2241, 260.0);
        // 160 s of boot frames (and their 4 pipelines) start nothing.
        let first = presses[0];
        assert!(first > 159.6 && first < 161.0, "{presses:?}");
        // A limit counted from input up would have stopped at 180, 20 s into
        // the first screen; counted from it, presses go on.
        assert!(presses.iter().filter(|&&p| p > 180.0 && p <= 210.6).count() >= 9);
        assert_eq!(closed, None);
    }

    #[test]
    fn slow_boot_2253_presses_once_its_first_screen_comes() {
        let (presses, closed, _) = replay(SLOW_2253, 100.0);
        assert!(presses[0] > 62.3 && presses[0] < 64.0, "{presses:?}");
        assert!(presses.iter().filter(|&&p| p <= 97.9).count() >= 11);
        assert_eq!(closed, None);
    }

    #[test]
    fn cold_cache_backlog_is_no_rise_until_the_last_screen() {
        let (presses, closed, _) = replay(COLD_2025, 120.0);
        assert!(presses[0] > 13.3 && presses[0] < 14.5, "{presses:?}");
        // 1,577 to 1,190 drained over the first screens; 1,506 in the
        // 33.6 to 38.8 window is the rise (316).
        let (at, why) = closed.unwrap();
        assert!(matches!(why, Why::Loading { from: 1190, to: 1506, .. }), "{why:?}");
        assert!((at - 43.63).abs() < 0.05, "{at}");
        assert!(any_in(&presses, 33.6, 38.8));
    }

    #[test]
    fn a_press_holds_the_key_for_its_hold_only() {
        let (presses, _, down) = replay(NORMAL, 30.0);
        let first = presses[0];
        let held: Vec<f64> = down.iter().copied().filter(|&t| t < first + 1.0).collect();
        assert!(held.first() == Some(&first));
        assert!(held.iter().all(|&t| t < first + HOLD));
        assert!(held.len() >= 24, "{}", held.len());
    }

    #[test]
    fn closed_stays_closed() {
        let mut s = State::new(menu());
        let mut frames = 0;
        let mut look = |s: &mut State, now: f64, draws: u64| {
            frames += 1;
            s.step(
                now,
                Sample {
                    frames,
                    draws,
                    waiting: 0,
                },
            )
        };
        for k in 0..40 {
            look(&mut s, k as f64 * 0.033, 12);
        }
        assert!(s.armed.is_some());
        // Pressed at 0.96 s (the 30th frame); a world frame mid-hold lets
        // the key up at once.
        let (held, event) = look(&mut s, 1.1, 3000);
        assert!(!held);
        assert!(matches!(event, Some(Event::Closed { why: Why::World { draws: 3000, .. }, presses: 1, .. })));
        // A quiet cutscene afterwards: never again.
        for k in 0..1000 {
            assert_eq!(look(&mut s, 2.0 + k as f64 * 0.1, 4), (false, None));
        }
    }

    #[test]
    fn a_world_before_any_screen_means_no_press_at_all() {
        let mut s = State::new(menu());
        assert_eq!(s.step(1.0, Sample { frames: 1, draws: 3, waiting: 0 }), (false, None));
        let (held, event) = s.step(1.1, Sample { frames: 2, draws: 51, waiting: 0 });
        assert!(!held);
        assert!(matches!(event, Some(Event::Closed { presses: 0, .. })));
    }

    #[test]
    fn a_rise_before_the_second_press_does_not_count() {
        let mut s = State::new(menu());
        let mut presses = 0;
        for k in 0..6000 {
            let now = k as f64 * 0.033;
            // 1,000 pipelines from between the first press (0.96 s) and the
            // second (3.96 s) to 10 s, then 0, then 90: no rise of 100.
            let waiting = match now {
                t if t < 2.0 => 0,
                t if t < 10.0 => 1000,
                t if t < 20.0 => 0,
                _ => 90,
            };
            let sample = Sample { frames: k + 1, draws: 12, waiting };
            match s.step(now, sample).1 {
                Some(Event::Pressed { .. }) => presses += 1,
                Some(Event::Closed { why, .. }) => {
                    assert_eq!(why, Why::Time);
                    break;
                }
                None => {}
            }
        }
        // Every 3 s from 0.96 s to the limit.
        assert_eq!(presses, 60);
    }

    #[test]
    fn lines_say_what_happened() {
        let s = State::new(menu());
        let press = Event::Pressed { at: 13.04, n: 1, frame: 312, draws: 12, waiting: 0 };
        assert_eq!(
            s.line(&press),
            "VICTORY_KEYS menu: Enter press 1 at 13.0 s after input came up (frame 312: 12 draws; 0 pipelines \
             being made)"
        );
        let closed = Event::Closed {
            at: 47.5,
            presses: 12,
            why: Why::Loading { at: 37.5, from: 0, to: 251 },
        };
        assert_eq!(
            s.line(&closed),
            "VICTORY_KEYS menu: Enter not pressed again in this run, from 47.5 s after input came up (12 \
             presses): pipelines being made rose from 0 to 251 at 37.5 s: the last start screen's preload or \
             the loading"
        );
    }
}
