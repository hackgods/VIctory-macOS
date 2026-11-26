//! mac: the plain arithmetic of the flights (`crate::host::unwaited::flights`,
//! `VICTORY_FLIGHTS=1`): plain std, so its tests run everywhere.

/// Title memory as ranges `[from, to)`: after [`Spans::join`], in order,
/// none touching the next.
#[derive(Debug, Default, Clone)]
pub(crate) struct Spans(Vec<(usize, usize)>);

impl Spans {
    /// Adds `[from, to)` (nothing if empty); [`Spans::join`] after the last.
    pub(crate) fn add(&mut self, from: usize, to: usize) {
        if from < to {
            self.0.push((from, to));
        }
    }

    /// Puts the ranges in order, those that overlap or touch as one.
    pub(crate) fn join(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(self.0.len());
        for &(from, to) in &self.0 {
            match out.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => out.push((from, to)),
            }
        }
        self.0 = out;
    }

    /// Whether `[from, from + len)` shares a byte with one of them (joined).
    pub(crate) fn touches(&self, from: usize, len: usize) -> bool {
        let to = from.saturating_add(len);
        if from >= to {
            return false;
        }
        let at = self.0.partition_point(|r| r.1 <= from);
        self.0.get(at).is_some_and(|r| r.0 < to)
    }

    #[cfg(test)]
    fn ranges(&self) -> &[(usize, usize)] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_join_where_they_touch_and_find_what_they_share() {
        let mut s = Spans::default();
        s.add(100, 200);
        s.add(300, 400);
        s.add(200, 250);
        s.add(500, 500);
        s.add(350, 360);
        s.join();
        assert_eq!(s.ranges(), &[(100, 250), (300, 400)]);
        assert!(s.touches(249, 1));
        assert!(!s.touches(250, 50));
        assert!(s.touches(250, 51));
        assert!(!s.touches(0, 100));
        assert!(s.touches(0, 101));
        assert!(!s.touches(400, 1000));
        assert!(!s.touches(120, 0));
        assert!(!Spans::default().touches(0, usize::MAX));
    }
}
