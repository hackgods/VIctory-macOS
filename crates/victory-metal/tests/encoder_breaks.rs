//! What compute and render work taking turns costs: the pattern of a title
//! frame as VIctory records it (`d3d12::breaks`), through D3D12/D3DMetal
//! and, for the same turns in Metal itself, through the bridge.
//!
//! `tools/macos/test-wine.sh -p victory-metal --release --test encoder_breaks -- --ignored --nocapture`
//!
//! Run it alone, with no title running: the GPU is shared.
#![cfg(windows)]

use std::time::Duration;

use d3d12::breaks::{Breaks, Pattern};

mod d3d12;

const RUNS: usize = 5;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
#[ignore = "a benchmark: run it alone, with --nocapture"]
fn turns_between_compute_and_render() {
    if victory_metal::Bridge::open().is_err() {
        eprintln!("not under Wine with the bridge: skipped");
        return;
    }
    let bridge = victory_metal::Bridge::open().unwrap();
    println!("Metal through the bridge, one command buffer (medians of {RUNS}):");
    for pairs in [1000, 2000, 3500] {
        let mut times = Vec::new();
        for batched in [false, true] {
            bridge.turns(pairs, batched).unwrap();
            let runs: Vec<_> = (0..RUNS)
                .map(|_| bridge.turns(pairs, batched).unwrap())
                .collect();
            let encode = median(runs.iter().map(|t| t.encode).collect());
            let wait = median(runs.iter().map(|t| t.wait).collect());
            let gpu = median(runs.iter().map(|t| t.gpu).collect());
            let how = if batched { "batched" } else { "taking turns" };
            println!(
                "  {pairs:>5} pairs, {how:<12} encode {:>7.2} ms, commit to done {:>7.2} ms, GPU {:>7.2} ms",
                ms(encode),
                ms(wait),
                ms(gpu)
            );
            times.push((encode, wait, gpu));
        }
        let turns = f64::from(2 * pairs - 2);
        let per = |a: Duration, b: Duration| (a.as_secs_f64() - b.as_secs_f64()) * 1e6 / turns;
        println!(
            "  {pairs:>5} pairs: per extra turn: encode {:.2} us, commit to done {:.2} us, GPU {:.2} us",
            per(times[0].0, times[1].0),
            per(times[0].1, times[1].1),
            per(times[0].2, times[1].2),
        );
    }
    let bench = d3d12::Bench::new();
    let breaks = Breaks::new(&bench);
    println!(
        "D3D12/D3DMetal, one submission (medians of {RUNS}; record = this thread, submit = to the fence):"
    );
    for pairs in [1000, 2000, 3500] {
        let mut times = Vec::new();
        for pattern in [
            Pattern::Interleaved,
            Pattern::Batched,
            Pattern::InterleavedPlain,
            Pattern::InterleavedUnbarred,
            Pattern::DrawsOnly,
        ] {
            breaks.run(pattern, pairs);
            let runs: Vec<_> = (0..RUNS).map(|_| breaks.run(pattern, pairs)).collect();
            let record = median(runs.iter().map(|f| f.record).collect());
            let submit = median(runs.iter().map(|f| f.execute).collect());
            println!(
                "  {pairs:>5} pairs, {pattern:<19?} record {:>7.2} ms, submit {:>7.2} ms",
                ms(record),
                ms(submit)
            );
            times.push((record, submit));
        }
        // Interleaved turns 2 * pairs - 1 times between compute and render
        // work; batched once.
        let turns = f64::from(2 * pairs - 2);
        let per = |a: Duration, b: Duration| (a.as_secs_f64() - b.as_secs_f64()) * 1e6 / turns;
        println!(
            "  {pairs:>5} pairs: per extra turn, indirect: record {:.2} us, submit {:.2} us; \
             over plain draws alone: with UAV barriers record {:.2} us, submit {:.2} us, \
             with no barriers record {:.2} us, submit {:.2} us",
            per(times[0].0, times[1].0),
            per(times[0].1, times[1].1),
            per(times[2].0, times[4].0),
            per(times[2].1, times[4].1),
            per(times[3].0, times[4].0),
            per(times[3].1, times[4].1),
        );
    }
}
