# Host GPU backends

`victory-gpu` translates the console's D3D12.X onto a host GPU. Until now that host was always D3D12: on Windows directly, and on a Mac through D3DMetal. Issue #221 proposes a second host, Metal, reached from Windows code through a Wine unixlib. The point is that Apple Silicon's memory is unified, as the console's is, so Metal can put buffers over title memory itself (`newBufferWithBytesNoCopy`). That removes the host copies of title memory, the guards that track them and the copies back, which D3DMetal cannot do (#197).

The step 0 probe for that work, which needs no game files, is in issue #221.

## The traits

`crates/victory-gpu/src/backend.rs` names what the platform code asks of the host GPU, as traits. `crate::host::Host`, the D3D12 host, implements them. D3D12 stays the first implementation and the reference that tests compare against. Each step below moves code behind a trait unchanged, so nothing changes for Windows or for the Wine and D3DMetal route.

| Trait | What it covers | Status |
| --- | --- | --- |
| `HostGpu` | What the host can run (mesh shaders, waves of 32, 64-bit heap atomics), new command lists, holding submissions off, and running a submission | Done |
| `HostMemory` | Host buffers by how the CPU and GPU share them: GPU-only, upload, readback, and shared (host copies of title memory) | Done |

## What comes next, in order

1. **Recording commands.** `HostList::list` is still a D3D12 command list, and about 200 places record into it directly, most of them in `draw.rs`, `memory.rs` and `widened.rs`. The first commands to move are copies and barriers, because the memory path is what Metal changes first. Draws, dispatches and bindings come after them.
2. **Title memory in place.** `HostMemory` gains a buffer over title memory itself, plus residency for it. On D3D12 that is `OpenExistingHeapFromAddress` where the host has it, the `mach_vm_remap` alias of #207 under D3DMetal, or nothing. On Metal it is a no-copy buffer, with all its buffers in a residency set. The probe in #221 showed that the GPU does not keep two of its addresses for one physical page coherent within a submission. Memory the title maps at several addresses therefore has to reach the GPU through one buffer, and `title_memory_aliased` already reports where that happens.
3. **Descriptors and root signatures**, as Metal argument buffers in the layout Apple's Metal Shader Converter expects.
4. **Pipelines.** DXIL from `victory-shader` goes through the Metal Shader Converter, and pipelines are kept in an `MTLBinaryArchive`, built ahead of time by `victory translate --cache --pipelines`.
5. **Textures** (detiling on the GPU from title memory) **and presentation.**
6. **Choosing the host** with a switch. D3D12 stays the default.

The Metal backend itself starts after step 2. It is first useful for title buffers, which is where the copies are.

## From Windows code to Metal: `victory-metal`

VIctory runs as a Windows program under Wine, so a Metal host needs a way from Windows code into macOS code. `crates/victory-metal` is that way, made as Wine's own builtin DLLs reach their macOS halves, and as D3DMetal's `d3d12.dll` reaches its own: a Wine unixlib.

- `unix/anchor.rs` is an empty DLL that `tools/macos/build-metal-bridge.sh` marks as a Wine builtin, and `unix/victory_metal.m` is a Mach-O library of the same name (x86_64, since the Wine process runs under Rosetta). With their folder in `WINEDLLPATH`, Wine pairs the two when the DLL is loaded, and `NtQueryVirtualMemory` with `MemoryWineUnixFuncs` hands the Windows side the library's call table. `Bridge` makes the calls through `__wine_unix_call_dispatcher`.
- A call runs on the calling thread, in the same process, so Windows memory has the same address on the macOS side. `newBufferWithBytesNoCopy` over `VirtualAlloc` memory lets the GPU write that memory in place, and the Windows side reads the result where it is.
- Each call takes one block of arguments that starts with its size (`unix/abi.h`, mirrored in `src/abi.rs`), so a unixlib built from another layout refuses the call.
- `tools/macos/test-wine.sh` builds the pair and sets `WINEDLLPATH` and `VICTORY_METAL_BRIDGE` for the tests. Nothing else uses the bridge yet.

What the calls cost, on an M5 Pro (macOS 27.2, CrossOver Wine 26.3, release build), from `bridge_against_d3d12` in `crates/victory-metal/tests/bridge.rs` and the same unixlib called without Wine:

| | Through the bridge to Metal | Through D3D12 to D3DMetal |
| --- | --- | --- |
| A call into macOS code and back (no-op) | 15–16 ns | |
| An empty submission, waited for | 22.8 µs | 52.4 µs |
| 5,000 dispatches writing 20 MB, one submission, wall | 5.5 ms (recording 0.58 ms, GPU 4.24 ms) | 6.0 ms with one root UAV, 7.5 ms with a view table bound per dispatch |
| Recording one dispatch, the unixlib's code without Wine | 116–129 ns under Rosetta, 127–174 ns native | |
| One dispatch with a heap of 20,000 buffers or 1,000,000 slots bound | | 0.26–0.77 ms, not growing with either |

What this says for the Metal host:

- Calls are cheap enough to make one per command: a frame's 50,000 or so would take under 1 ms. (D3DMetal's list calls cost about the same to record: 10,000 in 0.15 ms.) A command stream in memory both sides see, with one call per list or submission, still keeps the Windows side free of Metal's objects and lets recording run on the title's threads.
- Metal's own recording runs as fast under Rosetta as natively, so the unixlib can stay in the Wine process: a helper process of its own would gain nothing there and would need title memory shared across processes.
- For simple compute work D3DMetal adds little: 0.1–0.5 µs a dispatch and 30 µs a submission. What costs is compute and render work taking turns across a barrier, the pattern VIctory's inline argument clamps make (below).

### Compute and render work taking turns

`crates/victory-metal/tests/encoder_breaks.rs` runs a title frame's pattern: a dispatch of one thread writes an indirect draw's 20 bytes of arguments, a barrier, the draw through `ExecuteIndirect` into a render target, again and again in one list; against the same work batched (all dispatches, one barrier, all draws), and the same turns in Metal itself through the bridge. Each turn is one more encoder on Metal. Per turn, on the same M5 Pro, over 1,000 to 3,500 pairs in one list (medians of five):

| | Recording (CPU) | Submission to completion |
| --- | --- | --- |
| D3D12/D3DMetal, transition barriers and indirect draws | 12–38 µs, growing with the turns already in the list | 16–20 µs |
| D3D12/D3DMetal, UAV barriers and plain draws | 11–29 µs, growing the same way | 12.5–14.7 µs |
| D3D12/D3DMetal, the same turns with no barrier | 0.03 µs | 0.2 µs |
| Metal through the bridge, an encoder each | 6.4–7.2 µs | 9.9–10.7 µs, nearly all of it GPU time |
| Batched, either way | under 0.5 µs a pair | under 1.3 µs a pair |

So the cost is the barrier between compute and render work, not the switch itself: without one, D3DMetal keeps the work together. With one, recording a list in D3DMetal takes time that grows with the square of its turns (25, 96 and 263 ms for 1,000, 2,000 and 3,500 pairs), and even Metal itself needs about 10 µs of GPU time a turn to finish the one encoder before the next starts. A Metal host would halve the cost of each turn; batching the dispatches ahead of the draws removes nearly all of it on either host.
