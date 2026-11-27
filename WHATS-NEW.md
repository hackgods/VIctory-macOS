# What's new on `main-mac`

The `main-mac` branch runs VIctory on Apple Silicon. It starts from VIctory for macOS (Wine plus D3DMetal) and adds fixes from the Windows forks that already run the game well. Every change is tested on one MacBook Pro (M4 Pro, 48 GB) with the Xbox Series X profile.

This repository has source code only. It contains nothing from the game. Bring your own copy of the game and build the code yourself.

## Where it stands (27 November 2026)

- **Picture.** The sun now lights surfaces. Shadows have clean edges, rooms have fill light, and characters and their clothes draw. The picture is close to the console at the checkpoints we compare against. It is not the same everywhere: dark car interiors at night and a black patch on water in one boat shot are still open.
- **Speed.** On the default launcher settings:

  | Place | Frames a minute | About |
  |---|---|---|
  | Prison checkpoint | 125 to 156 | 2.1–2.6 fps |
  | Boat | 280 | 4.7 fps |
  | Prison, 1920x1080 output | 170 | 2.8 fps |
  | Boat, 1920x1080 output | 335 | 5.6 fps |

  The prison runs gave 147 and 156 on the previous build and 125 and 136 on the newest one; more runs are queued. The 1920x1080 rows were measured before zero-copy buffers. Most of the frame is still the time spent preparing GPU work, plus the GPU time itself. Neither yet overlaps the other.
- **Goal.** First, 5 fps on the Series X profile with the picture right. Then 10 fps at 1080p output. 30 fps is out of reach on this hardware with any code known today.

## Recent changes

- **Zero-copy buffers.** Many of the game's buffers are now shared with the GPU instead of being copied every frame (`VICTORY_ALIAS_BUFFERS=1`, a launcher default). Copies fell from about 372 to 292 MB a frame, and the boat went from 257 to 280 frames a minute. It uses 1 to 2 GB more memory. `VICTORY_ALIAS_BUFFERS=0` turns it off.
- **Direct sunlight.** The lighting multiplies the sun by a texture that a compute pass writes. D3DMetal cannot store to that texture's format (B5G5R5A1), so the writes were lost and the sun came out black. The fix gives such textures a format that Metal can store (taken from Belobug's AMD fork).
- **Black bands in sunlit areas.** Four tile-list programs now run in their exact form (`VICTORY_HALVES_WALK=1`, a launcher default).
- **Fill light.** The console lights rooms by tracing rays. Here a stand-in estimates the light from the scene, with a minimum level so dark rooms don't go black (`VICTORY_AMBIENT=estimate`, `VICTORY_AMBIENT_FLOOR`, default 0.3).
- **Characters and clothes.** About 60 character pipelines had vertex data a little over one of Apple's size limits, so they were dropped without a warning. They draw now.
- **Blocky sun edges in the prison yard.** One shadow program asks a writable image for its size and gets 0, so whole 16x16 tiles were lit or shadowed together. `VICTORY_UAV_SIZE_LEVEL0=07f451a14b13fecb` gives that one program the right size. It is a launcher default now.
- **Tools.**
  - Frame replay plays back one recorded frame without the game, so GPU changes can be measured in seconds.
  - A watch on the request list.
  - A sampling profiler of the game's threads (`VICTORY_PROFILE`, from Belobug's fork). It now tells threads that wait inside Wine, on registry watches or on file reads from threads that run.
  - The per-frame log line also counts waits for the page-guard lock, time in memory-protection calls and the exceptions the write-fault handler sees.
  - Optional switches that are off by default while they are measured: `VICTORY_TIME_STEP_MAX`, `VICTORY_CENSUS_FAST` (a cheaper lock on the list of what each shader resource slot reads), and `VICTORY_OPENED_FAULTS`, `VICTORY_VETTED_PAGES`, `VICTORY_WATCH_RUNS` (fewer lock waits and Wine calls when memory pages are guarded; two ideas from 64bitMango's fork). No verdict on them yet.

## What we tried that didn't help

- `VICTORY_FLIGHTS` gave no gain: the waiting just moved somewhere else.
- `VICTORY_DISPATCH_CLAMPS` cut clamp dispatches from 993 to 141 a frame but was no faster (139 frames a minute against 144 and 148).
- `VICTORY_COPY_BACK_SPLIT` left copying back at 42 ms a frame (146 frames a minute).
- `VICTORY_TIME_STEP_MAX=1` breaks physics at the boat. 0.25 looks safe but is not yet measured.

## Credits

This builds on the work of:

- Marvisak (VIctory)
- pragyaangaur and the contributors to VIctory for macOS
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder, and the fault write-ups by ZSlayerHQ

Each borrowed commit names its source at the start of its message. The licence is unchanged (see `LICENSE`).
