# What's new on `main-mac`

The `main-mac` branch runs VIctory on Apple Silicon. It starts from VIctory for macOS (Wine plus D3DMetal) and adds fixes from the Windows forks that already run the game well. Every change is tested on one MacBook Pro (M4 Pro, 48 GB) with the Xbox Series X profile.

This repository has source code only. It contains nothing from the game. Bring your own copy of the game and build the code yourself.

## Where it stands (27 November 2026)

- **Picture.** The sun now lights surfaces. Shadows have clean edges, rooms have fill light, characters and their clothes draw, and screen reflections reach the picture. The picture is close to the console at the checkpoints we compare against. It is not the same everywhere: dark car interiors at night and a black patch on water in one boat shot are still open, and one ray-traced reflection pass is left out for now.
- **Free roam.** From a free-roam save, the house, the street and the interiors look right, at about 2 fps. Some objects flicker black for one frame, mostly indoors. The game caps each frame's time step at 1/15 s, so at 2 to 4 fps the world ran at 15 to 25% of normal speed; the launcher now allows 0.25 s. Held buttons (doors) worked; short taps (sprint, climb) were lost, and the launcher now keeps them. Neither fix is tried in play yet.
- **Speed.** On the default launcher settings:

  | Place | Frames a minute | About |
  |---|---|---|
  | Prison checkpoint | 173 to 184 | 2.9–3.1 fps |
  | Boat | 279 | 4.7 fps |
  | Prison, 1920x1080 output | 179 | 3.0 fps |
  | Boat, 1920x1080 output | 361 | 6.0 fps |

  The prison rows come from three runs on the newest build. Runs of the same settings at the prison can differ by 10 to 30%, because in some runs the game submits its GPU work in more pieces; we now compare four runs of each setting. The 1920x1080 rows were measured just before the census change below. At 1920x1080 the game's interface is drawn twice too large. `VICTORY_OUTPUT=3840x2160 VICTORY_RENDER_SCALE=0.5` keeps it right but is 30% slower in the cell. For about half of each frame, the thread that submits GPU work waits for the GPU. Preparing work and GPU time don't overlap yet.
- **Goal.** First, 5 fps on the Series X profile with the picture right. Then 10 fps at 1080p output. 30 fps is out of reach on this hardware with any code known today.

## Recent changes

- **Screen reflections.** The game writes its reflection image through one texture and the lighting reads the same memory through a second view in another format. The port gave the two views separate GPU textures and copied nothing between them, so no reflection ever reached the picture: steel looked black and water reflected nothing. The view now uses the written texture (`VICTORY_PLACED_TWIN=1`), and a minimum reflection level of 0.15 gives reflections some light (`VICTORY_AMBIENT_REFLECT_FLOOR=0.15`). In the prison cell the steel box and plate get light; at the boat the water reflects the sky. One ray-traced reflection pass then drew rectangular blocks on the water. Its results are wrong anyway while the ray-tracing structure is held back, so the launcher leaves it out (`VICTORY_SKIP_PIPELINES=6d7fe7745ab9c6d8`). No speed cost. All three are launcher defaults.
- **Game speed for playing by hand.** `VICTORY_TIME_STEP_MAX=0.25` lets the game step its clock up to 0.25 s a frame, so the world runs closer to real time at low frame rates. A scripted walk in free roam went several times further. It is on in the launcher for playing; measured runs keep the game's own step. `VICTORY_TIME_STEP_MAX=` turns it off.
- **Fewer waits on one lock.** Threads that write GPU descriptors waited on a lock that guards the list of what each shader resource slot reads. The list is now updated in batches and checked once per barrier (`VICTORY_CENSUS_FAST=1`, a launcher default). Lock waits fell by half or more, and the prison cell got about 20% faster (no change at the boat). `VICTORY_CENSUS_FAST=0` turns it off.
- **Input for playing by hand** (on in the launcher, not yet tried in play; set any of them to 0 to turn it off):
  - `VICTORY_INPUT_LATCH=1`: the game reads the pad once a frame (250 to 500 ms here), so taps between two readings were lost. They are kept now.
  - `VICTORY_INPUT_REFOCUS=1`: keys held down while another app had focus wait for their release.
  - `VICTORY_STICK_DEADZONE=0.15`: a round dead zone on the sticks.
  - `VICTORY_INPUT_WATCH=1`: a log line a second with what the pad sends.
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
  - Optional switches that are off by default: `VICTORY_SHADER_OPT=64` (console loads read four values at a time, from 64bitMango's fork), `VICTORY_BVH_MISS` (from trustno1-coder's fork), and the ones listed below.

## What we tried that didn't help

Runs vary a lot, so some of these may get a second look.

- `VICTORY_OPENED_FAULTS`, `VICTORY_VETTED_PAGES` and `VICTORY_WATCH_RUNS` together (fewer lock waits and Wine calls when memory pages are guarded; two ideas from 64bitMango's fork): 104 and 148 frames a minute at the cell against 136 and 152.
- `VICTORY_HALVES_LDS_ORDER=1` (an exact order for one hair-shadow program's shared memory): 2% slower.
- `VICTORY_ASYNC_SETTLE_ALL=0`: no change, so it stays on.
- `VICTORY_UI_SIZE` (another size for the game's largest display mode): no visible effect.
- `VICTORY_SHADER_OPT=64`: about 3% faster in the prison cell, within run-to-run noise. Stays off for now.
- `VICTORY_BVH_MISS=1` (every ray-tracing node test misses): the blocks on water stayed.
- `VICTORY_UAV_SIZE_LEVEL0=1` (the image-size fix for every program instead of one): faces are clean now, but most runs went into the game's slower mode. It stays on one program.
- `VICTORY_FLIGHTS` gave no gain: the waiting just moved somewhere else.
- `VICTORY_DISPATCH_CLAMPS` cut clamp dispatches from 993 to 141 a frame but was no faster (139 frames a minute against 144 and 148).
- `VICTORY_COPY_BACK_SPLIT` left copying back at 42 ms a frame (146 frames a minute).
- `VICTORY_TIME_STEP_MAX=1` breaks physics at the boat. 0.25 keeps the boat whole and is the launcher's value.

## Credits

This builds on the work of:

- Marvisak (VIctory)
- pragyaangaur and the contributors to VIctory for macOS
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder, and the fault write-ups by ZSlayerHQ

Each borrowed commit names its source at the start of its message. The licence is unchanged (see `LICENSE`).
