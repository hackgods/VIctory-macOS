# What's new on `main-mac`

The `main-mac` branch runs VIctory on Apple Silicon. It starts from VIctory for macOS (Wine plus D3DMetal) and adds fixes from the Windows forks that already run the game well. Every change is tested on one MacBook Pro (M4 Pro, 48 GB) with the Xbox Series X profile.

This repository has source code only. It contains nothing from the game. Bring your own copy of the game and build the code yourself.

## Where it stands (27 November 2026)

- **Picture.** The sun now lights surfaces. Shadows have clean edges, rooms have fill light, and characters and their clothes draw. The picture is close to the console at the checkpoints we compare against. It is not the same everywhere: dark car interiors at night and a black patch on water in one boat shot are still open.
- **Speed.** On the default launcher settings:

  | Place | Frames a minute | About |
  |---|---|---|
  | Prison checkpoint | 146 | 2.4 fps |
  | Boat | 255 | 4.3 fps |
  | Prison, 1920x1080 output | 170 | 2.8 fps |
  | Boat, 1920x1080 output | 335 | 5.6 fps |

  Most of the frame is still the time spent preparing GPU work, plus the GPU time itself. Neither yet overlaps the other.
- **Goal.** First, 5 fps on the Series X profile with the picture right. Then 10 fps at 1080p output. 30 fps is out of reach on this hardware with any code known today.

## Recent changes

- **Direct sunlight.** The lighting multiplies the sun by a texture that a compute pass writes. D3DMetal cannot store to that texture's format (B5G5R5A1), so the writes were lost and the sun came out black. The fix gives such textures a format that Metal can store (taken from Belobug's AMD fork).
- **Black bands in sunlit areas.** Four tile-list programs now run in their exact form (`VICTORY_HALVES_WALK=1`, a launcher default).
- **Fill light.** The console lights rooms by tracing rays. Here a stand-in estimates the light from the scene, with a minimum level so dark rooms don't go black (`VICTORY_AMBIENT=estimate`, `VICTORY_AMBIENT_FLOOR`, default 0.3).
- **Characters and clothes.** About 60 character pipelines had vertex data a little over one of Apple's size limits, so they were dropped without a warning. They draw now.
- **Blocky sun edges in the prison yard (fix being tested).** One shadow program asks a writable image for its size and gets 0, so whole 16x16 tiles were lit or shadowed together. `VICTORY_UAV_SIZE_LEVEL0=07f451a14b13fecb` gives that one program the right size.
- **Tools.**
  - Frame replay plays back one recorded frame without the game, so GPU changes can be measured in seconds.
  - A watch on the request list.
  - Optional switches that are off by default while they are measured: `VICTORY_TIME_STEP_MAX`, `VICTORY_FLIGHTS`, `VICTORY_ALIAS_BUFFERS`.

## What we tried that didn't help

- `VICTORY_FLIGHTS` gave no gain: the waiting just moved somewhere else.
- `VICTORY_TIME_STEP_MAX=1` breaks physics at the boat. 0.25 looks safe but is not yet measured.

## Credits

This builds on the work of:

- Marvisak (VIctory)
- pragyaangaur and the contributors to VIctory for macOS
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder, and the fault write-ups by ZSlayerHQ

Each borrowed commit names its source at the start of its message. The licence is unchanged (see `LICENSE`).
