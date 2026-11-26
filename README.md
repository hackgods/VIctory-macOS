# VIctory for macOS: Apple Silicon fixes (`main-mac`)

This is a fork of [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS). It uses the same Wine and D3DMetal setup, and adds fixes from the Windows forks plus our own Mac fixes. Every change is measured on a MacBook Pro (M4 Pro, 48 GB) with the Xbox Series X profile.

**This repository has source code only. It contains nothing from the game.** You need your own copy of the game, and you build everything yourself.

## Where it stands (27 November 2026)

- The game starts, loads the world and gives player control. Saved games load.
- **Picture:**
  - Direct sunlight with proper shadows.
  - Fill light in rooms.
  - Characters and clothes draw.
  - No black bands or blocky shadow edges.
- **Speed** on the switches below:

  | Place | Frames a minute | About |
  |---|---|---|
  | Prison checkpoint | 146 | 2.4 fps |
  | Boat | 255 | 4.3 fps |
  | Prison, 1920x1080 output | 170 | 2.8 fps |
  | Boat, 1920x1080 output | 335 | 5.6 fps |

- **Still open:**
  - Dark car interiors at night.
  - A black patch on water in one boat shot.
  - Speed, which is far from playable.

What changed recently: [WHATS-NEW.md](WHATS-NEW.md).

## Main fixes

| Fix | Commit |
|---|---|
| Sunlight: packed 16-bit textures that D3DMetal can't store to are kept as R8G8B8A8 (from Belobug's fork) | [010f0f9](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/010f0f9b9d6a) |
| Characters and clothes: mesh pipelines that were just over Metal's size limit now fit | [df9aab4](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/df9aab472803), [f1f590a](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/f1f590a93234) |
| Black bands in sunlight: the tile-list programs run exactly | [e566e1d](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/e566e1d14b7e) |
| Fill light: an estimate with a minimum level (`VICTORY_AMBIENT=estimate`) | [9158460](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/915846063859) |

Every commit borrowed from another fork names its source at the start of its message.

## Build and run

Build and run it the same way as VIctory for macOS: run `./setup.sh`, and see [docs/macos.md](docs/macos.md). The speed and picture numbers above were measured with these switches:

```sh
export VICTORY_PATCH_MASKED=all VICTORY_FAULT_TRACKING=1
export VICTORY_HALVES_LOCAL=1 VICTORY_HALVES_SHARED_LDS=1 VICTORY_HALVES_EXCHANGE=1 VICTORY_HALVES_WALK=1
export VICTORY_GUARD_LOOKUP=1 VICTORY_AMBIENT=estimate VICTORY_MAPDATA_BUDGET=40
export VICTORY_COPY_ORDER=split VICTORY_BATCH_CLAMPS=1
export VICTORY_TEXTURE_BACK=256x256f41,256x256f16,64x64f16,128x128f36
export VICTORY_UAV_SIZE_LEVEL0=07f451a14b13fecb
export VICTORY_HEAP_LOCKS=1 VICTORY_HEAP_SYNC_THREADS=4
export VICTORY_ASYNC_SUBMIT=1 VICTORY_ASYNC_PREWARM=1 VICTORY_ASYNC_HEAP_SYNC=1 VICTORY_ASYNC_SIGNALS=1 VICTORY_ASYNC_SETTLE_ALL=1
# optional, faster: VICTORY_OUTPUT=1920x1080
```

This branch updates by itself as work goes on.

## Credits and licence

This builds on:

- [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
- [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by pragyaangaur and its contributors
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder

The licence is GPL-3.0, unchanged (see [LICENSE](LICENSE)).
