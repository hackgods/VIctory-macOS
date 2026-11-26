# VIctory for macOS: Apple Silicon fixes

Runs the Xbox Series build of GTA VI on an Apple Silicon Mac, from a copy of the game that you own. This is a fork of [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) (Wine plus Apple's D3DMetal). It adds fixes from the Windows forks plus our own Mac fixes, and every change is measured on a MacBook Pro M4 Pro (48 GB).

**Source code only, nothing from the game.** You need your own copy of the game.

## Status (27 November 2026)

**It works:**
- The game loads the world and gives player control. Saved games load.
- Sunlight and shadows are right, rooms have fill light, and characters and clothes draw.

**Speed:**

| Place | Series X profile | With 1920x1080 output |
|---|---|---|
| Prison checkpoint | 2.4 fps | 2.8 fps |
| Boat | 4.3 fps | 5.6 fps |

**Not yet:**
- Playable speed. The next goal is 5 fps, then 10 fps at 1080p.
- Dark car interiors at night, and one black patch on water.

Recent changes: [WHATS-NEW.md](WHATS-NEW.md).

## Try it

You need:
- An Apple Silicon Mac. 32 GB or more is best: a run uses about 20 GB.
- macOS 15 or newer.
- [Homebrew](https://brew.sh).
- About 15 GB of free space.

1. Get the code:
   ```bash
   git clone https://github.com/halfwave50/VIctory-macOS-halfwave.git
   cd VIctory-macOS-halfwave
   ```
2. Download Apple's **Game Porting Toolkit**, version 4.0 beta 3 or newer, from [Apple's developer downloads](https://developer.apple.com/download/all/?q=game%20porting%20toolkit). A free Apple account works. Leave the `.dmg` in Downloads, unopened.
3. Build everything:
   ```bash
   ./setup.sh
   ```
   The first run takes about an hour. If it stops, fix what it says and run it again.
4. Point `game_dir` in `victory.toml` at your game folder. Every package's files go directly in that folder.
5. Start the game with our settings:
   ```bash
   tools/macos/play-main-mac.sh <the game's executable>
   ```

[docs/macos.md](docs/macos.md) explains every Mac fix and problem in detail.

## Main fixes

| Problem | Fix | Commit |
|---|---|---|
| No direct sunlight | Packed 16-bit textures that D3DMetal can't store to are kept as R8G8B8A8 (from Belobug's fork) | [010f0f9](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/010f0f9b9d6a) |
| Characters and clothes missing | Mesh pipelines that were just over Metal's limit now fit | [df9aab4](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/df9aab472803), [f1f590a](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/f1f590a93234) |
| Black bands in sunlight | The tile-list programs run exactly | [e566e1d](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/e566e1d14b7e) |
| Dark rooms | Estimated fill light with a minimum level (`VICTORY_AMBIENT=estimate`) | [9158460](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/915846063859) |
| Blocky shadow edges | One shadow program gets its image size (`VICTORY_UAV_SIZE_LEVEL0`) | in `play-main-mac.sh` |

## Develop or fork it

[DEVELOPING.md](DEVELOPING.md) covers how the work is done: one fix per commit with its source named, new code behind switches, measuring, and the asset rule. AI coding tools can start from [AGENTS.md](AGENTS.md).

## Credits and licence

This builds on:
- [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
- [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by pragyaangaur and its contributors. Its own README has more on how it works, testing and debugging switches.
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder

The licence is GPL-3.0 ([LICENSE](LICENSE)).
