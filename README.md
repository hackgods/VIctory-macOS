# VIctory for macOS: Apple Silicon fixes

<img width="1600" alt="GTA VI on an M4 Pro: player control in the prison cell at sunset, VIctory's fps line in the window title" src="https://github.com/user-attachments/assets/3ef3c9ee-b506-40e6-a2e9-7ec69acc906a" />

Runs the Xbox Series build of GTA VI on an Apple Silicon Mac (Wine plus Apple's D3DMetal), from a copy of the game that you own. It combines our own Mac fixes with fixes from the Windows forks, and every change is measured on a MacBook Pro M4 Pro (48 GB).

**Source code only, nothing from the game.** You need your own copy of the game.

## Status (27 November 2026)

**It works:**
- The game loads the world and gives player control. Saved games load.
- Sunlight and shadows are right, rooms have fill light, and characters and clothes draw.

**Speed:**

| Place | Series X profile | With 1920x1080 output |
|---|---|---|
| Prison checkpoint | 2.1–2.6 fps | 2.8 fps |
| Boat | 4.7 fps | 5.6 fps |

**Not yet:**
- Playable speed. The next goal is 5 fps, then 10 fps at 1080p.
- Dark car interiors at night, and one black patch on water.

Recent changes: [WHATS-NEW.md](WHATS-NEW.md).

## Tested on

| Mac | Memory | Speed | Note | When (UTC) | Build |
|---|---|---|---|---|---|
| MacBook Pro M4 Pro | 48 GB | 2.5 fps prison, 4.7 boat (2.8 / 5.6 at 1080p) | Our test Mac | 26 Nov 23:30 | [e35196a](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/e35196aaf26c) |
| MacBook Pro M5 Pro | 24 GB | 2.6 fps prison cell (1080p render) | "By a wide margin the best picture and the best speed I have had on this Mac" ([report](https://github.com/pragyaangaur/VIctory-macOS/issues/455#issuecomment-7206500809)) | 27 Nov 00:29 | 8616bd6 + 2 memory fixes |
| MacBook Air M5 | 16 GB | 1.9 fps at the start (Series S), slowing to 0.25 fps over a 20-minute run | "The best result I've seen on my 16 GB M5 MacBook Air" ([report](https://github.com/pragyaangaur/VIctory-macOS/issues/455)) | 27 Nov 01:24 | 8616bd6 + memory marks (#456) |
| MacBook Pro M2 Pro | 32 GB | 1.3 fps prison cell (2.7 fps at 1080p) | "The picture is very close to your screenshot" ([report](https://github.com/halfwave50/VIctory-macOS-halfwave/issues/1#issuecomment-7207381523)) | 27 Nov 01:33 | ee60b34 + upstream #445 |

Tried it on your Mac? Post your Mac, memory and fps in [issue #1](https://github.com/halfwave50/VIctory-macOS-halfwave/issues/1) and we'll add a row.

<img width="1608" alt="Four scenes on an M4 Pro: beach at sunset, boat in daylight, night street, apartment" src="https://github.com/user-attachments/assets/c89e61c0-9552-4217-b547-7fa180598d32" />

More pictures, questions and results from other Macs: [issue #1](https://github.com/halfwave50/VIctory-macOS-halfwave/issues/1).

## Try it

You need:
- An Apple Silicon Mac. Confirmed working with 16 GB (MacBook Air M5) and up; see "Tested on".
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
| Slow buffer copies | Many of the game's buffers are shared with the GPU instead of copied every frame (`VICTORY_ALIAS_BUFFERS=1`) | [f517ed2](https://github.com/halfwave50/VIctory-macOS-halfwave/commit/f517ed2f43e1) |

## Develop or fork it

[DEVELOPING.md](DEVELOPING.md) covers how the work is done: one fix per commit with its source named, new code behind switches, measuring, and the asset rule. AI coding tools can start from [AGENTS.md](AGENTS.md).

## Credits and licence

This builds on:
- [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
- [VIctory for macOS](https://github.com/pragyaangaur/VIctory-macOS) by pragyaangaur and its contributors. Its own README has more on how it works, testing and debugging switches.
- The forks by hyphennn, Belobug, DexNom, myokex, avatarmanz, 64bitMango, darkworldhm and trustno1-coder

The licence is GPL-3.0 ([LICENSE](LICENSE)).
