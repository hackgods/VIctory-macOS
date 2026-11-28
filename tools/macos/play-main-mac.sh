#!/bin/sh
# Starts the game with the settings main-mac is measured with (see README.md).
# Usage: tools/macos/play-main-mac.sh <the game's executable> [more victory arguments]
# Any switch you set yourself before running this wins over the defaults below.
[ -n "$1" ] || { echo "usage: $0 <the game's executable>" >&2; exit 2; }
exe="$1"; shift
d() { eval "[ -n \"\${$1+x}\" ]" || export "$1=$2"; }
d VICTORY_PATCH_MASKED all;        d VICTORY_FAULT_TRACKING 1
d VICTORY_HALVES_LOCAL 1;          d VICTORY_HALVES_SHARED_LDS 1
d VICTORY_HALVES_EXCHANGE 1;       d VICTORY_HALVES_WALK 1
d VICTORY_GUARD_LOOKUP 1;          d VICTORY_AMBIENT estimate
d VICTORY_MAPDATA_BUDGET 40;       d VICTORY_COPY_ORDER split
d VICTORY_BATCH_CLAMPS 1;          d VICTORY_UAV_SIZE_LEVEL0 07f451a14b13fecb
d VICTORY_TEXTURE_BACK 256x256f41,256x256f16,64x64f16,128x128f36
d VICTORY_HEAP_LOCKS 1;            d VICTORY_HEAP_SYNC_THREADS 4
d VICTORY_ASYNC_SUBMIT 1;          d VICTORY_ASYNC_PREWARM 1
d VICTORY_ASYNC_HEAP_SYNC 1;       d VICTORY_ASYNC_SIGNALS 1
d VICTORY_ASYNC_SETTLE_ALL 1;      d VICTORY_ALIAS_BUFFERS 1
d VICTORY_CENSUS_FAST 1;           d VICTORY_CENSUS_BATCH 1
d VICTORY_SHADER_OPT 64;           d VICTORY_AMBIENT_FLOOR 0.1
# The next submission is prepared while the one before runs on the GPU
# (prison cell, 24 GB M5 Pro: 181 -> 172 ms a frame, 41.3 -> 36.3 ms per 1,000 draws).
d VICTORY_FLIGHTS 1
# A block-compressed texture keeps only its sRGB partner as a cast (#499):
# Metal aborts the process on a view in another compression (BC5 reused as BC7).
d VICTORY_METAL_BC_CASTS 1
# A title copy makes only the pending refreshes its window reaches (#452).
d VICTORY_FLUSH_OVERLAPPING 1
# VIctory-macOS #415, #416, #436, as this stack measured them (always on there):
# unmapped texture pages split in tracked refreshes, only touched rows detiled
# and uploaded, write snapshots that copy only what no earlier declaration took.
d VICTORY_UNMAPPED_SPLIT 1;        d VICTORY_DIRTY_ROWS 1
d VICTORY_SNAPSHOT_SPANS 1
# A shader read of the null page is no title-memory miss (upstream #634): views
# kept out of title memory 81.1 -> 0.0 MB a frame in the prison cell.
d VICTORY_NULL_PAGE 1
d VICTORY_PLACED_TWIN 1;           d VICTORY_AMBIENT_REFLECT_FLOOR 0.15
d VICTORY_SKIP_PIPELINES 6d7fe7745ab9c6d8
d VICTORY_HALVES_LDS_ROUNDS 1;     d VICTORY_PLACED_AT_ONCE keep
d VICTORY_INPUT_LATCH 1;           d VICTORY_INPUT_REFOCUS 1
d VICTORY_INPUT_WATCH 1;           d VICTORY_STICK_DEADZONE 0.15
d VICTORY_TIME_STEP_MAX 0.25;     d VICTORY_DESCRIPTOR_AS_DATA 1
d VICTORY_PAD_REARM 1;            d VICTORY_PAD_RUMBLE 0
# Presses Enter on the first screens (20, 30 and 40 s in). Set VICTORY_KEYS= to turn it off.
d VICTORY_KEYS 20:Enter,30:Enter,40:Enter
# Faster, lower resolution: VICTORY_OUTPUT=1920x1080
exec "$(dirname "$0")/victory.sh" run --execute "$exe" "$@"
