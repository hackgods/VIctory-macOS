#!/bin/sh
# Runs a captured frame again (docs/replay.md) on macOS: builds VIctory for
# Windows and runs `victory replay` under CrossOver Wine with Apple's
# D3DMetal, in a Wine prefix of its own, never the one a title runs in.
#   tools/macos/replay.sh <capture folder> [replay arguments]
# Paths among the arguments (the capture, --shaders, --retranslate, --image)
# are macOS paths; they are handed to Wine as Z: paths.
# WINE defaults to the Wine tools/macos/victory.sh takes, and the build is
# the one it makes (the same clamp compiler), so the two do not build in
# turn; WINEPREFIX to target/wineprefix-replay. VICTORY_DIAGNOSTICS=1 runs
# the build with the diagnostics feature. The VICTORY_* switches of the
# translator are passed on (for --retranslate).
# The GPU is shared: do not run this while a title runs.
set -e
case "${WINE:-}" in
    ''|/*) ;;
    *) WINE="$PWD/$WINE" ;;
esac
cd "$(dirname "$0")/../.."
ROOT="$PWD"
TARGET=x86_64-pc-windows-gnullvm
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
if [ -z "${WINE:-}" ]; then
    WINE="$ROOT/target/wine-cx/bin/wine"
    [ -x "$WINE" ] || WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine"
fi
[ -x "$WINE" ] || { echo "no Wine at $WINE: run tools/macos/build-crossover-wine.sh first" >&2; exit 1; }
[ $# -ge 1 ] || { echo "usage: tools/macos/replay.sh <capture folder> [replay arguments]" >&2; exit 1; }
# As tools/macos/victory.sh: the argument clamp compiled with D3DMetal's DXC
# when there is one.
DXC_LIB="$(dirname "$WINE")/../lib/external/D3DMetal.framework/Versions/A/Resources/libdxcompiler.dylib"
if [ -z "${VICTORY_HELPER_DXC:-}" ] && [ -f "$DXC_LIB" ]; then
    helper="$ROOT/target/dxc-d3dmetal"
    mkdir -p "$ROOT/target"
    if [ ! -x "$helper" ] || [ tools/macos/dxc-d3dmetal.sh -nt "$helper" ]; then
        sh tools/macos/dxc-d3dmetal.sh "$helper" || helper=
    fi
    if [ -n "$helper" ]; then
        export DXC_LIB VICTORY_HELPER_DXC="$helper"
    fi
fi
if [ "${VICTORY_DIAGNOSTICS:-0}" = 1 ]; then
    cargo build --quiet --release -p victory --features diagnostics --target "$TARGET"
else
    cargo build --quiet --release -p victory --target "$TARGET"
fi
metadata=$(cargo metadata --format-version 1 --no-deps)
target_dir=$(printf '%s\n' "$metadata" | /usr/bin/plutil -extract target_directory raw -o - -)
export WINEPREFIX="${WINEPREFIX:-$ROOT/target/wineprefix-replay}"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export ROSETTA_ADVERTISE_AVX=1
EXE="$target_dir/$TARGET/release/victory.exe"
# An argument that is an absolute macOS path becomes a Z: path.
set -- replay "$@"
n=$#
while [ "$n" -gt 0 ]; do
    a=$1
    shift
    case "$a" in
        /*) a="Z:$(printf '%s' "$a" | tr / '\\')" ;;
    esac
    set -- "$@" "$a"
    n=$((n - 1))
done
exec "$WINE" "$EXE" "$@"
