#!/bin/sh
# Builds victory-metal's bridge to Metal (crates/victory-metal, #221): the
# anchor DLL, marked as a Wine builtin, and the unixlib Wine pairs it with.
#   tools/macos/build-metal-bridge.sh [output folder]
# The output folder (target/metal-bridge by default) gets
#   x86_64-windows/victory_metal.dll   the anchor
#   x86_64-unix/victory_metal.so       the unixlib (a Mach-O library, x86_64)
# A Wine started with the folder in WINEDLLPATH finds the pair; set
# VICTORY_METAL_BRIDGE to the anchor's Z: path for the Windows side to load
# it. tools/macos/test-wine.sh does both. Everything is built from this
# repository's sources with Xcode's clang and the pinned Rust toolchain.
set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
OUT="${1:-$ROOT/target/metal-bridge}"
SRC="$ROOT/crates/victory-metal/unix"
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
SO="$OUT/x86_64-unix/victory_metal.so"
DLL="$OUT/x86_64-windows/victory_metal.dll"
# Nothing to do when both are newer than their sources and this script.
fresh=1
for src in "$SRC/victory_metal.m" "$SRC/abi.h" "$SRC/anchor.rs" "$ROOT/tools/macos/build-metal-bridge.sh"; do
    { [ "$SO" -nt "$src" ] && [ "$DLL" -nt "$src" ]; } || fresh=0
done
[ "$fresh" = 0 ] || exit 0
mkdir -p "$OUT/x86_64-windows" "$OUT/x86_64-unix"
# x86_64: the Wine process runs under Rosetta, and a unixlib is loaded into
# it. Metal itself is the system's, so the oldest macOS D3DMetal runs on is
# old enough.
xcrun clang -arch x86_64 -mmacosx-version-min=13.0 -dynamiclib -fobjc-arc -O2 \
    -Wall -Wextra -Werror -framework Foundation -framework Metal \
    -install_name @rpath/victory_metal.so \
    -o "$SO.new" "$SRC/victory_metal.m"
mv "$SO.new" "$SO"
WORK="$OUT/anchor-build"
mkdir -p "$WORK"
rustc --edition 2024 --crate-type cdylib --crate-name victory_metal \
    --target x86_64-pc-windows-gnullvm -C panic=abort -C opt-level=s -C strip=symbols \
    -C linker="$ROOT/tools/macos/gnullvm-cc.sh" --out-dir "$WORK" "$SRC/anchor.rs"
# Wine takes a PE file for a builtin when "Wine builtin DLL" follows the
# 64-byte DOS header (its server reads the 32 bytes after it). The linker
# puts the PE header further on (e_lfanew), so this only replaces the DOS
# stub's message.
lfanew="$(od -An -t u4 -j 60 -N 4 "$WORK/victory_metal.dll" | tr -d ' ')"
[ "$lfanew" -ge 96 ] || { echo "the anchor's PE header starts at $lfanew, inside the mark" >&2; exit 1; }
printf 'Wine builtin DLL\000' | dd of="$WORK/victory_metal.dll" bs=1 seek=64 conv=notrunc 2> /dev/null
mv "$WORK/victory_metal.dll" "$DLL"
rm -rf "$WORK"
