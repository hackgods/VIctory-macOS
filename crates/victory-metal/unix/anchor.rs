//! The anchor: an empty DLL that tools/macos/build-metal-bridge.sh marks as
//! a Wine builtin ("Wine builtin DLL" after the DOS header, where Wine looks
//! for it). Wine maps it as a builtin module and pairs it with the unixlib
//! of the same name in the x86_64-unix folder beside its own folder; the
//! Windows side asks Wine for that unixlib's call table by this module's
//! address. The DLL has no code of its own: the calls are made from
//! victory-metal's `Bridge`.
