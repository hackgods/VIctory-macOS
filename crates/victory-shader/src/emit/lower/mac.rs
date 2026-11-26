//! Settings of the Mac line (`emit::Options`), all off by default, in a
//! struct and a file of their own so that a new one needs no new field in
//! the test initialisers and little of `lower.rs`.
//!
//! Early depth-stencil ([`Mac::early_depth`], `emit::Options::early_depth`):
//! a pixel program whose `DB_SHADER_CONTROL` has `DEPTH_BEFORE_SHADER` (bit
//! 12) set gets DXIL's `ForceEarlyDepthStencil` shader flag (bit 3, HLSL's
//! `[earlydepthstencil]`). Both say the same thing: the depth and stencil
//! tests *and their writes* are done before the program runs, a fragment
//! that fails them never runs it (so it writes no image and no colour), and
//! a discard only takes back the colour outputs and the image writes after
//! it, not the depth write. The console's compiler sets the bit for exactly
//! that attribute; Mesa's AMD drivers set it for the same meaning (GL's
//! `early_fragment_tests`), together with `EXEC_ON_NOOP` (bit 10) when the
//! program writes memory and `KILL_ENABLE` (bit 6) when it discards. Grand
//! Theft Auto VI's see-through passes carry it (0x1450 = bits 4, 6, 10, 12
//! in the alpha pre-pass; `notes/hair.md`). Without the flag D3D12 runs a
//! program that writes images or discards before the depth test, so
//! fragments hidden behind opaque geometry record into the pre-pass images.
//!
//! The rule is narrower than "bit 12 set" where the two definitions could
//! differ or the host could refuse the program ([`early_depth`]):
//!
//! * not when the register lets the program export depth, a stencil value
//!   or a coverage mask (bits 0, 1, 2 and 8), nor when the translated
//!   program has such an output (`SV_Depth*`, `SV_StencilRef`,
//!   `SV_Coverage`): with the flag D3D ignores a depth output, and Metal
//!   refuses a function with early tests that returns depth or stencil;
//! * not when the program reads the coverage mask (`SPI_PS_INPUT_ADDR` bit
//!   14): under the flag D3D hands it the coverage left after the depth
//!   test, the console the rasterizer's unless another bit asks for that.
//!
//! No program of the title with bit 12 is in either group (3,023 pixel
//! programs counted, `notes/hair.md`), so there the rule is "bit 12". A
//! discard needs no exclusion: both definitions keep the early depth write
//! of a discarded fragment.
//!
//! The register's bit names are AMD's (as in Mesa's `sid.h`), written from
//! memory: the tree has no table of them.

use super::*;

/// What the Mac line's switches ask of a translation; all off (the
/// default): the translation is what it was without them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Mac {
    /// Pixel programs that ask for the depth test before the shader get
    /// the early depth-stencil flag (`emit::Options::early_depth`; the
    /// module notes).
    pub early_depth: bool,
}

/// `DB_SHADER_CONTROL`: the program exports depth (`Z_EXPORT_ENABLE`).
const Z_EXPORT: u32 = 1 << 0;
/// The program exports a stencil test value or a stencil operation value.
const STENCIL_EXPORTS: u32 = 1 << 1 | 1 << 2;
/// The program exports a coverage mask (`MASK_EXPORT_ENABLE`).
const MASK_EXPORT: u32 = 1 << 8;
/// `DEPTH_BEFORE_SHADER`: depth and stencil are tested and written before
/// the program runs.
const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
/// `SPI_PS_INPUT_ADDR`: the program reads the coverage mask
/// (`io::Input::SampleCoverage`).
const COVERAGE_INPUT: u32 = 1 << 14;

/// Whether a pixel program with these registers and translated outputs is
/// one whose early depth test D3D's flag states exactly (the module notes).
pub(crate) fn early_depth(db_shader_control: u32, input_addr: u32, outputs: &[SigElement]) -> bool {
    db_shader_control & DEPTH_BEFORE_SHADER != 0
        && db_shader_control & (Z_EXPORT | STENCIL_EXPORTS | MASK_EXPORT) == 0
        && input_addr & COVERAGE_INPUT == 0
        // SV_Coverage, SV_Depth and its two bounded forms, SV_StencilRef.
        && !outputs.iter().any(|o| matches!(o.kind, 14 | 17..=20))
}

impl Lower<'_> {
    /// Sets the early depth-stencil flag where [`Mac::early_depth`] is on
    /// and [`early_depth`] holds for the program.
    pub(super) fn early_depth_flag(&mut self) {
        if let StageIo::Ps {
            db_shader_control,
            input_addr,
            ..
        } = self.x.native.io
            && self.x.mac.early_depth
            && self.s.pixel
            && early_depth(db_shader_control, input_addr, &self.s.outputs)
        {
            self.s.flags |= flags::EARLY_DEPTH_STENCIL;
        }
    }
}
