//! A stand-in for the indirect light the title traces rays for, estimated
//! from the frame itself: avatarmanz's (`victory-gpu/src/ambient.rs` in
//! VIctory-7900XTX at 9576c45f, after beloooo's constant). On with
//! `VICTORY_AMBIENT=estimate` (or `estimate:<share>`), beside the constant
//! stand-in of `crate::ambient` (`VICTORY_AMBIENT=1`), which it leaves as it
//! is; off, nothing here runs.
//!
//! The constant stand-in gives every point of the picture one light: far
//! too little outdoors by day, flat grey indoors. This one works the light
//! out from what is on screen, so it follows the hour, the room and its
//! lamps by itself:
//!
//! - The title keeps the light its visible surfaces send out, for its own
//!   screen traces (`CS_CacheScreenSpaceRadiance`, an R11G11B10_FLOAT image
//!   half the picture's size, written near the end of a frame). Right after
//!   that program, in the same host list, the image is averaged in 64x36
//!   cells of a buffer of ours ([`keep`]), and a slow mean of all cells is
//!   moved a share of the way to this frame's.
//! - Right after the program that resolves the screen probes
//!   (`CS_ScreenProbes_TemporalResolve_*`, the next frame), the cells are
//!   placed in space by the depth image the resolve reads; at 160x90 points
//!   of the picture, the cells' light is summed, each cell weighted by the
//!   share of the point's surroundings it covers, as a mean and a direction
//!   per colour (first-order harmonics); and every texel of the resolve's
//!   light image is given what lies between the four points around it,
//!   those at its own depth first, turned by its normal, a fifth of it the
//!   slow mean, darker in creases ([`raise`]).
//!
//! What it assumes: what is not on screen sends out what is (with the slow
//! mean, so that turning into a dark corner is not black at once). No
//! occlusion but the creases'.
//! shortcut: stands in for the title's traced probes; gone once rays run.
//!
//! What differs from the fork's file, for this line (as `crate::ambient`
//! differs from Belobug's):
//! - The mode is chosen by `VICTORY_AMBIENT=estimate[:<share>]`; in the
//!   fork a number alone is the share and `constant:<level>` the constant.
//! - The root signatures are built in code, and the shader keeps to what
//!   Wine's compiler takes (`crate::ambient_estimate_set::SOURCE`).
//! - The passes have a view heap of their own, made with other threads'
//!   submissions held off (`Host::quiet`, VIctory-macOS issue #31), and
//!   make their own views of the title's images; the fork uses the views
//!   in the title's heap. The list's heaps, root signature and pipeline
//!   are set again by the next draw (`State::host_state_lost`).
//! - The light visible surfaces send out is read as a shader resource: its
//!   host texture is moved to that state in the list
//!   (`HostList::move_texture`) and back by the title's next use.
//! - An image is taken if a view of the format wanted can be made of its
//!   host texture (one host texture per texture memory, castable).
//! - What the title's resolve left is replaced unless
//!   `VICTORY_AMBIENT_LEFT` says how much of it counts (the fork: 4 times
//!   the estimate, fixed).
//! - Without a depth image of the light's size the slow mean alone is
//!   written; the fork then leaves the title's output.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::ambient_estimate_set as set;
use crate::host::{Host, HostList, host};
use crate::object::This;
use crate::resource::HostTexture;

pub(crate) use set::{KEEPS, RADIANCE, role_name};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The share of the estimate asked for (`VICTORY_AMBIENT=estimate` or
/// `estimate:<share>`), if the estimate is. Read once.
fn share() -> Option<f32> {
    static SHARE: OnceLock<Option<f32>> = OnceLock::new();
    *SHARE.get_or_init(|| set::share(var("VICTORY_AMBIENT").as_deref()))
}

/// Whether the stand-in is the estimate.
pub(crate) fn on() -> bool {
    share().is_some()
}

/// The roles the estimate follows programs in beside the constant
/// stand-in's (bits; 0 while it is off).
pub(crate) fn roles() -> u8 {
    if on() { KEEPS } else { 0 }
}

/// What the program of shader container `container` is to the estimate
/// alone (0 while it is off: the container is not searched).
pub(crate) fn role_of(container: &[u8]) -> u8 {
    if on() { set::role_of(container) } else { 0 }
}

/// The creases' shade, as the constant stand-in's
/// (`VICTORY_AMBIENT_SHADE=s,r,n,t`).
fn shade() -> [f32; 4] {
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| crate::ambient_set::shade(var("VICTORY_AMBIENT_SHADE").as_deref()))
}

/// How much of what the title's resolve left counts
/// (`VICTORY_AMBIENT_LEFT`, `set::left`).
fn left() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| set::left(var("VICTORY_AMBIENT_LEFT").as_deref()))
}

/// Says what is on, once, when the host device is ready.
pub(crate) fn announce() {
    if let Some(share) = share() {
        tracing::info!("{}", set::start_line(share, shade(), left()));
    }
}

/// Passes' views, used in turn: more than the passes of every list in
/// flight (two views a pass).
const SLOTS: u32 = 1024;

/// One pass: its pipeline and the root signature it was made with (the
/// same layout for all).
struct Stage {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct Pass {
    keep: Stage,
    mean: Stage,
    place: Stage,
    gather: Stage,
    raise: Stage,
    /// The cells' light, the slow mean and the cells' places
    /// (`set::CELL_BYTES`), and where the CPU sees them.
    cells: ID3D12Resource,
    cells_at: usize,
    /// The points (`set::POINT_BYTES`), and where the CPU sees them.
    points: ID3D12Resource,
    points_at: usize,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded; the mappings live as the buffers.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

/// A pass of `set::SOURCE`, compiled with the system's compiler (Wine's
/// under Wine); its root signature is built here (`crate::compute`): the
/// constants, a table of the image written and the image read, and the two
/// buffers, each as a structured and as a raw view.
fn stage(pass: &str, name: PCSTR) -> windows::core::Result<Stage> {
    use crate::compute::{Param, srv, uav};
    let ranges = [uav(0, 1), srv(0, 1)];
    let params = [
        Param::Constants { register: 0, count: set::CONSTANTS as u32 },
        Param::Table(&ranges),
        Param::Uav { register: 1 },
        Param::Uav { register: 2 },
        Param::Uav { register: 3 },
        Param::Uav { register: 4 },
    ];
    let (root, pso) = crate::compute::pipeline(&set::part(pass), name, 0, &params)?;
    Ok(Stage { root, pso })
}

fn make(host: &Host) -> windows::core::Result<Pass> {
    let keep = stage(set::KEEP, s!("ambient estimate: keep"))?;
    let mean = stage(set::MEAN, s!("ambient estimate: mean"))?;
    let place = stage(set::PLACE, s!("ambient estimate: place"))?;
    let gather = stage(set::GATHER, s!("ambient estimate: gather"))?;
    let raise = stage(set::RAISE, s!("ambient estimate: raise"))?;
    // Buffers the CPU sees too (`log` reads them), all zero to begin with:
    // nothing kept, no light.
    let buffer = |bytes: u64, what: &'static str| {
        let (buffer, data) = host.shared_buffer(bytes).ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the mapping holds the whole buffer.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        crate::host::name(&buffer, || what.into());
        windows::core::Result::Ok((buffer, data as usize))
    };
    let (cells, cells_at) = buffer(set::CELL_BYTES, "ambient estimate: cells")?;
    let (points, points_at) = buffer(set::POINT_BYTES, "ambient estimate: points")?;
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // Under Wine on D3DMetal, lists other threads submit while a view heap
    // is made can lose their views (VIctory-macOS issue #31).
    let _quiet = host.quiet();
    // SAFETY: plain creations.
    unsafe {
        let views = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            keep,
            mean,
            place,
            gather,
            raise,
            cells,
            cells_at,
            points,
            points_at,
            views,
            step: host.device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

/// The passes, made on first use; `None` if the host does not take them
/// (said once, with the compiler's messages).
fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        make(host)
            .inspect_err(|e| tracing::warn!(error = %e, "no ambient estimate: its passes were not made"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether views of `wanted` can be made of the host texture `copy`, whose
/// own format is `own` (as `crate::ambient` asks).
fn viewed_as(copy: &HostTexture, own: DXGI_FORMAT, wanted: DXGI_FORMAT) -> bool {
    own == wanted || copy.castable.contains(&(wanted.0 as u32))
}

/// An image a pass writes through `u0` or reads through `t0`, and the
/// format of its view.
#[derive(Clone, Copy)]
struct Image<'a> {
    image: &'a ID3D12Resource,
    format: DXGI_FORMAT,
}

impl Pass {
    /// Makes the two views of a pass's table in slots no list in flight
    /// reads, and sets the passes' view heap in `list`: the table. A null
    /// view where the pass has no image.
    ///
    /// # Safety
    /// An open list; live textures the list keeps.
    unsafe fn table(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        written: Option<Image>,
        read: Option<Image>,
    ) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let slot = self.next.fetch_add(1, Ordering::Relaxed) % SLOTS * 2;
        // SAFETY: the caller's; the slots are inside the heap.
        unsafe {
            let cpu = self.views.GetCPUDescriptorHandleForHeapStart();
            let gpu = self.views.GetGPUDescriptorHandleForHeapStart();
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + (slot * self.step) as usize };
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: written.map_or(DXGI_FORMAT_R32_UINT, |w| w.format),
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            host.device.CreateUnorderedAccessView(
                written.map(|w| w.image),
                None::<&ID3D12Resource>,
                Some(&view),
                at,
            );
            let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: read.map_or(DXGI_FORMAT_R32_FLOAT, |r| r.format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            };
            let beside = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + self.step as usize };
            host.device.CreateShaderResourceView(read.map(|r| r.image), Some(&seen), beside);
            list.SetDescriptorHeaps(&[Some(self.views.clone())]);
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(slot * self.step) }
        }
    }

    /// Records one pass: its root signature and pipeline, the constants,
    /// the table, the buffers, the dispatch and a barrier for what it
    /// wrote.
    ///
    /// # Safety
    /// An open list with the passes' view heap set (`Pass::table`).
    unsafe fn run(
        &self,
        list: &ID3D12GraphicsCommandList,
        stage: &Stage,
        constants: &[u32; set::CONSTANTS],
        table: D3D12_GPU_DESCRIPTOR_HANDLE,
        groups: [u32; 3],
    ) {
        // SAFETY: the caller's; buffers that live as the passes do.
        unsafe {
            list.SetComputeRootSignature(&stage.root);
            list.SetPipelineState(&stage.pso);
            list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.SetComputeRootDescriptorTable(1, table);
            for (parameter, buffer) in [&self.cells, &self.cells, &self.points, &self.points].into_iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(2 + parameter as u32, buffer.GetGPUVirtualAddress());
            }
            list.Dispatch(groups[0], groups[1], groups[2]);
        }
        uav_barrier(list);
    }

    /// The keeping: the cells' light from `radiance` (a shader resource of
    /// `size` texels in `list` now), then the slow mean.
    fn keep(&self, host: &Host, list: &HostList, radiance: Image, size: [u32; 2], given: &set::Pass) {
        list.keep(&self.cells);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, ..*given });
        uav_barrier(l);
        // SAFETY: an open list; a live texture the caller's list keeps, in
        // a shader-resource state.
        unsafe {
            let table = self.table(host, l, None, Some(radiance));
            self.run(l, &self.keep, &constants, table, [set::CX / 8, set::CY / 4, 1]);
            self.run(l, &self.mean, &constants, table, [1, 1, 1]);
        }
    }

    /// The raising: the cells placed by `depth`, the points' light, and the
    /// light image `light` (of `size` texels, in the unordered-access state
    /// in `list` now) given the estimate. Without `depth` (a shader
    /// resource of the same size), the slow mean alone.
    fn raise(
        &self,
        host: &Host,
        list: &HostList,
        light: Image,
        depth: Option<Image>,
        size: [u32; 2],
        given: &set::Pass,
    ) {
        list.keep(&self.cells);
        list.keep(&self.points);
        list.keep(&self.views);
        let l = &list.list;
        let constants = set::constants(&set::Pass { size, flat: depth.is_none(), ..*given });
        uav_barrier(l);
        // SAFETY: an open list; live textures the caller's list keeps, in
        // the states named.
        unsafe {
            let table = self.table(host, l, Some(light), depth);
            if depth.is_some() {
                self.run(l, &self.place, &constants, table, [set::CX / 8, set::CY / 4, 1]);
                self.run(l, &self.gather, &constants, table, [set::PX / 8, set::PY / 2, 1]);
            }
            self.run(l, &self.raise, &constants, table, [size[0].div_ceil(8), size[1].div_ceil(8), 1]);
        }
    }
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: AtomicBool = AtomicBool::new(false);

/// Keepings and raisings recorded since the last frame line.
static RECORDED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// With the frame line, over `frames` frames: what the estimate is, to
/// compare with the constant stand-in's level in a run log. Read from the
/// buffers as the CPU sees them now (the GPU may be writing them: a
/// diagnostic, not a measurement of one frame). Nothing while the estimate
/// is off.
pub(crate) fn log(frames: f64) {
    let Some(share) = share() else {
        return;
    };
    let [keeps, raises] = [0, 1].map(|i| RECORDED[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
    let made = KEPT.load(Ordering::Relaxed).then(|| host().and_then(pass)).flatten();
    let seen = made.map(|pass| {
        // SAFETY: inside the buffers' mappings, which live as the passes do.
        let float4 = |at: usize, index: u32| unsafe { ((at + index as usize * 16) as *const [f32; 4]).read_volatile() };
        let mean = float4(pass.cells_at, set::CELLS);
        let points = (0..set::PX * set::PY).map(|i| [0, 1, 2].map(|colour| float4(pass.points_at, i * 3 + colour)[3]));
        ([mean[0], mean[1], mean[2]], set::over_the_picture(points))
    });
    tracing::info!("{}", set::frame_line(share, seen, [keeps, raises]));
}

/// What the passes are given now.
fn settings(share: f32) -> set::Pass {
    set::Pass { level: share, shade: shade(), follow: set::FOLLOW, most: left(), size: [0, 0], flat: false }
}

/// After the title's program that keeps the light visible surfaces send
/// out, in `list`: the image it wrote, `radiance`, is averaged into the
/// cells. Whether it recorded anything (the list's heaps, compute root
/// signature and pipeline are then ours).
pub(crate) fn keep(list: &HostList, radiance: This) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    let Some(copy) = crate::resource::host_copy(radiance) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let format = DXGI_FORMAT_R11G11B10_FLOAT;
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !viewed_as(&copy, d.Format, format) {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "no ambient estimate: the light visible surfaces send out is not an R11G11B10_FLOAT image"
            );
        });
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            "ambient estimate: the first light it keeps"
        );
    });
    list.keep(&copy.texture);
    // The program before wrote it through an unordered view: read as a
    // shader resource here, and moved back by the title's next use.
    if let Some(barrier) = list.move_texture(&copy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32) {
        // SAFETY: an open list and a live texture the list keeps.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    let radiance = Image { image: &copy.texture, format };
    pass.keep(host, list, radiance, [d.Width as u32, d.Height], &settings(share));
    KEPT.store(true, Ordering::Relaxed);
    RECORDED[0].fetch_add(1, Ordering::Relaxed);
    true
}

/// After the title's program that resolves the screen probes, in `list`:
/// what `light` holds is replaced by the estimate (or raised to it,
/// `VICTORY_AMBIENT_LEFT`). `depth` is the scene's depth the resolve read.
/// Whether it recorded anything (the list's heaps, compute root signature
/// and pipeline are then ours).
pub(crate) fn raise(list: &HostList, light: This, depth: Option<This>) -> bool {
    let (Some(share), Some(host)) = (share(), host()) else {
        return false;
    };
    // Nothing kept yet: the title's own output for a frame.
    if !KEPT.load(Ordering::Relaxed) {
        return false;
    }
    let Some(light) = crate::resource::host_copy(light) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !(d.Format == DXGI_FORMAT_R32_TYPELESS || viewed_as(&light, d.Format, DXGI_FORMAT_R32_UINT))
    {
        static REFUSED: std::sync::Once = std::sync::Once::new();
        REFUSED.call_once(|| {
            tracing::warn!(
                format = d.Format.0,
                dimension = d.Dimension.0,
                "no ambient estimate: the light image is not one a 32-bit integer view can be made of"
            );
        });
        return false;
    }
    let Some(pass) = pass(host) else {
        return false;
    };
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the slow mean alone is written.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            share,
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            depth = depth.is_some(),
            "ambient estimate: the first image it writes"
        );
        if depth.is_none() {
            tracing::warn!(
                "ambient estimate: no depth image of the light's size: every texel gets the mean of the light around"
            );
        }
    });
    list.keep(&light.texture);
    if let Some(depth) = &depth {
        list.keep(depth);
    }
    let image = Image { image: &light.texture, format: DXGI_FORMAT_R32_UINT };
    let depth = depth.as_ref().map(|d| Image { image: d, format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS });
    pass.raise(host, list, image, depth, [d.Width as u32, d.Height], &settings(share));
    RECORDED[1].fetch_add(1, Ordering::Relaxed);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::transition;
    use set::reference::{self, Scene, V3, V4};

    /// The passes compile with the system's compiler and the host takes
    /// them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(host).expect("the passes");
    }

    /// A value as an unsigned float of `bits` mantissa bits and a 5-bit
    /// exponent (R11G11B10_FLOAT's parts); exact for the values used here.
    fn small_float(v: f32, bits: u32) -> u32 {
        if v <= 0.0 {
            return 0;
        }
        let e = v.log2().floor() as i32;
        let m = ((v / 2f32.powi(e) - 1.0) * (1 << bits) as f32).round() as u32;
        ((e + 15) as u32) << bits | m
    }

    fn r11g11b10(c: V3) -> u32 {
        small_float(c[0], 6) | small_float(c[1], 6) << 11 | small_float(c[2], 5) << 22
    }

    /// A texture of `format` holding `words` (one 32-bit word a texel),
    /// left in `state`; its upload buffer goes to `keep`.
    fn texture(
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        format: DXGI_FORMAT,
        size: [u32; 2],
        words: &[u32],
        state: D3D12_RESOURCE_STATES,
        keep: &mut Vec<ID3D12Resource>,
    ) -> ID3D12Resource {
        let texture = crate::composite::result_texture(&host.device, size[0].into(), size[1], format).unwrap();
        let pitch = (size[0] * 4).next_multiple_of(256);
        let (upload, data) = host.upload_buffer(u64::from(pitch) * u64::from(size[1])).unwrap();
        // SAFETY: a mapped upload buffer written in bounds; an open list
        // and copy locations naming live resources.
        unsafe {
            for y in 0..size[1] {
                let row = &words[(y * size[0]) as usize..((y + 1) * size[0]) as usize];
                std::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), data.add((y * pitch) as usize), row.len() * 4);
            }
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyTextureRegion(&whole(&texture), 0, 0, 0, &rows(&upload, format, size, pitch), None);
            list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
        keep.push(upload);
        texture
    }

    fn whole(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        }
    }

    fn rows(buffer: &ID3D12Resource, format: DXGI_FORMAT, size: [u32; 2], pitch: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: size[0],
                        Height: size[1],
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        }
    }

    /// The `float4`s of a buffer of ours the CPU sees, from `first` on.
    fn floats(at: usize, first: u32, count: u32) -> Vec<V4> {
        // SAFETY: inside the buffer's mapping; the submission has run.
        (0..count).map(|i| unsafe { ((at + ((first + i) * 16) as usize) as *const V4).read_volatile() }).collect()
    }

    fn near(a: f32, b: f32, part: f32) -> bool {
        (a - b).abs() <= part * a.abs().max(b.abs()) + 1e-5
    }

    /// The scene of the GPU tests: a floor 0.6 m under the eye up to a wall
    /// 3 m ahead, whose left half is lit orange and whose right half is
    /// dim blue, the floor grey; one texel in sixteen brighter than any
    /// light should count for.
    const LIGHT: [u32; 2] = [256, 144];
    const RADIANCE: [u32; 2] = [128, 72];

    fn depth_at(_: u32, y: u32) -> f32 {
        let down = ((y as f32 + 0.5) / LIGHT[1] as f32 * 2.0 - 1.0) * 0.45;
        0.1 / if down > 0.0 { (0.6 / down).min(3.0) } else { 3.0 }
    }

    fn radiance_at(x: u32, y: u32) -> V3 {
        if x % 4 == 1 && y % 4 == 2 {
            [1024.0, 512.0, 256.0]
        } else if y >= RADIANCE[1] * 13 / 18 {
            [0.25, 0.25, 0.25]
        } else if x < RADIANCE[0] / 2 {
            [2.0, 1.0, 0.25]
        } else {
            [0.0625, 0.125, 0.5]
        }
    }

    /// Runs a keeping and a raising on the test scene with `set`, the light
    /// image holding `was` before: the cells' light, the slow mean, the
    /// cells' places, the points and the light image after.
    #[allow(clippy::type_complexity)]
    fn run(host: &Host, pass: &Pass, given: &set::Pass, was: V3, depth: bool) -> (Vec<V4>, V4, Vec<V4>, Vec<V4>, Vec<u32>) {
        let run = host.new_list().unwrap();
        let l = &run.list;
        let mut keep = Vec::new();
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let texels = |size: [u32; 2], at: &dyn Fn(u32, u32) -> u32| -> Vec<u32> {
            (0..size[0] * size[1]).map(|i| at(i % size[0], i / size[0])).collect()
        };
        let radiance = texture(
            host,
            l,
            DXGI_FORMAT_R11G11B10_FLOAT,
            RADIANCE,
            &texels(RADIANCE, &|x, y| r11g11b10(radiance_at(x, y))),
            read,
            &mut keep,
        );
        let scene_depth = texture(
            host,
            l,
            DXGI_FORMAT_R32_FLOAT,
            LIGHT,
            &texels(LIGHT, &|x, y| depth_at(x, y).to_bits()),
            read,
            &mut keep,
        );
        let light = texture(
            host,
            l,
            DXGI_FORMAT_R32_UINT,
            LIGHT,
            &texels(LIGHT, &|_, _| reference::pack(was)),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            &mut keep,
        );
        pass.keep(host, &run, Image { image: &radiance, format: DXGI_FORMAT_R11G11B10_FLOAT }, RADIANCE, given);
        pass.raise(
            host,
            &run,
            Image { image: &light, format: DXGI_FORMAT_R32_UINT },
            depth.then_some(Image { image: &scene_depth, format: DXGI_FORMAT_R32_FLOAT }),
            LIGHT,
            given,
        );
        let pitch = (LIGHT[0] * 4).next_multiple_of(256);
        let (back, data) = host.readback_buffer(u64::from(pitch) * u64::from(LIGHT[1])).unwrap();
        // SAFETY: an open list and live resources.
        unsafe {
            l.ResourceBarrier(&[transition(&light, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyTextureRegion(&rows(&back, DXGI_FORMAT_R32_UINT, LIGHT, pitch), 0, 0, 0, &whole(&light), None);
        }
        run.close();
        host.execute(&[&run]);
        // SAFETY: the copy has run; inside the mapped buffer.
        let written = texels(LIGHT, &|x, y| unsafe { data.add((y * pitch + x * 4) as usize).cast::<u32>().read_volatile() });
        let cells = floats(pass.cells_at, 0, set::CELLS);
        let mean = floats(pass.cells_at, set::CELLS, 1)[0];
        let places = floats(pass.cells_at, set::CELLS + 1, set::CELLS);
        let points = floats(pass.points_at, 0, set::PX * set::PY * 3);
        (cells, mean, places, points, written)
    }

    /// The passes on the GPU give what the arithmetic on the CPU gives
    /// (`set::reference`): the cells' light and places, the slow mean, the
    /// points, and the light image, with and without what the resolve left.
    #[test]
    fn the_passes_match_the_arithmetic() {
        let Some(host) = host() else {
            return;
        };
        let pass = make(host).expect("the passes");
        let scene = Scene { depth: &depth_at, size: LIGHT, near: 0.1, tan_y: 0.45 };
        let base = set::Pass { level: 0.75, shade: [1.0, 0.6, 0.1, 0.45], follow: 0.25, most: 0.0, size: [0, 0], flat: false };
        let want_cells = reference::keep(&radiance_at, RADIANCE);
        let want_places = scene.place();
        let want_points = scene.gather(&want_cells, &want_places);
        let mut slow = [0.0f32; 4];
        for (given, was, depth) in [
            (base, [9.0, 0.0, 9.0], true),
            (set::Pass { most: 4.0, shade: [0.0, 0.6, 0.1, 0.45], ..base }, [9.0, 0.0, 9.0], true),
            (base, [0.0; 3], false),
        ] {
            let (cells, mean, places, points, written) = run(host, &pass, &given, was, depth);
            for (i, (got, want)) in cells.iter().zip(&want_cells).enumerate() {
                assert!((0..3).all(|k| near(got[k], want[k], 1e-3)), "cell {i}: {got:?} {want:?}");
            }
            // The second keeping goes a share of the way from the first.
            slow = reference::mean(&want_cells, slow, given.follow);
            assert!((0..3).all(|k| near(mean[k], slow[k], 1e-3)) && mean[3] > 0.0, "{mean:?} {slow:?}");
            if !depth {
                // The slow mean alone, everywhere.
                let flat = reference::pack([0, 1, 2].map(|k| slow[k] * given.level));
                let odd = written.iter().filter(|&&w| !close(reference::unpack(w), reference::unpack(flat))).count();
                assert_eq!(odd, 0, "texels that are not the slow mean");
                continue;
            }
            for (i, (got, want)) in places.iter().zip(&want_places).enumerate() {
                assert!((0..4).all(|k| near(got[k], want[k], 1e-3)), "place {i}: {got:?} {want:?}");
            }
            for (i, want) in want_points.iter().enumerate() {
                for (colour, want) in want.iter().enumerate() {
                    let got = points[i * 3 + colour];
                    // Directions are differences of sums: to a part of the mean.
                    assert!(
                        near(got[3], want[3], 2e-3) && (0..3).all(|k| (got[k] - want[k]).abs() <= 2e-3 * want[3] + 1e-5),
                        "point {i} colour {colour}: {got:?} {want:?}"
                    );
                }
            }
            let want_points: Vec<[V4; 3]> = want_points.clone();
            let around = [slow[0], slow[1], slow[2]];
            let mut odd = 0;
            for y in 0..LIGHT[1] {
                for x in 0..LIGHT[0] {
                    let want = scene.raise([x as i32, y as i32], &want_points, around, &given, reference::unpack(reference::pack(was)));
                    let got = reference::unpack(written[(y * LIGHT[0] + x) as usize]);
                    if !close(got, reference::unpack(reference::pack(want))) {
                        odd += 1;
                    }
                }
            }
            // A texel at the edge of two points' cells, or with a shade
            // sample on a depth edge, may round the other way on the GPU.
            let most = LIGHT[0] * LIGHT[1] / 100;
            assert!(odd <= most, "{odd} texels differ (most {most}), shade {}", given.shade[0]);
        }
    }

    /// Two packed lights within a fiftieth of the brighter one's top part
    /// (the packing keeps 9 bits of it).
    fn close(a: V3, b: V3) -> bool {
        let top = a.iter().chain(&b).fold(0.0f32, |m, &v| m.max(v));
        (0..3).all(|k| (a[k] - b[k]).abs() <= top / 50.0 + 1e-5)
    }
}
