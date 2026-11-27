//! A title frame's pattern of compute and render work taking turns, as
//! VIctory records it: before each indirect draw, a dispatch of one thread
//! writes that draw's arguments (20 bytes, as the inline argument clamps
//! do), a barrier makes them the draw's, and the draw runs through
//! `ExecuteIndirect` into a render target. On Metal each turn ends one
//! encoder and starts another. Against it: the same dispatches and draws
//! batched, all dispatches, one barrier, then all draws.

use std::mem::ManuallyDrop;

use super::*;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16_UINT};

/// One draw's arguments: `D3D12_DRAW_INDEXED_ARGUMENTS`, which is Metal's
/// `MTLDrawIndexedPrimitivesIndirectArguments` too.
pub(crate) const RECORD: u32 = 20;
/// The render target's width and height.
const SIZE: u32 = 64;

const CLAMP: &str = "cbuffer c : register(b0) { uint record; };\n\
    RWByteAddressBuffer a : register(u0);\n\
    [numthreads(1, 1, 1)] void main() {\n\
        uint o = record * 20;\n\
        a.Store(o, 3); a.Store(o + 4, 1); a.Store(o + 8, 0); a.Store(o + 12, 0); a.Store(o + 16, 0);\n\
    }\n";
/// One triangle over the whole target, from the vertex index alone.
const VERTEX: &str = "float4 main(uint id : SV_VertexID) : SV_Position {\n\
        float2 p = float2((id << 1) & 2, id & 2);\n\
        return float4(p * 2 - 1, 0, 1);\n\
    }\n";
const PIXEL: &str = "float4 main() : SV_Target { return float4(1, 1, 1, 1); }\n";

/// How the dispatches and the draws of one submission are ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pattern {
    /// Dispatch, barrier, indirect draw, barrier: `pairs` times.
    Interleaved,
    /// All dispatches, one barrier, all indirect draws.
    Batched,
    /// Dispatch, UAV barrier, plain draw: the turns without the indirect
    /// draws.
    InterleavedPlain,
    /// Dispatch, plain draw, with no barrier (the draws read nothing the
    /// dispatches write): the turns alone.
    InterleavedUnbarred,
    /// Only the plain draws, in one render pass.
    DrawsOnly,
}

pub(crate) struct Breaks<'a> {
    bench: &'a Bench,
    clamp_root: ID3D12RootSignature,
    clamp: ID3D12PipelineState,
    draw_root: ID3D12RootSignature,
    draw: ID3D12PipelineState,
    signature: ID3D12CommandSignature,
    target: ID3D12Resource,
    targets: ID3D12DescriptorHeap,
    indices: ID3D12Resource,
}

impl<'a> Breaks<'a> {
    pub(crate) fn new(bench: &'a Bench) -> Self {
        let device = &bench.device;
        let clamp_root = root_signature(
            device,
            &[
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 1,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ],
            D3D12_ROOT_SIGNATURE_FLAG_NONE,
        );
        let clamp = compute_pipeline(device, &clamp_root, CLAMP);
        let draw_root = root_signature(device, &[], D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let (vertex, pixel) = (compile(VERTEX, s!("vs_5_1")), compile(PIXEL, s!("ps_5_1")));
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let mut formats = [Default::default(); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the root signature and both shaders live through the call.
        let draw = unsafe {
            device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&draw_root),
                VS: bytecode(&vertex),
                PS: bytecode(&pixel),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            })
        }
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            ..Default::default()
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: one argument, described by a live struct.
        unsafe {
            device.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: RECORD,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &argument,
                    NodeMask: 0,
                },
                None,
                &mut signature,
            )
        }
        .unwrap();
        let mut target: Option<ID3D12Resource> = None;
        // SAFETY: a plain render target description; the out pointer is ours.
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(SIZE),
                    Height: SIZE,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
        }
        .unwrap();
        let target = target.unwrap();
        // SAFETY: a heap of one view, written once.
        let targets: ID3D12DescriptorHeap = unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            device.CreateRenderTargetView(&target, None, heap.GetCPUDescriptorHandleForHeapStart());
            heap
        };
        let indices = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            256,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: an upload buffer of 256 bytes, mapped for six.
        unsafe {
            let mut p = std::ptr::null_mut();
            indices.Map(0, None, Some(&mut p)).unwrap();
            std::ptr::copy_nonoverlapping([0u16, 1, 2].as_ptr(), p.cast(), 3);
            indices.Unmap(0, None);
        }
        Self {
            bench,
            clamp_root,
            clamp,
            draw_root,
            draw,
            signature: signature.unwrap(),
            target,
            targets,
            indices,
        }
    }

    /// One submission of `pairs` dispatches and draws in `pattern`, then a
    /// check of what they made in a submission of its own: every record
    /// the dispatches wrote, and the target the draws covered.
    pub(crate) fn run(&self, pattern: Pattern, pairs: u32) -> Fill {
        let device = &self.bench.device;
        let bytes = (pairs * RECORD).next_multiple_of(256) as usize;
        let args = committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let zeros = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: a live heap of one view.
        let rtv = unsafe { self.targets.GetCPUDescriptorHandleForHeapStart() };
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            // SAFETY: a live upload buffer.
            BufferLocation: unsafe { self.indices.GetGPUVirtualAddress() },
            SizeInBytes: 6,
            Format: DXGI_FORMAT_R16_UINT,
        };
        let indirect = matches!(pattern, Pattern::Interleaved | Pattern::Batched);
        let fill = self.bench.run(|l| {
            // SAFETY: every object bound outlives the submission, which is
            // waited for; the barriers name the states the list left.
            unsafe {
                // Fresh arguments each time: what the draws find is what
                // this submission's dispatches wrote.
                l.CopyBufferRegion(&args, 0, &zeros, 0, bytes as u64);
                l.ResourceBarrier(&[transition(
                    &args,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                l.ClearRenderTargetView(rtv, &[0.0; 4], None);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: SIZE as f32,
                    Height: SIZE as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: SIZE as i32,
                    bottom: SIZE as i32,
                }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                l.IASetIndexBuffer(Some(&ibv));
                l.SetGraphicsRootSignature(&self.draw_root);
                l.SetComputeRootSignature(&self.clamp_root);
                l.SetComputeRootUnorderedAccessView(1, args.GetGPUVirtualAddress());
                let dispatch = |i: u32| {
                    l.SetComputeRoot32BitConstant(0, i, 0);
                    l.Dispatch(1, 1, 1);
                };
                let to_args = transition(
                    &args,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                );
                let to_uav = transition(
                    &args,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let draw = |i: u32| {
                    if indirect {
                        l.ExecuteIndirect(
                            &self.signature,
                            1,
                            &args,
                            u64::from(i * RECORD),
                            None,
                            0,
                        );
                    } else {
                        l.DrawIndexedInstanced(3, 1, 0, 0, 0);
                    }
                };
                match pattern {
                    Pattern::Interleaved
                    | Pattern::InterleavedPlain
                    | Pattern::InterleavedUnbarred => {
                        for i in 0..pairs {
                            l.SetPipelineState(&self.clamp);
                            dispatch(i);
                            if indirect {
                                l.ResourceBarrier(std::slice::from_ref(&to_args));
                            } else if pattern == Pattern::InterleavedPlain {
                                l.ResourceBarrier(&[uav_barrier(&args)]);
                            }
                            l.SetPipelineState(&self.draw);
                            draw(i);
                            if indirect {
                                l.ResourceBarrier(std::slice::from_ref(&to_uav));
                            }
                        }
                    }
                    Pattern::Batched => {
                        l.SetPipelineState(&self.clamp);
                        (0..pairs).for_each(dispatch);
                        l.ResourceBarrier(std::slice::from_ref(&to_args));
                        l.SetPipelineState(&self.draw);
                        (0..pairs).for_each(draw);
                        l.ResourceBarrier(std::slice::from_ref(&to_uav));
                    }
                    Pattern::DrawsOnly => {
                        l.SetPipelineState(&self.draw);
                        (0..pairs).for_each(draw);
                    }
                }
            }
        });
        self.check(&args, pairs, pattern != Pattern::DrawsOnly);
        fill
    }

    /// Reads back the arguments and the target in a submission of its own.
    fn check(&self, args: &ID3D12Resource, pairs: u32, dispatched: bool) {
        let device = &self.bench.device;
        let records = (pairs * RECORD) as usize;
        let args_back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            records.next_multiple_of(256),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let pitch = SIZE * 4;
        let target_back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            (pitch * SIZE) as usize,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        self.bench.run(|l| {
            // SAFETY: all four resources live through the waited submission.
            unsafe {
                l.CopyBufferRegion(&args_back, 0, args, 0, records as u64);
                l.ResourceBarrier(&[transition(
                    &self.target,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyTextureRegion(
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&target_back),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 0,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: SIZE,
                                    Height: SIZE,
                                    Depth: 1,
                                    RowPitch: pitch,
                                },
                            },
                        },
                    },
                    0,
                    0,
                    0,
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&self.target),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    },
                    None,
                );
                l.ResourceBarrier(&[transition(
                    &self.target,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
        });
        // SAFETY: readback buffers, mapped whole after the waited copies.
        unsafe {
            if dispatched {
                let mut p = std::ptr::null_mut();
                args_back.Map(0, None, Some(&mut p)).unwrap();
                let words = std::slice::from_raw_parts(p.cast::<u32>(), records / 4);
                for (i, record) in words.chunks(5).enumerate() {
                    assert_eq!(record, [3, 1, 0, 0, 0], "record {i} the dispatches wrote");
                }
                args_back.Unmap(0, None);
            }
            let mut p = std::ptr::null_mut();
            target_back.Map(0, None, Some(&mut p)).unwrap();
            let pixels = std::slice::from_raw_parts(p.cast::<u32>(), (SIZE * SIZE) as usize);
            assert_eq!(
                pixels[(SIZE * SIZE / 2 + SIZE / 2) as usize],
                u32::MAX,
                "the draws' target"
            );
            target_back.Unmap(0, None);
        }
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a copy of the pointer without a reference of its
                // own, never released (ManuallyDrop); the caller keeps the
                // resource alive.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: as in `transition`.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}
