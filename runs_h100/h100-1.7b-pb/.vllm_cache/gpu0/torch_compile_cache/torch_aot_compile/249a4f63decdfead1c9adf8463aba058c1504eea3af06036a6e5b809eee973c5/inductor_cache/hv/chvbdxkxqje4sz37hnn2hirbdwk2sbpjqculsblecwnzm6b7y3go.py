r"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/sa/csa7mjkela3ttsahhtilr5uiqswvppb2jcjm6wpcc7lsshbrgd7j.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_4, add_tensor_5, convert_element_type_default_11, convert_element_type_default_8, convert_element_type_default_9, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
# Graph fragment:
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_default_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_4, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_4, %rsqrt_default_3), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_6, torch.bfloat16), kwargs = {})
#   %mul_tensor_7 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg3_1), kwargs = {})
#   return %buf1,%mul_tensor_7
triton_red_fused_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 33558528}}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 2048.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/3u/c3ugiy4hgvhiipsntjqxxgzt6zzrb4a74kw3alrkz7takiyjo4s3.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_16
#   silu => add_18, convert_element_type_4, convert_element_type_5, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s72, 12288][12288, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s72, 6144][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 6144), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_18 : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add_18), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 6144][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 6144, 9223372036854775807), kwargs = {})
#   %mul_16 : Tensor "bf16[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %slice_2), kwargs = {})
#   return %mul_16
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12288*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (6144 + x0 + 12288*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/wh/cwhzto35hfbbzszohtfuadolc4jdfp5u3kcwkexlxdqzsbcbhzlr.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_4, convert_element_type_default_10, convert_element_type_default_8, convert_element_type_default_9
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %mm_2 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf7 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf7]
#   %arg7_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_default_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %convert_element_type_default_10 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_10, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg7_1), kwargs = {})
#   return %buf7,%convert_element_type_default_6,%mul_tensor_5
triton_red_fused_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 58724352}}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 2048.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/2t/c2tesixuxwpmefccnjv3x5pyqa5gbghrfynktrzfypylt7kfaeiw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_3 = async_compile.triton('triton_red_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_3', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_3(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 16)
        x1 = xindex // 16
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x0 + 4096*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = (xindex % 8)
        x5 = xindex // 8
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (2048 + r0_6 + 128*x4 + 4096*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x7), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048, 16, 1), (16, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 8, 1), (8, 1, 16384), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 32768, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/rh/crhz6qriomrayonuh2bga5myb6lr2yc7e24u2gmha3kgrzwkqodw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 8)
        x2 = xindex // 512
        x3 = xindex // 64
        tmp0 = tl.load(in_ptr0 + (2048 + x0 + 128*x1 + 4096*x2), xmask).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
        tmp20 = tl.load(in_ptr0 + (2112 + x0 + 128*x1 + 4096*x2), xmask).to(tl.float32)
        tmp24 = tl.load(in_ptr2 + (64 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tl.full([1], 128.0, tl.float32)
        tmp4 = (tmp2 / tmp3)
        tmp5 = tl.full([1], 1e-06, tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.rsqrt(tmp6)
        tmp8 = tmp1 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 * tmp10
        tmp13 = tl.full([XBLOCK], 40960, tl.int32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp12 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp12)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 40960)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 40960")
        tmp18 = tl.load(in_ptr4 + (x0 + 128*tmp16), xmask).to(tl.float32)
        tmp19 = tmp11 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp7
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 * tmp24
        tmp26 = tl.load(in_ptr4 + (64 + x0 + 128*tmp16), xmask).to(tl.float32)
        tmp27 = tmp25 * tmp26
        tmp28 = tmp19 - tmp27
        tmp29 = tmp25 * tmp18
        tmp30 = tmp11 * tmp26
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr0 + (x0 + 128*x3), tmp28, xmask)
        tl.store(out_ptr1 + (x0 + 128*x3), tmp31, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 64)
        x5 = ((xindex // 64) % 16)
        x6 = xindex // 1024
        x7 = xindex // 64
        tmp32 = tl.load(in_ptr0 + (x4 + 128*x5 + 4096*x6), xmask).to(tl.float32)
        tmp34 = tl.load(in_ptr5 + (x7), xmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp44 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
        tmp52 = tl.load(in_ptr0 + (64 + x4 + 128*x5 + 4096*x6), xmask).to(tl.float32)
        tmp56 = tl.load(in_ptr6 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp35 = tl.full([1], 128.0, tl.float32)
        tmp36 = (tmp34 / tmp35)
        tmp37 = tl.full([1], 1e-06, tl.float32)
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp33 * tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp43 = tmp41 * tmp42
        tmp45 = tl.full([XBLOCK], 40960, tl.int32)
        tmp46 = tmp44 + tmp45
        tmp47 = tmp44 < 0
        tmp48 = tl.where(tmp47, tmp46, tmp44)
        tl.device_assert(((0 <= tmp48) & (tmp48 < 40960)) | ~(xmask), "index out of bounds: 0 <= tmp48 < 40960")
        tmp50 = tl.load(in_ptr4 + (x4 + 128*tmp48), xmask).to(tl.float32)
        tmp51 = tmp43 * tmp50
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp53 * tmp39
        tmp55 = tmp54.to(tl.float32)
        tmp57 = tmp55 * tmp56
        tmp58 = tl.load(in_ptr4 + (64 + x4 + 128*tmp48), xmask).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp51 - tmp59
        tmp61 = tmp57 * tmp50
        tmp62 = tmp43 * tmp58
        tmp63 = tmp61 + tmp62
        tl.store(out_ptr2 + (x4 + 128*x7), tmp60, xmask)
        tl.store(out_ptr3 + (x4 + 128*x7), tmp63, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048, 8, 1), (8, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((2048, 16, 1), (16, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((2048, 8, 64), (1024, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((2048, 8, 64), (1024, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((2048, 16, 64), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((2048, 16, 64), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1048576, 2097152,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    stream0 = get_raw_stream(0)
stream0 = get_raw_stream(0)
buf0 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
arg4_1 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
arg3_1 = generate_example_value((2048,), (1,), 'cuda:0', torch.bfloat16, 0, (2048,))
buf2 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_0.run(buf0, arg4_1, arg3_1, buf2, 2048, 2048, stream=stream0)
del arg3_1, buf2

stream0 = get_raw_stream(0)
buf3 = generate_example_value((2048, 12288), (12288, 1), 'cuda:0', torch.bfloat16, 0, (2048, 12288))
buf4 = generate_example_value((2048, 6144), (6144, 1), 'cuda:0', torch.bfloat16, 0, (2048, 6144))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, 12582912, stream=stream0)
del buf3, buf4

stream0 = get_raw_stream(0)
buf5 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
arg7_1 = generate_example_value((2048,), (1,), 'cuda:0', torch.bfloat16, 0, (2048,))
buf6 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
buf8 = generate_example_value((2048, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (2048, 2048))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_2.run(buf5, buf0, arg4_1, arg7_1, buf6, buf8, 2048, 2048, stream=stream0)
del buf0, arg4_1, buf5, arg7_1, buf6, buf8

stream0 = get_raw_stream(0)
buf9 = generate_example_value((2048, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (2048, 4096))
buf10 = generate_example_value((2048, 16, 1), (16, 1, 32768), 'cuda:0', torch.float32, 0, (2048, 16, 1))
buf11 = generate_example_value((2048, 8, 1), (8, 1, 16384), 'cuda:0', torch.float32, 0, (2048, 8, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_3.run(buf9, buf10, buf11, 32768, 16384, stream=stream0)

stream0 = get_raw_stream(0)
arg10_1 = generate_example_value((128,), (1,), 'cuda:0', torch.bfloat16, 0, (128,))
arg11_1 = generate_example_value((2048,), (1,), 'cuda:0', torch.int64, 0, (2048,))
arg12_1 = generate_example_value((40960, 128), (128, 1), 'cuda:0', torch.bfloat16, 0, (40960, 128))
arg9_1 = generate_example_value((128,), (1,), 'cuda:0', torch.bfloat16, 0, (128,))
buf12 = generate_example_value((2048, 8, 64), (1024, 128, 1), 'cuda:0', torch.bfloat16, 0, (2048, 8, 64))
buf13 = generate_example_value((2048, 8, 64), (1024, 128, 1), 'cuda:0', torch.bfloat16, 0, (2048, 8, 64))
buf15 = generate_example_value((2048, 16, 64), (2048, 128, 1), 'cuda:0', torch.bfloat16, 0, (2048, 16, 64))
buf16 = generate_example_value((2048, 16, 64), (2048, 128, 1), 'cuda:0', torch.bfloat16, 0, (2048, 16, 64))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_4.run(buf9, buf11, arg10_1, arg11_1, arg12_1, buf10, arg9_1, buf12, buf13, buf15, buf16, 1048576, 2097152, stream=stream0)
del buf9, buf10, buf11, arg10_1, arg11_1, arg12_1, arg9_1, buf12, buf13, buf15, buf16

"""
# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/sa/csa7mjkela3ttsahhtilr5uiqswvppb2jcjm6wpcc7lsshbrgd7j.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_4, add_tensor_5, convert_element_type_default_11, convert_element_type_default_8, convert_element_type_default_9, mean_dim_3, mul_tensor_6, mul_tensor_7, pow_tensor_scalar_3, rsqrt_default_3
# Graph fragment:
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_default_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %pow_tensor_scalar_3 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_4, 2), kwargs = {})
#   %mean_dim_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_3, [-1], True), kwargs = {})
#   %add_tensor_5 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_3, 1e-06), kwargs = {})
#   %rsqrt_default_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_tensor_6 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_4, %rsqrt_default_3), kwargs = {})
#   %convert_element_type_default_11 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_6, torch.bfloat16), kwargs = {})
#   %mul_tensor_7 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_11, %arg3_1), kwargs = {})
#   return %buf1,%mul_tensor_7
triton_red_fused_fused_add_rms_norm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_fused_add_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 33558528}}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 2048.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/3u/c3ugiy4hgvhiipsntjqxxgzt6zzrb4a74kw3alrkz7takiyjo4s3.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   mul => mul_16
#   silu => add_18, convert_element_type_4, convert_element_type_5, div, exp, neg
# Graph fragment:
#   %mm_1 : Tensor "bf16[s72, 12288][12288, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s72, 6144][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 6144), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_18 : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add_18), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 6144][12288, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 6144, 9223372036854775807), kwargs = {})
#   %mul_16 : Tensor "bf16[s72, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %slice_2), kwargs = {})
#   return %mul_16
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6144)
    x1 = xindex // 6144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12288*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (6144 + x0 + 12288*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/wh/cwhzto35hfbbzszohtfuadolc4jdfp5u3kcwkexlxdqzsbcbhzlr.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_4, convert_element_type_default_10, convert_element_type_default_8, convert_element_type_default_9
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %mm_2 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf7 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf7]
#   %arg7_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_default_8 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_default_9 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_8, %convert_element_type_default_9), kwargs = {})
#   %convert_element_type_default_10 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_10, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.bfloat16), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_7 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg7_1), kwargs = {})
#   return %buf7,%convert_element_type_default_6,%mul_tensor_5
triton_red_fused_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_fused_add_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 58724352}}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 2048.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/2t/c2tesixuxwpmefccnjv3x5pyqa5gbghrfynktrzfypylt7kfaeiw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_3 = async_compile.triton('triton_red_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_3', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_3(in_ptr0, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % 16)
        x1 = xindex // 16
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x0 + 4096*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tl.store(out_ptr0 + (x3), tmp4, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = (xindex % 8)
        x5 = xindex // 8
        _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp6 = tl.load(in_ptr0 + (2048 + r0_6 + 128*x4 + 4096*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7 * tmp7
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
            tmp11 = _tmp10 + tmp9
            _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp10 = tl.sum(_tmp10, 1)[:, None]
        tl.store(out_ptr1 + (x7), tmp10, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048, 16, 1), (16, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048, 8, 1), (8, 1, 16384), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 32768, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: runs/h100-1.7b-pb/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/249a4f63decdfead1c9adf8463aba058c1504eea3af06036a6e5b809eee973c5/inductor_cache/rh/crhz6qriomrayonuh2bga5myb6lr2yc7e24u2gmha3kgrzwkqodw.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 8)
        x2 = xindex // 512
        x3 = xindex // 64
        tmp0 = tl.load(in_ptr0 + (2048 + x0 + 128*x1 + 4096*x2), xmask).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
        tmp20 = tl.load(in_ptr0 + (2112 + x0 + 128*x1 + 4096*x2), xmask).to(tl.float32)
        tmp24 = tl.load(in_ptr2 + (64 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tl.full([1], 128.0, tl.float32)
        tmp4 = (tmp2 / tmp3)
        tmp5 = tl.full([1], 1e-06, tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = libdevice.rsqrt(tmp6)
        tmp8 = tmp1 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 * tmp10
        tmp13 = tl.full([XBLOCK], 40960, tl.int32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp12 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp12)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 40960)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 40960")
        tmp18 = tl.load(in_ptr4 + (x0 + 128*tmp16), xmask).to(tl.float32)
        tmp19 = tmp11 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp7
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 * tmp24
        tmp26 = tl.load(in_ptr4 + (64 + x0 + 128*tmp16), xmask).to(tl.float32)
        tmp27 = tmp25 * tmp26
        tmp28 = tmp19 - tmp27
        tmp29 = tmp25 * tmp18
        tmp30 = tmp11 * tmp26
        tmp31 = tmp29 + tmp30
        tl.store(out_ptr0 + (x0 + 128*x3), tmp28, xmask)
        tl.store(out_ptr1 + (x0 + 128*x3), tmp31, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 64)
        x5 = ((xindex // 64) % 16)
        x6 = xindex // 1024
        x7 = xindex // 64
        tmp32 = tl.load(in_ptr0 + (x4 + 128*x5 + 4096*x6), xmask).to(tl.float32)
        tmp34 = tl.load(in_ptr5 + (x7), xmask, eviction_policy='evict_last')
        tmp42 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp44 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
        tmp52 = tl.load(in_ptr0 + (64 + x4 + 128*x5 + 4096*x6), xmask).to(tl.float32)
        tmp56 = tl.load(in_ptr6 + (64 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp35 = tl.full([1], 128.0, tl.float32)
        tmp36 = (tmp34 / tmp35)
        tmp37 = tl.full([1], 1e-06, tl.float32)
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp33 * tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp43 = tmp41 * tmp42
        tmp45 = tl.full([XBLOCK], 40960, tl.int32)
        tmp46 = tmp44 + tmp45
        tmp47 = tmp44 < 0
        tmp48 = tl.where(tmp47, tmp46, tmp44)
        tl.device_assert(((0 <= tmp48) & (tmp48 < 40960)) | ~(xmask), "index out of bounds: 0 <= tmp48 < 40960")
        tmp50 = tl.load(in_ptr4 + (x4 + 128*tmp48), xmask).to(tl.float32)
        tmp51 = tmp43 * tmp50
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp53 * tmp39
        tmp55 = tmp54.to(tl.float32)
        tmp57 = tmp55 * tmp56
        tmp58 = tl.load(in_ptr4 + (64 + x4 + 128*tmp48), xmask).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp51 - tmp59
        tmp61 = tmp57 * tmp50
        tmp62 = tmp43 * tmp58
        tmp63 = tmp61 + tmp62
        tl.store(out_ptr2 + (x4 + 128*x7), tmp60, xmask)
        tl.store(out_ptr3 + (x4 + 128*x7), tmp63, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((2048, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048, 8, 1), (8, 1, 16384), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((2048, 16, 1), (16, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((2048, 8, 64), (1024, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((2048, 8, 64), (1024, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_9 = rand_strided((2048, 16, 64), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_10 = rand_strided((2048, 16, 64), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 1048576, 2097152,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_4.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        s72 = arg1_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg0_1, (s72, 2048), (2048, 1), 0), reinterpret_tensor(arg2_1, (2048, 2048), (1, 2048), 0), out=buf0)
            del arg0_1
            del arg2_1
            buf2 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace], Original ATen: [vllm_ir.fused_add_rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_0.run(buf0, arg4_1, arg3_1, buf2, s72, 2048, stream=stream0)
            del arg3_1
            buf3 = empty_strided_cuda((s72, 12288), (12288, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, linear_1], Original ATen: [vllm_ir.fused_add_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf2, reinterpret_tensor(arg5_1, (2048, 12288), (1, 2048), 0), out=buf3)
            del arg5_1
            del buf2
            buf4 = empty_strided_cuda((s72, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_1_xnumel = 6144*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, triton_poi_fused_mul_silu_slice_1_xnumel, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, linear_2], Original ATen: [aten.slice, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf4, reinterpret_tensor(arg6_1, (6144, 2048), (1, 6144), 0), out=buf5)
            del arg6_1
            del buf4
            buf6 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            buf8 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1], Original ATen: [vllm_ir.fused_add_rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_2.run(buf5, buf0, arg4_1, arg7_1, buf6, buf8, s72, 2048, stream=stream0)
            del arg4_1
            del arg7_1
            del buf0
            buf9 = empty_strided_cuda((s72, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, linear_3], Original ATen: [vllm_ir.fused_add_rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf8, reinterpret_tensor(arg8_1, (2048, 4096), (1, 2048), 0), out=buf9)
            del arg8_1
            buf10 = empty_strided_cuda((s72, 16, 1), (16, 1, 16*s72), torch.float32)
            buf11 = empty_strided_cuda((s72, 8, 1), (8, 1, 8*s72), torch.float32)
            # Topologically Sorted Source Nodes: [split, view_1, rms_norm_default, view_3, rms_norm_default_1], Original ATen: [aten.split_with_sizes, aten.view, vllm_ir.rms_norm]
            triton_red_fused_3_xnumel_0 = 16*s72
            triton_red_fused_3_xnumel_1 = 8*s72
            stream0 = get_raw_stream(0)
            triton_red_fused_3.run(buf9, buf10, buf11, triton_red_fused_3_xnumel_0, triton_red_fused_3_xnumel_1, stream=stream0)
            buf14 = empty_strided_cuda((s72, 8, 128), (1024, 128, 1), torch.bfloat16)
            buf12 = reinterpret_tensor(buf14, (s72, 8, 64), (1024, 128, 1), 0)  # alias
            buf13 = reinterpret_tensor(buf14, (s72, 8, 64), (1024, 128, 1), 64)  # alias
            buf17 = reinterpret_tensor(buf8, (s72, 16, 128), (2048, 128, 1), 0); del buf8  # reuse
            buf15 = reinterpret_tensor(buf17, (s72, 16, 64), (2048, 128, 1), 0)  # alias
            buf16 = reinterpret_tensor(buf17, (s72, 16, 64), (2048, 128, 1), 64)  # alias
            # Unsorted Source Nodes: [], Original ATen: []
            triton_poi_fused_4_xnumel_0 = 512*s72
            triton_poi_fused_4_xnumel_1 = 1024*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_4.run(buf9, buf11, arg10_1, arg11_1, arg12_1, buf10, arg9_1, buf12, buf13, buf15, buf16, triton_poi_fused_4_xnumel_0, triton_poi_fused_4_xnumel_1, stream=stream0)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg9_1
            del buf10
            del buf11
            buf18 = buf5; del buf5  # reuse
        return (buf14, reinterpret_tensor(buf9, (s72, 8, 128), (4096, 128, 1), 3072), buf17, reinterpret_tensor(buf18, (s72, 16, 128), (2048, 128, 1), 0), buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 16, 128), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 2048
    arg2_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg12_1 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
