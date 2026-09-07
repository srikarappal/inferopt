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


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/jn/cjngnb5yiwso2utrmpuw4bqygicsvb4fdowej4yj4a4znlol73ey.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_5
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 134221824}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 2048*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 2048*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 2048.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/6g/c6gvh5xzwxfl56jwbv3hh6j5jp7ce56v43vgazfylh7k26wx4dzs.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x0 + 4096*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp76 = tl.load(in_ptr4 + (0))
        tmp77 = tl.broadcast_to(tmp76, [1, 1])
        tmp78 = tl.where(xmask, tmp77, 0.0)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp6 = r0_2
            tmp7 = tl.full([1, 1], 0, tl.int64)
            tmp8 = tmp6 >= tmp7
            tmp9 = tl.full([1, 1], 64, tl.int64)
            tmp10 = tmp6 < tmp9
            tmp11 = tl.load(in_ptr0 + (128*x0 + 4096*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp12 = tmp11.to(tl.float32)
            tmp13 = tl.full([1, 1], 128.0, tl.float32)
            tmp14 = (tmp4 / tmp13)
            tmp15 = tl.full([1, 1], 1e-06, tl.float32)
            tmp16 = tmp14 + tmp15
            tmp17 = libdevice.rsqrt(tmp16)
            tmp18 = tmp12 * tmp17
            tmp19 = tmp18.to(tl.float32)
            tmp20 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp21 = tmp19 * tmp20
            tmp22 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
            tmp23 = tl.full([1, 1], 40960, tl.int32)
            tmp24 = tmp22 + tmp23
            tmp25 = tmp22 < 0
            tmp26 = tl.where(tmp25, tmp24, tmp22)
            tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 40960)) | ~(r0_mask & tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 40960")
            tmp28 = tl.load(in_ptr3 + (tl.broadcast_to(128*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp29 = tmp21 * tmp28
            tmp30 = tl.load(in_ptr0 + (64 + 128*x0 + 4096*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp31 = tmp30.to(tl.float32)
            tmp32 = tmp31 * tmp17
            tmp33 = tmp32.to(tl.float32)
            tmp34 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp35 = tmp33 * tmp34
            tmp36 = tl.load(in_ptr3 + (tl.broadcast_to(64 + 128*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp37 = tmp35 * tmp36
            tmp38 = tmp29 - tmp37
            tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
            tmp40 = tl.where(tmp10, tmp38, tmp39)
            tmp41 = tmp6 >= tmp9
            tmp42 = tl.full([1, 1], 128, tl.int64)
            tmp43 = tmp6 < tmp42
            tmp44 = tl.load(in_ptr0 + (64 + 128*x0 + 4096*x1 + ((-64) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp45 = tmp44.to(tl.float32)
            tmp46 = tl.full([1, 1], 128.0, tl.float32)
            tmp47 = (tmp4 / tmp46)
            tmp48 = tl.full([1, 1], 1e-06, tl.float32)
            tmp49 = tmp47 + tmp48
            tmp50 = libdevice.rsqrt(tmp49)
            tmp51 = tmp45 * tmp50
            tmp52 = tmp51.to(tl.float32)
            tmp53 = tl.load(in_ptr1 + (tl.broadcast_to(64 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp54 = tmp52 * tmp53
            tmp55 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
            tmp56 = tl.full([1, 1], 40960, tl.int32)
            tmp57 = tmp55 + tmp56
            tmp58 = tmp55 < 0
            tmp59 = tl.where(tmp58, tmp57, tmp55)
            tl.device_assert(((0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 40960)) | ~(r0_mask & tmp41 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 40960")
            tmp61 = tl.load(in_ptr3 + (tl.broadcast_to(128*tmp59 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp62 = tmp54 * tmp61
            tmp63 = tl.load(in_ptr0 + (128*x0 + 4096*x1 + ((-64) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp64 = tmp63.to(tl.float32)
            tmp65 = tmp64 * tmp50
            tmp66 = tmp65.to(tl.float32)
            tmp67 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp68 = tmp66 * tmp67
            tmp69 = tl.load(in_ptr3 + (tl.broadcast_to(64 + 128*tmp59 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp70 = tmp68 * tmp69
            tmp71 = tmp62 + tmp70
            tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
            tmp73 = tl.where(tmp41, tmp71, tmp72)
            tmp74 = tl.where(tmp10, tmp40, tmp73)
            tmp75 = tmp74.to(tl.float32)
            tmp79 = tl.full([1, 1], 1, tl.int32)
            tmp80 = (tmp79 / tmp78)
            tmp81 = tmp75 * tmp80
            tmp82 = tl.full([1, 1], -448.0, tl.float32)
            tmp83 = triton_helpers.maximum(tmp81, tmp82)
            tmp84 = tl.full([1, 1], 448.0, tl.float32)
            tmp85 = triton_helpers.minimum(tmp83, tmp84)
            tmp86 = tmp85.to(tl.float8e4nv)
            tl.store(out_ptr2 + (r0_2 + 128*x3), tmp86, r0_mask & xmask)
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
        _tmp91 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp87 = tl.load(in_ptr0 + (2048 + r0_6 + 128*x4 + 4096*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp88 = tmp87.to(tl.float32)
            tmp89 = tmp88 * tmp88
            tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
            tmp92 = _tmp91 + tmp90
            _tmp91 = tl.where(r0_mask & xmask, tmp92, _tmp91)
        tmp91 = tl.sum(_tmp91, 1)[:, None]
        tl.store(out_ptr3 + (x7), tmp91, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg_6 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 131072, 65536,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/td/ctdlrbix2d4hzduxtvcwf4mdmrhpbnn7p2cnjkkuaceq2oxb3aeh.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, rms_norm_default_2, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_159
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_4 => mul_83
#   mul_5 => mul_86
#   mul_6 => mul_91
#   mul_7 => mul_94
#   rms_norm_default_2 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   split => split_with_sizes
#   sub_1 => sub_42
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_2 => view_2
# Graph fragment:
#   %mm : Tensor "bf16[s72, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf5 : Tensor "f32[s72, 8, 1][8, 1, 8*s72]cuda:0" = PlaceHolder[target=buf5]
#   %arg6_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "bf16[40960, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%mm, [2048, 1024, 1024], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg8_1, [%arg7_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_2 : Tensor "bf16[s72, 8, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, 8, 128]), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg6_1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_1, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_83 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_86 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_42 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, %mul_86), kwargs = {})
#   %mul_91 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_94 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_159 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %mul_94), kwargs = {})
#   return %sub_42,%add_159
triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 262144, 'x': 50331904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (2048 + x2 + 128*y0 + 4096*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2112 + x2 + 128*y0 + 4096*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (64 + x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 128.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1, 1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tl.full([1, 1], 40960, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 40960)) | ~(ymask), "index out of bounds: 0 <= tmp16 < 40960")
    tmp18 = tl.load(in_ptr4 + (x2 + 128*tmp16), xmask & ymask).to(tl.float32)
    tmp19 = tmp11 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.load(in_ptr4 + (64 + x2 + 128*tmp16), xmask & ymask).to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 - tmp27
    tmp29 = tmp25 * tmp18
    tmp30 = tmp11 * tmp26
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2 + 128*y3), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 128*y3), tmp31, xmask & ymask)
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
arg0_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int32, 0, (8192,))
arg2_1 = generate_example_value((151936, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (151936, 2048))
arg3_1 = generate_example_value((2048,), (1,), 'cuda:0', torch.bfloat16, 0, (2048,))
buf0 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2048))
buf2 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.bfloat16, 0, (8192, 2048))
with torch.cuda._DeviceGuard(0):
    triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, 8192, 2048, stream=stream0)
del arg0_1, arg2_1, arg3_1, buf0, buf2

stream0 = get_raw_stream(0)
buf3 = generate_example_value((8192, 4096), (4096, 1), 'cuda:0', torch.bfloat16, 0, (8192, 4096))
arg5_1 = generate_example_value((128,), (1,), 'cuda:0', torch.bfloat16, 0, (128,))
arg7_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg8_1 = generate_example_value((40960, 128), (128, 1), 'cuda:0', torch.bfloat16, 0, (40960, 128))
arg9_1 = generate_example_value((), (), 'cuda:0', torch.float32, 0, ())
buf10 = generate_example_value((8192, 2048), (2048, 1), 'cuda:0', torch.float8_e4m3fn, 0, (8192, 2048))
buf5 = generate_example_value((8192, 8, 1), (8, 1, 65536), 'cuda:0', torch.float32, 0, (8192, 8, 1))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_1.run(buf3, arg5_1, arg7_1, arg8_1, arg9_1, buf10, buf5, 131072, 65536, stream=stream0)
del arg5_1, arg9_1, buf10

stream0 = get_raw_stream(0)
arg6_1 = generate_example_value((128,), (1,), 'cuda:0', torch.bfloat16, 0, (128,))
buf6 = generate_example_value((8192, 8, 64), (1024, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8, 64))
buf7 = generate_example_value((8192, 8, 64), (1024, 128, 1), 'cuda:0', torch.bfloat16, 0, (8192, 8, 64))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2.run(buf3, buf5, arg6_1, arg7_1, arg8_1, buf6, buf7, 65536, 64, stream=stream0)
del buf3, arg7_1, arg8_1, buf5, arg6_1, buf6, buf7

"""
# AOT ID: ['0_inference']
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


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/jn/cjngnb5yiwso2utrmpuw4bqygicsvb4fdowej4yj4a4znlol73ey.py
# Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
#   rms_norm_default => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, mean_dim_2, mul_tensor_4, mul_tensor_5, pow_tensor_scalar_2, rsqrt_default_2
# Graph fragment:
#   %arg0_1 : Tensor "i32[s72][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg2_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type : Tensor "i64[s72][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_tensor_scalar_2 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default_4, 2), kwargs = {})
#   %mean_dim_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_2, [-1], True), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_2, 1e-06), kwargs = {})
#   %rsqrt_default_2 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_tensor_4 : Tensor "f32[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_4, %rsqrt_default_2), kwargs = {})
#   %convert_element_type_default_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_4, torch.bfloat16), kwargs = {})
#   %mul_tensor_5 : Tensor "bf16[s72, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_5, %arg3_1), kwargs = {})
#   return %embedding,%buf1,%mul_tensor_5
triton_red_fused__to_copy_embedding_rms_norm_0 = async_compile.triton('triton_red_fused__to_copy_embedding_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768, 'r0_': 134221824}}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_1 + 2048*tmp5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_1 + 2048*x0), tmp7, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(out_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.full([1, 1], 2048.0, tl.float32)
        tmp16 = (tmp11 / tmp15)
        tmp17 = tl.full([1, 1], 1e-06, tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp21 * tmp22
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/6g/c6gvh5xzwxfl56jwbv3hh6j5jp7ce56v43vgazfylh7k26wx4dzs.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_red_fused_1 = async_compile.triton('triton_red_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr2': '*fp8e4nv', 'out_ptr3': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_red_fused_1', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
            tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x0 + 4096*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        tmp76 = tl.load(in_ptr4 + (0))
        tmp77 = tl.broadcast_to(tmp76, [1, 1])
        tmp78 = tl.where(xmask, tmp77, 0.0)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp6 = r0_2
            tmp7 = tl.full([1, 1], 0, tl.int64)
            tmp8 = tmp6 >= tmp7
            tmp9 = tl.full([1, 1], 64, tl.int64)
            tmp10 = tmp6 < tmp9
            tmp11 = tl.load(in_ptr0 + (128*x0 + 4096*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp12 = tmp11.to(tl.float32)
            tmp13 = tl.full([1, 1], 128.0, tl.float32)
            tmp14 = (tmp4 / tmp13)
            tmp15 = tl.full([1, 1], 1e-06, tl.float32)
            tmp16 = tmp14 + tmp15
            tmp17 = libdevice.rsqrt(tmp16)
            tmp18 = tmp12 * tmp17
            tmp19 = tmp18.to(tl.float32)
            tmp20 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp21 = tmp19 * tmp20
            tmp22 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
            tmp23 = tl.full([1, 1], 40960, tl.int32)
            tmp24 = tmp22 + tmp23
            tmp25 = tmp22 < 0
            tmp26 = tl.where(tmp25, tmp24, tmp22)
            tl.device_assert(((0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 40960)) | ~(r0_mask & tmp10 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK]) < 40960")
            tmp28 = tl.load(in_ptr3 + (tl.broadcast_to(128*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp29 = tmp21 * tmp28
            tmp30 = tl.load(in_ptr0 + (64 + 128*x0 + 4096*x1 + (r0_2)), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp31 = tmp30.to(tl.float32)
            tmp32 = tmp31 * tmp17
            tmp33 = tmp32.to(tl.float32)
            tmp34 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp35 = tmp33 * tmp34
            tmp36 = tl.load(in_ptr3 + (tl.broadcast_to(64 + 128*tmp26 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp37 = tmp35 * tmp36
            tmp38 = tmp29 - tmp37
            tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
            tmp40 = tl.where(tmp10, tmp38, tmp39)
            tmp41 = tmp6 >= tmp9
            tmp42 = tl.full([1, 1], 128, tl.int64)
            tmp43 = tmp6 < tmp42
            tmp44 = tl.load(in_ptr0 + (64 + 128*x0 + 4096*x1 + ((-64) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp45 = tmp44.to(tl.float32)
            tmp46 = tl.full([1, 1], 128.0, tl.float32)
            tmp47 = (tmp4 / tmp46)
            tmp48 = tl.full([1, 1], 1e-06, tl.float32)
            tmp49 = tmp47 + tmp48
            tmp50 = libdevice.rsqrt(tmp49)
            tmp51 = tmp45 * tmp50
            tmp52 = tmp51.to(tl.float32)
            tmp53 = tl.load(in_ptr1 + (tl.broadcast_to(64 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp54 = tmp52 * tmp53
            tmp55 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
            tmp56 = tl.full([1, 1], 40960, tl.int32)
            tmp57 = tmp55 + tmp56
            tmp58 = tmp55 < 0
            tmp59 = tl.where(tmp58, tmp57, tmp55)
            tl.device_assert(((0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 40960)) | ~(r0_mask & tmp41 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp59, [XBLOCK, R0_BLOCK]) < 40960")
            tmp61 = tl.load(in_ptr3 + (tl.broadcast_to(128*tmp59 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp62 = tmp54 * tmp61
            tmp63 = tl.load(in_ptr0 + (128*x0 + 4096*x1 + ((-64) + r0_2)), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp64 = tmp63.to(tl.float32)
            tmp65 = tmp64 * tmp50
            tmp66 = tmp65.to(tl.float32)
            tmp67 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp68 = tmp66 * tmp67
            tmp69 = tl.load(in_ptr3 + (tl.broadcast_to(64 + 128*tmp59 + ((-64) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp70 = tmp68 * tmp69
            tmp71 = tmp62 + tmp70
            tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
            tmp73 = tl.where(tmp41, tmp71, tmp72)
            tmp74 = tl.where(tmp10, tmp40, tmp73)
            tmp75 = tmp74.to(tl.float32)
            tmp79 = tl.full([1, 1], 1, tl.int32)
            tmp80 = (tmp79 / tmp78)
            tmp81 = tmp75 * tmp80
            tmp82 = tl.full([1, 1], -448.0, tl.float32)
            tmp83 = triton_helpers.maximum(tmp81, tmp82)
            tmp84 = tl.full([1, 1], 448.0, tl.float32)
            tmp85 = triton_helpers.minimum(tmp83, tmp84)
            tmp86 = tmp85.to(tl.float8e4nv)
            tl.store(out_ptr2 + (r0_2 + 128*x3), tmp86, r0_mask & xmask)
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
        _tmp91 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp87 = tl.load(in_ptr0 + (2048 + r0_6 + 128*x4 + 4096*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp88 = tmp87.to(tl.float32)
            tmp89 = tmp88 * tmp88
            tmp90 = tl.broadcast_to(tmp89, [XBLOCK, R0_BLOCK])
            tmp92 = _tmp91 + tmp90
            _tmp91 = tl.where(r0_mask & xmask, tmp92, _tmp91)
        tmp91 = tl.sum(_tmp91, 1)[:, None]
        tl.store(out_ptr3 + (x7), tmp91, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg_6 = rand_strided((8192, 8, 1), (8, 1, 65536), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 131072, 65536,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_red_fused_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_red_fused_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


# kernel path: runs/h100-1.7b-lossy/.vllm_cache/gpu0/torch_compile_cache/torch_aot_compile/6d8d49b2166f510ee9a6ffc9925d50879da5fa374fb7f4f8b912113ca757a7f4/inductor_cache/td/ctdlrbix2d4hzduxtvcwf4mdmrhpbnn7p2cnjkkuaceq2oxb3aeh.py
# Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, rms_norm_default_2, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_159
#   chunk => split
#   chunk_2 => split_2
#   index_select => index
#   mul_4 => mul_83
#   mul_5 => mul_86
#   mul_6 => mul_91
#   mul_7 => mul_94
#   rms_norm_default_2 => add_tensor, convert_element_type_default, convert_element_type_default_1, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   split => split_with_sizes
#   sub_1 => sub_42
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
#   view_2 => view_2
# Graph fragment:
#   %mm : Tensor "bf16[s72, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm]
#   %buf5 : Tensor "f32[s72, 8, 1][8, 1, 8*s72]cuda:0" = PlaceHolder[target=buf5]
#   %arg6_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "i64[s72][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "bf16[40960, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%mm, [2048, 1024, 1024], -1), kwargs = {})
#   %index : Tensor "bf16[s72, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg8_1, [%arg7_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_2 : Tensor "bf16[s72, 8, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg1_1, 8, 128]), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_default, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_1 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.bfloat16), kwargs = {})
#   %mul_tensor_1 : Tensor "bf16[s72, 8, 128][1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %arg6_1), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%mul_tensor_1, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_83 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s72, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_86 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_42 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, %mul_86), kwargs = {})
#   %mul_91 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_94 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_159 : Tensor "bf16[s72, 8, 64][512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %mul_94), kwargs = {})
#   return %sub_42,%add_159
triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 262144, 'x': 50331904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 8)
    y1 = yindex // 8
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (2048 + x2 + 128*y0 + 4096*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (y1), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (2112 + x2 + 128*y0 + 4096*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (64 + x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 128.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1, 1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp13 = tl.full([1, 1], 40960, tl.int32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp12 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp12)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 40960)) | ~(ymask), "index out of bounds: 0 <= tmp16 < 40960")
    tmp18 = tl.load(in_ptr4 + (x2 + 128*tmp16), xmask & ymask).to(tl.float32)
    tmp19 = tmp11 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.load(in_ptr4 + (64 + x2 + 128*tmp16), xmask & ymask).to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 - tmp27
    tmp29 = tmp25 * tmp18
    tmp30 = tmp11 * tmp26
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2 + 128*y3), tmp28, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 128*y3), tmp31, xmask & ymask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s72 = arg1_1
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            buf2 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [long, embedding, rms_norm_default], Original ATen: [aten._to_copy, aten.embedding, vllm_ir.rms_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_embedding_rms_norm_0.run(arg0_1, arg2_1, arg3_1, buf0, buf2, s72, 2048, stream=stream0)
            del arg0_1
            del arg2_1
            del arg3_1
            buf3 = empty_strided_cuda((s72, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [rms_norm_default, linear], Original ATen: [vllm_ir.rms_norm, aten.t, aten.mm]
            extern_kernels.mm(buf2, reinterpret_tensor(arg4_1, (2048, 4096), (1, 2048), 0), out=buf3)
            del arg4_1
            del buf2
            buf10 = empty_strided_cuda((s72, 2048), (2048, 1), torch.float8_e4m3fn)
            buf5 = empty_strided_cuda((s72, 8, 1), (8, 1, 8*s72), torch.float32)
            # Topologically Sorted Source Nodes: [split, view_2, rms_norm_default_2], Original ATen: [aten.split_with_sizes, aten.view, vllm_ir.rms_norm]
            triton_red_fused_1_xnumel_0 = 16*s72
            triton_red_fused_1_xnumel_1 = 8*s72
            stream0 = get_raw_stream(0)
            triton_red_fused_1.run(buf3, arg5_1, arg7_1, arg8_1, arg9_1, buf10, buf5, triton_red_fused_1_xnumel_0, triton_red_fused_1_xnumel_1, stream=stream0)
            del arg5_1
            del arg9_1
            buf8 = empty_strided_cuda((s72, 8, 128), (1024, 128, 1), torch.bfloat16)
            buf6 = reinterpret_tensor(buf8, (s72, 8, 64), (1024, 128, 1), 0)  # alias
            buf7 = reinterpret_tensor(buf8, (s72, 8, 64), (1024, 128, 1), 64)  # alias
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_2, rms_norm_default_2, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, sub_1, mul_6, mul_7, add_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, vllm_ir.rms_norm, aten.unsqueeze, aten.mul, aten.sub, aten.add]
            triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2_ynumel = 8*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2.run(buf3, buf5, arg6_1, arg7_1, arg8_1, buf6, buf7, triton_poi_fused_add_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_2_ynumel, 64, stream=stream0)
            del arg6_1
            del arg7_1
            del arg8_1
            del buf5
            buf11 = empty_strided_cuda((s72, 2048), (2048, 1), torch.bfloat16)
        return (buf8, reinterpret_tensor(buf3, (s72, 8, 128), (4096, 128, 1), 3072), reinterpret_tensor(buf10, (s72, 16, 128), (2048, 128, 1), 0), reinterpret_tensor(buf11, (s72, 16, 128), (2048, 128, 1), 0), buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((151936, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4096, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg8_1 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
