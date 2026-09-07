
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
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
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
        triton_poi_fused_2.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_2.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
