
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'AE9C989C502A611D3F269B64D3068764F09C37B597919243C4BB6E23C3E0E199', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'kernel_num_gb': 0.179634432, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 40)
    x2 = xindex // 2560
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 7168*x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 7168*x2), xmask).to(tl.float32)
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


def get_args():
    arg_0 = rand_strided((8192, 7168), (7168, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 40, 1), (40, 1, 327680), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((40960, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 40, 64), (5120, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((8192, 40, 64), (5120, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 20971520,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 0.179634432
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
