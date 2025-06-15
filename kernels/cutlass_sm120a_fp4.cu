// CUTLASS example 79

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CUTLASS_CHECK(status) \
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass error: ", cutlassGetStatusString(status))

using namespace cute;

using ElementD           = cutlass::bfloat16_t;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ElementAccumulator = float;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

#define DEFINE_FP4_MM(prefix)                                                                                                    \
  torch::Tensor prefix##fp4_mm(torch::Tensor A, torch::Tensor B, torch::Tensor scales_A, torch::Tensor scales_B)                 \
  {                                                                                                                              \
    int M = A.size(0);                                                                                                           \
    int K = A.size(1) * 2;                                                                                                       \
    int N = B.size(1);                                                                                                           \
    torch::Tensor D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));                                                 \
                                                                                                                                 \
    using ElementAB = cutlass::prefix##_float4_t<cutlass::float_e2m1_t>;                                                         \
                                                                                                                                 \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<                                        \
        ArchTag, OperatorClass,                                                                                                  \
        ThreadBlockShape, ClusterShape,                                                                                          \
        cutlass::epilogue::collective::EpilogueTileAuto,                                                                         \
        ElementAccumulator, ElementAccumulator,                                                                                  \
        ElementD, cutlass::layout::RowMajor, AlignmentD,                                                                         \
        ElementD, cutlass::layout::RowMajor, AlignmentD,                                                                         \
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;                                                      \
                                                                                                                                 \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<                                            \
        ArchTag, OperatorClass,                                                                                                  \
        ElementAB, cutlass::layout::RowMajor, 32,                                                                                \
        ElementAB, cutlass::layout::ColumnMajor, 32,                                                                             \
        ElementAccumulator,                                                                                                      \
        ThreadBlockShape, ClusterShape,                                                                                          \
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, \
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;                                                            \
                                                                                                                                 \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                                                                     \
        Shape<int, int, int, int>,                                                                                               \
        CollectiveMainloop,                                                                                                      \
        CollectiveEpilogue,                                                                                                      \
        void>;                                                                                                                   \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                                                        \
                                                                                                                                 \
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;                                  \
    using DataType             = typename ElementAB::DataType;                                                                   \
    using ScaleFactorType      = typename ElementAB::ScaleFactorType;                                                            \
                                                                                                                                 \
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});                                 \
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});                                 \
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});                                 \
                                                                                                                                 \
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));                                \
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));                                \
                                                                                                                                 \
    typename Gemm::Arguments arguments{                                                                                          \
        cutlass::gemm::GemmUniversalMode::kGemm,                                                                                 \
        {M, N, K, 1},                                                                                                            \
        {                                                                                                                        \
            reinterpret_cast<DataType *>(A.data_ptr()),                                                                          \
            stride_A,                                                                                                            \
            reinterpret_cast<DataType *>(B.data_ptr()),                                                                          \
            stride_B,                                                                                                            \
            reinterpret_cast<ScaleFactorType *>(scales_A.data_ptr()),                                                            \
            layout_SFA,                                                                                                          \
            reinterpret_cast<ScaleFactorType *>(scales_B.data_ptr()),                                                            \
            layout_SFB,                                                                                                          \
        },                                                                                                                       \
        {                                                                                                                        \
            {1.0, 0.0},                                                                                                          \
            nullptr,                                                                                                             \
            stride_D,                                                                                                            \
            reinterpret_cast<ElementD *>(D.data_ptr()),                                                                          \
            stride_D,                                                                                                            \
        }};                                                                                                                      \
                                                                                                                                 \
    Gemm gemm;                                                                                                                   \
    CUTLASS_CHECK(gemm.can_implement(arguments));                                                                                \
                                                                                                                                 \
    long workspace_size = Gemm::get_workspace_size(arguments);                                                                   \
    torch::Tensor workspace = torch::empty({workspace_size}, A.options().dtype(torch::kByte));                                   \
    auto stream = at::cuda::getCurrentCUDAStream();                                                                              \
                                                                                                                                 \
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));                                                     \
    CUTLASS_CHECK(gemm.run(stream));                                                                                             \
                                                                                                                                 \
    return D;                                                                                                                    \
  }

DEFINE_FP4_MM(nv);
DEFINE_FP4_MM(mx);

TORCH_LIBRARY_IMPL(qtrain, CUDA, m)
{
  m.impl("qtrain::nvfp4_mm", &nvfp4_mm);
  m.impl("qtrain::mxfp4_mm", &mxfp4_mm);
}
