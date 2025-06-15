// https://github.com/NVIDIA/cutlass/blob/v3.9.2/test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op.cu

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
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

torch::Tensor cutlass_fp8_mm(torch::Tensor A, torch::Tensor B)
{
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  torch::Tensor D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));

  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutputTag = cutlass::layout::RowMajor;
  constexpr int AlignmentInput = 128 / cutlass::sizeof_bits<ElementInput>::value;
  constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementOutput, LayoutOutputTag, AlignmentOutput,
      ElementOutput, LayoutOutputTag, AlignmentOutput,
      EpilogueScheduleType>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementInput, cutlass::layout::RowMajor, AlignmentInput,
      ElementInput, cutlass::layout::ColumnMajor, AlignmentInput,
      ElementAccumulator,
      ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ElementInput *>(A.data_ptr()),
          stride_A,
          reinterpret_cast<ElementInput *>(B.data_ptr()),
          stride_B,
      },
      {
          {1.0, 0.0},
          nullptr,
          stride_D,
          reinterpret_cast<ElementOutput *>(D.data_ptr()),
          stride_D,
      }};

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));

  long workspace_size = Gemm::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty({workspace_size}, A.options().dtype(torch::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(stream));

  return D;
}

TORCH_LIBRARY_IMPL(qtrain, CUDA, m)
{
  m.impl("qtrain::cutlass_fp8_mm", &cutlass_fp8_mm);
}
