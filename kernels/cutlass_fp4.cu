// CUTLASS example 79

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
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

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideD   = typename Gemm::GemmKernel::StrideD;

torch::Tensor nvfp4_mm(torch::Tensor A, torch::Tensor B, torch::Tensor scales_A, torch::Tensor scales_B) {
  int M = A.size(0);
  int K = A.size(1) * 2;
  int N = B.size(1);

  torch::Tensor D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
  auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
  auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    { // Mainloop arguments
      reinterpret_cast<cutlass::float_e2m1_t *>(A.data_ptr()), stride_A,
      reinterpret_cast<cutlass::float_e2m1_t *>(B.data_ptr()), stride_B,
      reinterpret_cast<cutlass::float_ue4m3_t *>(scales_A.data_ptr()), layout_SFA,
      reinterpret_cast<cutlass::float_ue4m3_t *>(scales_B.data_ptr()), layout_SFB,
    },
    { // Epilogue arguments
      {1.0, 0.0},
      nullptr, stride_D,
      reinterpret_cast<ElementD *>(D.data_ptr()), stride_D,
    }
  };

  Gemm gemm;
  CUTLASS_CHECK(gemm.can_implement(arguments));

  long workspace_size = Gemm::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty({workspace_size}, A.options().dtype(torch::kUInt8));
  auto stream = at::cuda::getCurrentCUDAStream();

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run());

  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_mm", &nvfp4_mm);
}
