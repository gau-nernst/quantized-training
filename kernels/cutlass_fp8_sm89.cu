#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#define CUTLASS_CHECK(status) \
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass error: ", cutlassGetStatusString(status))


// define common params
using ElementA           = cutlass::float_e4m3_t;
using ElementB           = cutlass::float_e4m3_t;
using ElementC           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using OpClass            = cutlass::arch::OpClassTensorOp;
using ArchTag            = cutlass::arch::Sm89;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

torch::Tensor fp8_mm(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));

  // TODO: use better config
  // static int const kStages = 3;
  using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,    // A matrix
    ElementB, cutlass::layout::ColumnMajor, // B matrix
    ElementC, cutlass::layout::RowMajor,    // C matrix
    ElementAccumulator, OpClass, ArchTag
  >;
  Gemm::Arguments args {
    {M, N, K},
    {reinterpret_cast<ElementA *>(A.data_ptr()), K},
    {reinterpret_cast<ElementB *>(B.data_ptr()), K},
    {reinterpret_cast<ElementC *>(C.data_ptr()), N},
    {reinterpret_cast<ElementC *>(C.data_ptr()), N},
    {1, 0}  // epilogue
  };
  Gemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm_op(args, nullptr, stream));

  return C;
}

// this function is based on the following cutlass example
// https://github.com/NVIDIA/cutlass/blob/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu
// also with the help of emitted code from cutlass Python
torch::Tensor scaled_fp8_mm(torch::Tensor A, torch::Tensor B, torch::Tensor row_scale, torch::Tensor col_scale) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, row_scale.options());

  using namespace cute;

  // https://github.com/NVIDIA/cutlass/blob/v3.9.2/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  constexpr int numStages = 3;
  constexpr int numEpilogueStages = 1;

  // build epilogue visitor tree
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, numEpilogueStages
  >;

  using ElementEpilogue = float;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  constexpr auto RoundMode = cutlass::FloatRoundStyle::round_to_nearest;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementEpilogue, ElementEpilogue, RoundMode
  >;

  // (1, N)
  using ColScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<_0, _1, int32_t>  // MNL
  >;
  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accum, ColScale>;

  // (M, 1)
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<_1, _0, int32_t>  // MNL
  >;
  using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<Multiply, EVTCompute0, RowScale>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC, RoundMode,
    cute::Stride<int64_t, _1, int64_t>  // MNL
  >;
  using EVTOutput = cutlass::epilogue::threadblock::Sm80EVT<Output, EVTCompute1>;

  using EVTKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, cutlass::layout::RowMajor,    cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, cutlass::layout::RowMajor,                                      AlignmentC,
    ElementAccumulator, ElementEpilogue, OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVTOutput,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    numStages,
    cutlass::arch::OpMultiplyAdd,
    numEpilogueStages
  >::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<EVTKernel>;

  const ElementA *A_ptr         = reinterpret_cast<ElementA *>(A.data_ptr());
  const ElementB *B_ptr         = reinterpret_cast<ElementB *>(B.data_ptr());
  const ElementC *col_scale_ptr = reinterpret_cast<ElementC *>(col_scale.data_ptr());
  const ElementC *row_scale_ptr = reinterpret_cast<ElementC *>(row_scale.data_ptr());
  ElementC *C_ptr               = reinterpret_cast<ElementC *>(C.data_ptr());

  typename EVTOutput::Arguments callback_args{
    {
      {
        {},                                                      // Accum
        {col_scale_ptr, ElementC(0), {_0{}, _1{}, int32_t(N)}},  // ColScale
        {}                                                       // Multiply
      },                                                         // EVTCompute0
      {row_scale_ptr, ElementC(0), {_1{}, _0{}, int32_t(M)}},    // RowScale
      {}                                                         // Multiply
    },                                                           // EVTCompute1
    {C_ptr, {int64_t{N}, _1{}, int64_t{M*N}}}                    // EVTOutput
  };

  typename DeviceGemm::Arguments args(
    cutlass::gemm::GemmUniversalMode::kGemm,
    cutlass::gemm::GemmCoord{M, N, K},
    1,                              // batch_split
    callback_args,
    A_ptr, B_ptr, nullptr, nullptr, // unsued C_ptr and D_ptr
    M * K, N * K, 0, 0,             // batch_stride A, B, C, D
    K, K, 0, 0                      // stride A, B, C, D
  );

  DeviceGemm gemm_op;
  auto stream = at::cuda::getCurrentCUDAStream();
  CUTLASS_CHECK(gemm_op.can_implement(args));
  CUTLASS_CHECK(gemm_op(args, nullptr, stream));

  return C;
}

TORCH_LIBRARY_IMPL(qtrain, CUDA, m) {
  m.impl("qtrain::fp8_mm", &fp8_mm);
  m.impl("qtrain::scaled_fp8_mm", &scaled_fp8_mm);
}
