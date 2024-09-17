#include "cutlass/gemm/device/gemm.h"
#include <torch/extension.h>


#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


void cutlass_int4_mm(
  int32_t const *A,
  int32_t const *B,
  int32_t *C,
  int M,
  int N,
  int K
) {
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,    // A matrix
    cutlass::int4b_t, cutlass::layout::ColumnMajor, // B matrix
    int32_t, cutlass::layout::RowMajor,             // C matrix
    int32_t,                                        // accumulate dtype
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
  >;
  using GemmCoord = cutlass::gemm::GemmCoord;
  Gemm gemm_op;
  cutlass::Status status = gemm_op({
    {
      static_cast<GemmCoord::Index>(M),
      static_cast<GemmCoord::Index>(N),
      static_cast<GemmCoord::Index>(K)
    },
    {reinterpret_cast<cutlass::int4b_t const *>(A), K},
    {reinterpret_cast<cutlass::int4b_t const *>(B), K},
    {C, N},
    {C, N},
    {1, 0} // epilogue
  });
  CUTLASS_CHECK(status);
}

torch::Tensor int4_mm(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1) * 8;
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());
  cutlass_int4_mm(
    A.data_ptr<int32_t>(),
    B.data_ptr<int32_t>(),
    C.data_ptr<int32_t>(),
    M, N, K
  );
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("int4_mm", &int4_mm); }
