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


// we will do input checks in python
torch::Tensor int4_mm(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1) * 8;  // 8x 4-bit in 32-bit
  int N = B.size(1);
  torch::Tensor C = torch::empty({M, N}, A.options());

  cutlass::gemm::device::Gemm<
    cutlass::int4b_t, cutlass::layout::RowMajor,    // A matrix
    cutlass::int4b_t, cutlass::layout::ColumnMajor, // B matrix
    int32_t,          cutlass::layout::RowMajor,    // C matrix
    int32_t,                                        // accumulate dtype
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
  > gemm_op;
  cutlass::Status status = gemm_op({
    {M, N, K},
    {reinterpret_cast<cutlass::int4b_t const *>(A.data_ptr<int32_t>()), K},
    {reinterpret_cast<cutlass::int4b_t const *>(B.data_ptr<int32_t>()), K},
    {C.data_ptr<int32_t>(), N},
    {C.data_ptr<int32_t>(), N},
    {1, 0} // epilogue
  });
  CUTLASS_CHECK(status);

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("int4_mm", &int4_mm); }
