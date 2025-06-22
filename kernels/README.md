# My kernels collection

Kernels in this folder can be installed as a package to provide code sharing for my projects.

```bash
uv pip install git+https://github.com/gau-nernst/quantized-training#subdirectory=kernels --no-build-isolation
```

Available kernels

- SM80: Cutlass INT4 + rowwise-scaled INT4
- SM89: Cutlass FP8 + rowwise-scaled FP8
- SM120:
  - Cutlass FP8 + rowwise-scaled FP8
  - Cutlass FP4 + rowwise-scaled FP4
- Triton:
  - Matmul with configurable input dtype, accumulate dtype e.g. FP16 MMA with FP16 accumulate
  - Rowwise-scaled matmul
  - Tile-scaled matmul (i.e. DeepSeek style)
  - Conv2d (unmaintained)
