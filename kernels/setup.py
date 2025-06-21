from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

CURRENT_DIR = Path(__file__).parent


def get_extension(arch: str):
    return CUDAExtension(
        name=f"kernels.cutlass_sm{arch}",
        sources=list(CURRENT_DIR.glob(f"csrc/cutlass_sm{arch}_*.cu")),
        py_limited_api=True,
        extra_compile_args=dict(
            nvcc=[
                f"-I{CURRENT_DIR / 'cutlass/include'}",
                f"-I{CURRENT_DIR / 'cutlass/tools/util/include'}",
                # compute_xx is PTX, sm_xx is SASS
                f"-gencode=arch=compute_{arch},code=compute_{arch}",
                # "-DCUTLASS_DEBUG_TRACE_LEVEL=1",
            ]
        ),
        define_macros=[("Py_LIMITED_API", "0x03090000")],
    )


setup(
    name="kernels",
    packages=find_packages(),
    version="0.1",
    ext_modules=[
        get_extension("80"),
        get_extension("89"),
        get_extension("120a"),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
