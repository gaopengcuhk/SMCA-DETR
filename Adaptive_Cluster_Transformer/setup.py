from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def get_extensions():
 
    return extensions


if __name__ == "__main__":
    extensions = [
        CUDAExtension(
            "ACT.extensions.broadcast",
            sources=[
                "ACT/extensions/broadcast.cu"
            ],
            extra_compile_args=["-arch=compute_50"]
        ),
        CUDAExtension(
            "ACT.extensions.ada_cluster",
            sources=[
                "ACT/extensions/ada_cluster.cu"
            ],
            extra_compile_args=["-arch=compute_50"]
        ),
        CUDAExtension(
            "ACT.extensions.weighted_sum",
            sources=[
                "ACT/extensions/weighted_sum.cu"
            ],
            extra_compile_args=["-arch=compute_50"]
        )
    ]

    setup(
        name="ACT",
        packages=find_packages(),
        ext_modules=extensions,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"]
    )
