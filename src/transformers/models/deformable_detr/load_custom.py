import os

def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_kernel")
    src_files = [
        os.path.join(root, filename)
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    load(
        "MultiScaleDeformableAttention",
        src_files,
        # verbose=True,
        with_cuda=True,
        extra_include_paths=[root],
        # build_directory=os.path.dirname(os.path.realpath(__file__)),
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )

    import MultiScaleDeformableAttention as MSDA

    return MSDA
