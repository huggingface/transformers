"""Workarounds for two Inductor bugs on MPS (torch 2.15 nightly) that Inkling2Loop hits, each a candidate upstream fix.

1. `Identity(Identity(0))` is a number but not comparable, so sympy's Min / Max reject an index such as
   `Max(64 - Identity(0), ...)` ("The argument '64 - ((0))' is not comparable"). Identity of a literal is the literal.
2. `cpp_wrapper=True` (JIT): the `get_mps_lib_N_handle()` getters are emitted by
   `CppWrapperMps.codegen_additional_funcs`, which only the AOTI entry point calls, so the JIT wrapper uses undefined
   getters ("use of undeclared identifier 'get_mps_lib_0_handle'"). Define each getter right after its library's
   source, at file scope.
"""

import sympy
from torch.utils._sympy.functions import Identity


def apply():
    Identity.eval = classmethod(lambda cls, arg: arg if isinstance(arg, sympy.Number) else None)

    from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps

    define_kernel = CppWrapperMps.define_kernel

    def define_kernel_with_getter(self, kernel_name, kernel_body, *args, **kwargs):
        if kernel_name.startswith("mps_lib_") and kernel_body.startswith(f"const char* {kernel_name}_source"):
            lib_name = kernel_name
            self._used_kernel_names.add(lib_name)  # codegen_additional_funcs (AOTI) must not define it again
            kernel_body = kernel_body.rstrip() + f"""

AOTIMetalKernelFunctionHandle get_{lib_name}_handle() {{
    static auto kernel_handle = []() {{
        AOTIMetalShaderLibraryHandle lib_handle = nullptr;
        AOTIMetalKernelFunctionHandle kern_handle = nullptr;
        auto lib_deleter = [](AOTIMetalShaderLibraryHandle h) {{
            if (h) aoti_torch_mps_delete_shader_library(h);
        }};
        using LibPtr = std::unique_ptr<AOTIMetalShaderLibraryOpaque, decltype(lib_deleter)>;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_create_shader_library({lib_name}_source, &lib_handle));
        LibPtr lib(lib_handle, lib_deleter);
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_get_kernel_function(lib_handle, "generated_kernel", &kern_handle));
        return std::make_pair(kern_handle, std::move(lib));
    }}();
    return kernel_handle.first;
}}
"""
        return define_kernel(self, kernel_name, kernel_body, *args, **kwargs)

    CppWrapperMps.define_kernel = define_kernel_with_getter
