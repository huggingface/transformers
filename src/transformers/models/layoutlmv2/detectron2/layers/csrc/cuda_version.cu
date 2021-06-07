// Copyright (c) Facebook, Inc. and its affiliates.

#include <cuda_runtime_api.h>

namespace detectron2 {
int get_cudart_version() {
// Not a ROCM platform: Either HIP is not used, or
// it is used, but platform is not ROCM (i.e. it is CUDA)
#if !defined(__HIP_PLATFORM_HCC__)
  return CUDART_VERSION;
#else
  int version = 0;

#if HIP_VERSION_MAJOR != 0
  // Create a convention similar to that of CUDA, as assumed by other
  // parts of the code.

  version = HIP_VERSION_MINOR;
  version += (HIP_VERSION_MAJOR * 100);
#else
  hipRuntimeGetVersion(&version);
#endif
  return version;
#endif
}
} // namespace detectron2
