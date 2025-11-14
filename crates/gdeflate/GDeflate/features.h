#pragma once

#include_next <features.h>

#if defined(__CUDACC__)
// This wrapper used to clamp __GNUC_PREREQ for NVCC, but it now exists solely
// to ensure we continue including the system header while keeping the search
// path override available for other shims.
#endif
