#pragma once

#if defined(__CUDACC__)
#  ifdef __GNUC_PREREQ
#    pragma push_macro("__GNUC_PREREQ")
#    define __NVCC_FLOATN_PREREQ_PUSHED 1
#  else
#    define __NVCC_FLOATN_PREREQ_BACKFILL 1
#  endif
#  if defined(__GNUC__) && defined(__GNUC_MINOR__)
#    undef __GNUC_PREREQ
#    define __GNUC_PREREQ(maj, min) \
       ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((maj) << 16) + (min)) && (maj) < 13)
#  endif
#endif

#include_next <bits/floatn-common.h>

#if defined(__CUDACC__)
#  ifdef __NVCC_FLOATN_PREREQ_PUSHED
#    pragma pop_macro("__GNUC_PREREQ")
#    undef __NVCC_FLOATN_PREREQ_PUSHED
#  elif defined(__NVCC_FLOATN_PREREQ_BACKFILL)
#    undef __NVCC_FLOATN_PREREQ_BACKFILL
#    if defined(__GNUC__) && defined(__GNUC_MINOR__)
#      undef __GNUC_PREREQ
#      define __GNUC_PREREQ(maj, min) \
         ((((__GNUC__) << 16) + (__GNUC_MINOR__)) >= (((maj) << 16) + (min)))
#    endif
#  endif
#endif
