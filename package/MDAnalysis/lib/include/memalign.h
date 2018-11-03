/* -*- Mode: C; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*- */
/* vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 */
/*
 * MDAnalysis --- https://www.mdanalysis.org
 * Copyright (c) 2006-2018 The MDAnalysis Development Team and contributors
 * (see the file AUTHORS for the full list of names)
 *
 * Released under the GNU Public Licence, v2 or any higher version
 *
 * Please cite your use of MDAnalysis in published work:
 *
 * R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
 * D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
 * MDAnalysis: A Python package for the rapid analysis of molecular dynamics
 * simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
 * Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
 *
 * N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
 * MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
 * J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
 */
#ifndef __ALIGNMENT_H
#define __ALIGNMENT_H

/*
 * Memory alignment macros to support cache-aligned operations and
 * facilitate auto-vectorization of loops using SSE and/or AVX.
 */
// Up to AVX-512, 64 byte alignment is sufficient, and the vast majority of
// modern CPUs have 64 byte cache lines. For future CPUs with larger vector
// units, this number can be adjusted accordingly.
#define MEMORY_ALIGNMENT 64
// If we're on a *nix system, try to use POSIX memory alignment features.
// Reason: Even if the compiler supports C11, an old glibc might be installed
// that doesn't provide the required C11 functionality (as it is the case for
// CentOS 6).
#ifdef __unix__
    #if (!defined(_POSIX_C_SOURCE)) || (_POSIX_C_SOURCE < 200112L)
        #define _POSIX_C_SOURCE 200112L
    #endif
    #define USE_POSIX_ALIGNMENT
// Otherwise, check if we have C11 support and hope for the best.
#elif __STDC_VERSION__ >= 201112L
   #define USE_C11_ALIGNMENT
#endif
#if defined USE_C11_ALIGNMENT || defined USE_POSIX_ALIGNMENT
    #define USE_ALIGNMENT
#endif
#ifdef USE_ALIGNMENT
// Now we're going to define compiler-specific alignment macros.
    #ifndef __clang__
        #define __has_builtin(X) 0
    #endif
// Intel-specific alignment macros
    #if (defined __INTEL_COMPILER) || (defined __ICL)
        #define __attaligned \
                __attribute__((aligned(MEMORY_ALIGNMENT)))
// Unlike GCC's __builtin_assume_aligned, Intel's __assume_aligned macro doesn't
// return its first argument, so we keep it disabled for now.
        #define __assaligned(X) (X)
// GCC >= 4.7 and Clang-specific alignment macros:
    #elif (defined __GNUC__ && ((__GNUC__ > 4) || \
          ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 7)))) || \
          (defined __clang__ && (__has_builtin(__builtin_assume_aligned)))
        #define __attaligned \
                __attribute__((__aligned__(MEMORY_ALIGNMENT)))
        #define __assaligned(X) \
                (__builtin_assume_aligned((X), MEMORY_ALIGNMENT))
// Disable alignment macros for all other compilers:
    #else
        #define __attaligned
        #define __assaligned(X) (X)
    #endif
// If C11 is supported, use _Alignas() macro:
    #if __STDC_VERSION__ >= 201112L
        #define __memaligned _Alignas(MEMORY_ALIGNMENT)
// Otherwise, disable the macro:
    #else
        #define __memaligned
    #endif
// If we cannot use memory alignment at all, disable all related macros:
#else
    #define __attaligned
    #define __assaligned(X) (X)
    #define __memaligned
#endif

// USED_MEMALIGN macro for Cython to check if aligned memory is used
// (similar to the USED_OPENMP macro in calc_distances)
#ifdef USE_ALIGNMENT
    #define USED_MEMALIGN 1
#else
    #define USED_MEMALIGN 0
#endif

#include "static_assert.h"
#include <stdlib.h>
#include <string.h>

static_assert(((MEMORY_ALIGNMENT > 0) && \
              !(MEMORY_ALIGNMENT & (MEMORY_ALIGNMENT - 1))), \
              "MEMORY_ALIGNMENT is not a positive power of 2");

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Memory-aligned malloc
 *
 * Uses either C11 aligned_alloc() or POSIX posix_memalign() to yield a
 * memory-aligned data field of @p size bytes. Its interface is the same as for
 * malloc(), and memory alignment is handled automatically. In case of failure,
 * the returned pointer will be NULL.
 * If C11 or POSIX features are not available, good ol' malloc() is used and
 * alignment is not guaranteed. If this is the case, the macro @c USE_ALIGNMENT
 * will be undefined, i.e., it can be used to check for guaranteed alignment.
 */
static inline void* aligned_malloc(size_t size)
{
#ifdef USE_ALIGNMENT
    #ifdef USE_C11_ALIGNMENT
    return aligned_alloc(MEMORY_ALIGNMENT, size);
    #elif defined USE_POSIX_ALIGNMENT
    void* ptr;
    int failure = posix_memalign(&ptr, MEMORY_ALIGNMENT, size);
    if (failure) ptr = NULL;
    return ptr;
    #endif
#else
    return malloc(size);
#endif
}

/**
 * @brief Memory-aligned calloc
 *
 * Neither C11 nor POSIX offer a memory-aligned calloc() routine, so here's our
 * own. Its interface is the same as for calloc(), and memory alignment is
 * handled automatically. In case of failure, the returned pointer will be NULL.
 * If C11 or POSIX features are not available, good ol' calloc() is used and
 * alignment is not guaranteed. If this is the case, the macro @c USE_ALIGNMENT
 * will be undefined, i.e., it can be used to check for guaranteed alignment.
 */
static inline void* aligned_calloc(size_t num, size_t size)
{
#ifdef USE_ALIGNMENT
    size *= num;
    void* ptr = aligned_malloc(size);
    // If memory allocation failed (ptr is NULL), stop here:
    if (!ptr) return ptr;
    // Clear the allocated memory using memset():
    // This is certainly not the fastest solution, but it is portable.
    return memset(ptr, 0, size);
#else
    return calloc(num, size);
#endif
}

#ifdef __cplusplus
}
#endif

#endif /*__ALIGNMENT_H*/
