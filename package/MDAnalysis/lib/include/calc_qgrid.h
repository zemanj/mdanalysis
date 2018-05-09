#ifndef __QGRID_H
#define __QGRID_H

/*
 * Memory alignment macros to support auto-vectorization using SSE and/or AVX
 */
// Up to AVX-512, 64 byte alignment is sufficient. For future CPUs with larger
// vector units, this number can be adjusted accordingly.
#define __MEMORY_ALIGNMENT 64
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
// Now we're going to define compiler-specific alignment macros:
#ifndef __clang__
    #define __has_builtin(X) 0
#endif
#ifdef USE_ALIGNMENT
// Intel-specific alignment macros
    #if (defined __INTEL_COMPILER) || (defined __ICL)
        #define __attaligned \
                __attribute__((aligned(__MEMORY_ALIGNMENT)))
// Unfortunately, Intel's __assume_aligned macro is not the same as GCCs, so we
// keep it disabled for now.
        #define __assaligned(X) (X)
        #define USING_ALIGNMENT_MACROS
// GCC >= 4.7 and Clang-specific alignment macros:
    #elif (defined __GNUC__ && ((__GNUC__ > 4) || \
          ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 7)))) || \
          (defined __clang__ && (__has_builtin(__builtin_assume_aligned)))
        #define __attaligned \
                __attribute__((__aligned__(__MEMORY_ALIGNMENT)))
        #define __assaligned(X) \
                (__builtin_assume_aligned((X), __MEMORY_ALIGNMENT))
        #define USING_ALIGNMENT_MACROS
// Disable alignment macros for all other compilers:
    #else
        #define __attaligned
        #define __assaligned(X) (X)
    #endif
// If C11 is supported, use _Alignas() macro:
    #if __STDC_VERSION__ >= 201112L
        #define __memaligned _Alignas(__MEMORY_ALIGNMENT)
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

/*
 * Macro to suppress unjustified unused variable compiler warnings when variable
 * initialization has side effects. Used, e.g., in aligned_calloc().
 */
#define __unused(X) ((void)(X))

/*
 * All auto-vectorizable routines process particle coordinates in blocks of 32,
 * which guarantees 64-byte cachline-optimized memory usage and facilitates the
 * desired auto-vectorization of loops, especially when compiled with
 * -march=native on AVX-enabled machines.
 * Note that _BLOCKSIZE * sizeof(float) MUST be an integer multiple of
 * __MEMORY_ALIGNMENT!
 */
#define _BLOCKSIZE (__MEMORY_ALIGNMENT / 2)
#define _2_BLOCKSIZE (2 * _BLOCKSIZE)
#define _3_BLOCKSIZE (3 * _BLOCKSIZE)
#define _BLOCKSIZE_2 (_BLOCKSIZE * _BLOCKSIZE)
#define _3_BLOCKSIZE_2 (3 * _BLOCKSIZE_2)

/*
 * Enable static (i.e., compile-time) assertions, taken in part from
 * http://www.pixelbeat.org/programming/gcc/static_assert.html
 * (GNU All-Permissive License)
 */
#if __STDC_VERSION__ >= 201112L
    #define STATIC_ASSERT(e, m) _Static_assert(e, m)
#else
    #define ASSERT_CONCAT_(a, b) a##b
    #define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
    // These can't be used after statements in C89
    #ifdef __COUNTER__
        #define STATIC_ASSERT(e, m) \
                ;enum { ASSERT_CONCAT(static_assert_, __COUNTER__) = \
                        1 / (int) (!!(e)) }
    #else
        // This can't be used twice on the same line so ensure if using in
        // headers that the headers are not included twice (wrap them in
        // #ifndef...#endif). Note it doesn't cause any issues when used on the
        // same line of separate modules compiled with
        // gcc -combine -fwhole-program.
        #define STATIC_ASSERT(e,m) \
                ;enum { ASSERT_CONCAT(assert_line_, __LINE__) = \
                        1 / (int) (!!(e)) }
    #endif
#endif

// Assert that _BLOCKSIZE * sizeof(float) is divisible by __MEMORY_ALIGNMENT:
STATIC_ASSERT(!(_BLOCKSIZE * sizeof(float) % __MEMORY_ALIGNMENT), \
"BLOCKSIZE * sizeof(float) is not an integer multiple of __MEMORY_ALIGNMENT!");

/*
 * Include required headers:
 *   - assert.h: for runtime assertions in DEBUG mode
 *   - float.h: for FLT_EPSILON
 *   - inttypes.h: to avoid overflows, we use 64-bit integers for histograms
 *   - limits.h: for INT_MAX
 *   - math.h: for floor()
 *   - stdlib.h: for dynamic memory allocations
 *   - string.h: for memset()
 *   - stdio.h: for fprintf()
 */
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 *Include OpenMP header if required:
 */
#ifdef PARALLEL
    #include <omp.h>
    #define USED_OPENMP 1
#else
    #define USED_OPENMP 0
#endif


/**
 * @brief Type definition for xyz coordinates
 */
typedef float coordinate[3];

/**
 * @brief Type definition for histogram bins
 */
typedef int64_t histbin;

/**
 * @brief Type definition for PBC types
 */
typedef enum ePBC {
    PBCortho,
    PBCtriclinic,
    PBCnone,
    PBCunknown
} ePBC;


static void _ortho_pbc(coordinate* coords, int numcoords, float* box)
{
    double s[3];
    float box_inverse[3];
    box_inverse[0] = ((box[0] > FLT_EPSILON) ?  1.0 / box[0] : 0.0);
    box_inverse[1] = ((box[1] > FLT_EPSILON) ?  1.0 / box[1] : 0.0);
    box_inverse[2] = ((box[2] > FLT_EPSILON) ?  1.0 / box[2] : 0.0);
    if ((box_inverse[0] == 0.0) && (box_inverse[1] == 0.0) && \
        (box_inverse[2] == 0.0)) {
        return;
    }
#ifdef PARALLEL
    #pragma omp parallel for private(s) shared(coords)
#endif
    for (int i = 0; i < numcoords; i++){
        s[0] = floor(coords[i][0] * box_inverse[0]);
        s[1] = floor(coords[i][1] * box_inverse[1]);
        s[2] = floor(coords[i][2] * box_inverse[2]);
        coords[i][0] -= s[0] * box[0];
        coords[i][1] -= s[1] * box[1];
        coords[i][2] -= s[2] * box[2];
        if (coords[i][0] < 0.0) {
            coords[i][0] += box[0];
        }
        if (coords[i][1] < 0.0) {
            coords[i][1] += box[1];
        }
        if (coords[i][2] < 0.0) {
            coords[i][2] += box[2];
        }
        if (coords[i][0] >= box[0]) {
            coords[i][0] -= box[0];
        }
        if (coords[i][1] >= box[1]) {
            coords[i][1] -= box[1];
        }
        if (coords[i][2] >= box[2]) {
            coords[i][2] -= box[2];
        }
    }
}


/**
 * @brief Memory-aligned calloc
 *
 * Neither C11 nor POSIX offer a memory-aligned calloc() routine, so here's our
 * own. Its interface is the same as for calloc(), and memory alignment and
 * padding is handled automatically. If C11 or POSIX features are not available,
 * good ol' calloc() is used.
 */
static void* aligned_calloc(size_t num, size_t size)
{
#ifdef USE_ALIGNMENT
    // calculate total number of bytes to allocate and add padding if required:
    size_t aligned_size = num * size;
    if(aligned_size % __MEMORY_ALIGNMENT){
        aligned_size += __MEMORY_ALIGNMENT - aligned_size % __MEMORY_ALIGNMENT;
    }
    assert(aligned_size % __MEMORY_ALIGNMENT == 0);
    // now do the actual allocation
    #ifdef USE_C11_ALIGNMENT
    void* ptr = aligned_alloc(__MEMORY_ALIGNMENT, aligned_size);
    assert(ptr != NULL && "Failed to allocate memory. Out of RAM?");
    #elif defined USE_POSIX_ALIGNMENT
    void* ptr = NULL;
    int retval = posix_memalign(&ptr, __MEMORY_ALIGNMENT, aligned_size);
    assert((retval == 0) && "Failed to allocate memory. Out of RAM?");
    __unused(retval);
    #endif
    // set all allocated bytes to zero:
    return memset(ptr, 0, aligned_size);
#else
    void* ptr = calloc(num, size);
    assert(ptr != NULL && "Failed to allocate memory. Out of RAM?");
    return ptr;
#endif
}


static void _map_charges_on_grid(coordinate* coords, int numcoords,
                                 float* charges, float* qgrid,
                                 int numqgridx, int numqgridy, int numqgridz,
                                 double dx, double dy, double dz)
{
    // Compute linear grid index for each atom and add its charge to the
    // corresponding grid cell:
    size_t numqgrid = numqgridx * numqgridy * numqgridz;
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int i = 0; i < numcoords; i++) {
        size_t ix = (size_t) (coords[i][0] / dx);
        size_t iy = (size_t) (coords[i][1] / dy);
        size_t iz = (size_t) (coords[i][2] / dz);
        size_t lidx = (ix * numqgridy + iy) * numqgridz + iz;
        if ((lidx < 0) || (lidx >= numqgrid)) {
            fprintf(stderr, "\n\nERROR: Position %f %f %f maps on invalid index %ld.\n\n", coords[i][0], coords[i][1], coords[i][2], lidx);
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        qgrid[lidx] += charges[i];
    }
}


static void _charge_histogram(float* qgrid, int numqgridx, int numqgridy,
                               int numqgridz, histbin* histograms, int numhistos,
                               int numhisto, double qmin, double qmax)
{
    double dq = (qmax - qmin) / numhisto;
#ifdef PARALLEL
    #pragma omp parallel for
#endif
    for (int x_origin = 0; x_origin < numqgridx; x_origin++) {
        size_t lin_x_origin = x_origin * numqgridy;
        for (int y_origin = 0; y_origin < numqgridy; y_origin++) {
            size_t lin_xy_origin = (lin_x_origin + y_origin) * numqgridz;
            for (int z_origin = 0; z_origin < numqgridz; z_origin++) {
                size_t lin_xyz_origin = lin_xy_origin + z_origin;
                double net_charge = (double) qgrid[lin_xyz_origin];
                histbin* histogram = histograms;
                int histidx = (int) ((net_charge - qmin) / dq);
                if ((histidx >= 0) && (histidx < numhisto)) {
#ifdef PARALLEL
                    #pragma omp atomic
#endif
                    histogram[histidx] += 1;
                }
                else {
                    fprintf(stderr, "WARNING: Charge %.3lf out of histogram range.\n",
                            net_charge);
                }
                size_t _x, _y, _z;
                for (int cube_len = 2; cube_len <= numhistos; cube_len++) {
                    int x_end = x_origin + cube_len;
                    int y_end = y_origin + cube_len;
                    int z_end = z_origin + cube_len;
                    _x = (x_end - 1) % numqgridx;
                    for (int y = y_origin; y < y_end; y++) {
                        _y = y % numqgridy;
                        for (int z = z_origin; z < z_end; z++) {
                            _z = z % numqgridz;
                            size_t linidx_xyz = (_x * numqgridy + _y) * numqgridz + _z;
                            net_charge += (double) qgrid[linidx_xyz];
                        }
                    }
                    _y = (y_end - 1) % numqgridy;
                    for (int x = x_origin; x < x_end - 1; x++) {
                        _x = x % numqgridx;
                        for (int z = z_origin; z < z_end; z++) {
                            _z = z % numqgridz;
                            size_t linidx_xyz = (_x * numqgridy + _y) * numqgridz + _z;
                            net_charge += (double) qgrid[linidx_xyz];
                        }
                    }
                    _z = (z_end - 1) % numqgridz;
                    for (int x = x_origin; x < x_end - 1; x++) {
                        _x = x % numqgridx;
                        for (int y = y_origin; y < y_end - 1; y++) {
                            _y = y % numqgridy;
                            size_t linidx_xyz = (_x * numqgridy + _y) * numqgridz + _z;
                            net_charge += (double) qgrid[linidx_xyz];
                        }
                    }
                    histogram += numhisto;
                    histidx = (int) ((net_charge - qmin) / dq);
                    if ((histidx >= 0) && (histidx < numhisto)) {
#ifdef PARALLEL
                        #pragma omp atomic
#endif
                        histogram[histidx] += 1;
                    }
                    else {
                        fprintf(stderr, "WARNING: Charge %.3lf out of histogram range.\n",
                                net_charge);
                    }
                }
            }
        }
    }
}


static void _calc_charge_per_volume_histogram(coordinate* coords,
                                              int numcoords,
                                              float* charges,
                                              int numqgridx, int numqgridy,
                                              int numqgridz, float* box,
                                              ePBC pbc_type,
                                              histbin* cube_counts,
                                              double* cube_volumes,
                                              histbin* histograms,
                                              int numhistos,
                                              int numhisto, double qmin,
                                              double qmax)
{
    // Check charge grid size:
    if ((numqgridx <= 0) || (numqgridy <= 0) || (numqgridz <= 0)) {
        fprintf(stderr, "Number of charge grid points have to be greater than" \
                        " zero in all dimensions.");
        return;
    }
    // Check PBC:
    if (pbc_type != PBCortho) {
        fprintf(stderr, "Only non-zero orthogonal boxes are supported.");
        return;
    }
    // Apply PBC:
    _ortho_pbc(coords, numcoords, box);
    // Calculate grid spacings:
    double dx = ((double) box[0]) / numqgridx;
    double dy = ((double) box[1]) / numqgridy;
    double dz = ((double) box[2]) / numqgridz;
    double cell_volume = dx * dy * dz;
    // Allocate charge grid:
    size_t numqgrid = numqgridx * numqgridy * numqgridz;
    float* qgrid __attaligned = (float*) aligned_calloc(numqgrid,
                                                        sizeof(float));
    // Fill in charges:
    _map_charges_on_grid(coords, numcoords, charges, qgrid, numqgridx,
                         numqgridy, numqgridz, dx, dy, dz);
    // Determine maximum cube edge length:
    int max_cube_edge_len = INT_MAX;
    if (numqgridx < max_cube_edge_len) {
        max_cube_edge_len = numqgridx;
    }
    if (numqgridy < max_cube_edge_len) {
        max_cube_edge_len = numqgridy;
    }
    if (numqgridz < max_cube_edge_len) {
        max_cube_edge_len = numqgridz;
    }
    max_cube_edge_len /= 2;
    if (numhistos != max_cube_edge_len) {
        fprintf(stderr, "ERROR: Number of histograms are not equal to number " \
                        "of cube edge lengths.\n");
        return;
    }
    // Compute cube volumes for all edge lengths:
    for (size_t i = 1; i <= max_cube_edge_len; i++) {
        cube_counts[i - 1] = numqgridx * numqgridy * numqgridz;
        cube_volumes[i - 1] = (i * i * i) * cell_volume;
    }
    // Process all possible cubes:
    _charge_histogram(qgrid, numqgridx, numqgridy, numqgridz,
                               histograms, numhistos, numhisto, qmin, qmax);
    free(qgrid);
}
#endif /* __QGRID_H */
