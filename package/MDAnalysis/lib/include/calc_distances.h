/* -*- Mode: C; tab-width: 4; indent-tabs-mode:nil; -*- */
/* vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 */
/*
  MDAnalysis --- http://mdanalysis.googlecode.com

  Copyright (c) 2006-2014 Naveen Michaud-Agrawal,
                Elizabeth J. Denning, Oliver Beckstein,
                and contributors (see AUTHORS for the full list)
  Released under the GNU Public Licence, v2 or any higher version

  Please cite your use of MDAnalysis in published work:

      N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and
      O. Beckstein. MDAnalysis: A Toolkit for the Analysis of
      Molecular Dynamics Simulations. J. Comput. Chem. 32 (2011), 2319--2327,
      in press.
*/

#ifndef __DISTANCES_H
#define __DISTANCES_H
/*#ifdef NDEBUG*/
/*    #undef NDEBUG*/
/*#endif*/
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
 * In grid-based distance calculations, we use a maximum number of grid cells of
 * 4096:
 */
#define __MAX_NUM_GRID_CELLS (4096)

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
 *   - math.h: for square roots
 *   - stdlib.h: for dynamic memory allocations
 *   - string.h: for memset()
 */
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/*#include <stdio.h>*/

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

static void _minimum_image_ortho(double *dx, float *box, float *inverse_box)
{
    // Minimum image convention for orthogonal boxes.
    double s;
    for (int i = 0; i < 3; i++) {
        if (box[i] > FLT_EPSILON) {
            s = inverse_box[i] * dx[i];
            dx[i] = box[i] * (s - round(s));
        }
    }
}

static inline void _minimum_image_ortho_lazy(double* dx, float* box,
                                             float* half_box)
{
    // Minimum image convention for orthogonal boxes.
    // Assumes that the maximum separation is less than 1.5 times the box length
    // (enforced in distance_array functions by packing all particles into the
    // box before calculating separations). For O(n) algorithms it is faster
    // to use _minimum_image_ortho() without prior packing!
    for (int i = 0; i < 3; i++) {
        if (box[i] > FLT_EPSILON) {
            if (dx[i] > half_box[i]) {
                dx[i] -= box[i];
            }
            else if (dx[i] <= -half_box[i]) {
                dx[i] += box[i];
            }
        }
    }
}

static inline void _minimum_image_triclinic_lazy(double *dx, float* box_vectors)
{
    // Minimum image convention for triclinic boxes, modelled after domain.cpp
    // in LAMMPS. Assumes that there is a maximum separation of 1 box length
    // (enforced in dist functions by moving all particles to inside the box
    // before calculating separations).
    // Requires a box (flattened array box_vectors).
    // Assumes box having zero values for box_vectors[1], box_vectors[2] and
    // box_vectors[5]
    double rx_0, ry_0, ry_1, d, min = FLT_MAX;
    double rz[3], dmin[3] = {FLT_MAX};
    for (int x = -1; x < 2; ++x) {
        rx_0 = dx[0] + box_vectors[0] * x;
        for (int y = -1; y < 2; ++y) {
            ry_0 = rx_0  + box_vectors[3] * y;
            ry_1 = dx[1] + box_vectors[4] * y;
            for (int z = -1; z < 2; ++z) {
                rz[0] = ry_0  + box_vectors[6] * z;
                rz[1] = ry_1  + box_vectors[7] * z;
                rz[2] = dx[2] + box_vectors[8] * z;
                d = rz[0] * rz[0] + rz[1] * rz[1] + rz[2] * rz[2];
                if (d < min) {
                    min = d;
                    for (int i = 0; i < 3; ++i){
                        dmin[i] = rz[i];
                    }
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        dx[i] = dmin[i];
    }
}

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
    }
}

static void _triclinic_pbc(coordinate* coords, int numcoords,
                           float* box_vectors)
{
    // Moves all coordinates to within the box boundaries for a triclinic box
    // Assumes box_vectors having zero values for box_vectors[1], box_vectors[2]
    // and box_vectors[5]

    // Inverse bi of matrix box b (row-major indexing):
    //   [ 1/b0                      ,  0         , 0   ]
    //   [-b3/(b0*b4)                ,  1/b4      , 0   ]
    //   [ (b3*b7/(b0*b4) - b6/b0)/b8, -b7/(b4*b8), 1/b8]
    float bi0 = ((box_vectors[0] > FLT_EPSILON) ? 1.0 / box_vectors[0] : 0.0);
    float bi4 = ((box_vectors[4] > FLT_EPSILON) ? 1.0 / box_vectors[4] : 0.0);
    float bi8 = ((box_vectors[8] > FLT_EPSILON) ? 1.0 / box_vectors[8] : 0.0);
    if ((bi0 == 0.0) && (bi4 == 0.0) && (bi8 == 0.0)) {
        return;
    }
    float bi3 = -box_vectors[3] * bi0 * bi4;
    float bi6 = (box_vectors[3] * box_vectors[7] * bi4 - box_vectors[6]) * \
                bi0 * bi8;
    float bi7 = -box_vectors[7] * bi4 * bi8;
#ifdef PARALLEL
    #pragma omp parallel for shared(coords)
#endif
    for (int i = 0; i < numcoords; i++){
        double s;
        // translate coords[i] to central cell along c-axis
        s = floor(coords[i][2] * bi8);
        coords[i][0] -= s * box_vectors[6];
        coords[i][1] -= s * box_vectors[7];
        coords[i][2] -= s * box_vectors[8];
        // translate remainder of coords[i] to central cell along b-axis
        s = floor(coords[i][1] * bi4 + coords[i][2] * bi7);
        coords[i][0] -= s * box_vectors[3];
        coords[i][1] -= s * box_vectors[4];
        // translate remainder of coords[i] to central cell along a-axis
        s = floor(coords[i][0] * bi0 + coords[i][1] * bi3 + coords[i][2] * bi6);
        coords[i][0] -= s * box_vectors[0];
    }
}

static void _calc_distance_array(coordinate* ref, int numref, coordinate* conf,
                                 int numconf, float* box, ePBC pbc_type,
                                 double* distances)
{
    double dx[3];
    float half_box[3] = {0.0};

    switch(pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc(ref, numref, box);
            _ortho_pbc(conf, numconf, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc(ref, numref, box);
            _triclinic_pbc(conf, numconf, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel for private(dx) shared(distances)
#endif
    for (int i = 0; i < numref; i++) {
        for (int j = 0; j < numconf; j++) {
            dx[0] = conf[j][0] - ref[i][0];
            dx[1] = conf[j][1] - ref[i][1];
            dx[2] = conf[j][2] - ref[i][2];
            switch(pbc_type) {
                case PBCortho:
                    _minimum_image_ortho_lazy(dx, box, half_box);
                    break;
                case PBCtriclinic:
                    _minimum_image_triclinic_lazy(dx, box);
                    break;
                default:
                    break;
            };
            double rsq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            *(distances + i * numconf + j) = sqrt(rsq);
        }
    }
}

static void _calc_self_distance_array(coordinate* ref, int numref, float* box,
                                      ePBC pbc_type, double* distances)
{
    int distpos = 0;
    double dx[3];
    float half_box[3] = {0.0};

    switch(pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc(ref, numref, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc(ref, numref, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel for private(distpos, dx) shared(distances)
#endif
    for (int i = 0; i < numref; i++) {
#ifdef PARALLEL
        // calculate the offset into distances:
        distpos = i * (2 * numref - i - 1) / 2;
#endif
        for (int j = i + 1; j < numref; j++) {
            dx[0] = ref[j][0] - ref[i][0];
            dx[1] = ref[j][1] - ref[i][1];
            dx[2] = ref[j][2] - ref[i][2];
            switch(pbc_type) {
                case PBCortho:
                    _minimum_image_ortho_lazy(dx, box, half_box);
                    break;
                case PBCtriclinic:
                    _minimum_image_triclinic_lazy(dx, box);
                    break;
                default:
                    break;
            };
            double rsq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            *(distances + distpos) = sqrt(rsq);
            distpos++;
        }
    }
}

static void _calc_distance_histogram(coordinate* ref, int numref,
                                     coordinate* conf, int numconf, float* box,
                                     ePBC pbc_type, double rmin, double rmax,
                                     histbin* histo, int numhisto)
{
    assert((rmin >= 0.0) && \
           "Minimum distance must be greater than or equal to zero");
    assert(((rmax - rmin) > FLT_EPSILON) && \
           "Maximum distance must be greater than minimum distance.");
    float half_box[3] = {0.0};
    double inverse_binw = numhisto / (rmax - rmin);
    double r2_min = rmin * rmin;
    double r2_max = rmax * rmax;

    switch(pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc(ref, numref, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc(ref, numref, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        double dx[3];
        histbin* local_histo = (histbin*) calloc(numhisto + 1, sizeof(histbin));
        assert(local_histo != NULL);
#ifdef PARALLEL
        #pragma omp for nowait
#endif
        for (int i = 0; i < numref; i++) {
            for (int j = 0; j < numconf; j++) {
                dx[0] = conf[j][0] - ref[i][0];
                dx[1] = conf[j][1] - ref[i][1];
                dx[2] = conf[j][2] - ref[i][2];
                switch(pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy(dx, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy(dx, box);
                        break;
                    default:
                        break;
                };
                double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if ((r2 >= r2_min) && (r2 <= r2_max)) {
                    int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                    if (k >= 0) {
                        local_histo[k]++;
                    }
                }
            }
        }
        // gather local results
        for (int i = 0; i < numhisto; i++) {
#ifdef PARALLEL
            #pragma omp atomic
#endif
            histo[i] += local_histo[i];
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        histo[numhisto - 1] += local_histo[numhisto]; // Numpy consistency
        free(local_histo);
    }
}

static void _calc_self_distance_histogram(coordinate* ref, int numref,
                                          float* box, ePBC pbc_type,
                                          double rmin, double rmax,
                                          histbin* histo, int numhisto)
{
    assert((rmin >= 0.0) && \
           "Minimum distance must be greater than or equal to zero");
    assert(((rmax - rmin) > FLT_EPSILON) && \
           "Maximum distance must be greater than minimum distance.");
    float half_box[3] = {0.0};
    double inverse_binw = numhisto / (rmax - rmin);
    double r2_min = rmin * rmin;
    double r2_max = rmax * rmax;

    switch(pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc(ref, numref, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc(ref, numref, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };
#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        double dx[3];
        histbin* local_histo = (histbin*) calloc(numhisto + 1, sizeof(histbin));
        assert(local_histo != NULL);
#ifdef PARALLEL
        #pragma omp for nowait
#endif
        for (int i = 0; i < numref - 1; i++) {
            for (int j = i + 1; j < numref; j++) {
                dx[0] = ref[j][0] - ref[i][0];
                dx[1] = ref[j][1] - ref[i][1];
                dx[2] = ref[j][2] - ref[i][2];
                switch(pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy(dx, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy(dx, box);
                        break;
                    default:
                        break;
                };
                double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if ((r2 >= r2_min) && (r2 <= r2_max)) {
                    int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                    if (k >= 0) {
                        local_histo[k]++;
                    }
                }
            }
        }
        // gather local results from threads
        for (int i = 0; i < numhisto; i++) {
#ifdef PARALLEL
            #pragma omp atomic
#endif
            histo[i] += local_histo[i];
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        histo[numhisto - 1] += local_histo[numhisto]; // Numpy consistency
        free(local_histo);
    }
}

static void _coord_transform(coordinate* coords, int numCoords, coordinate* box)
{
    float new[3];
    // Matrix multiplication inCoords * box = outCoords
    // Multiplication done in place using temp array 'new'
    // Used to transform coordinates to/from S/R space in trilinic boxes
#ifdef PARALLEL
    #pragma omp parallel for private(new) shared(coords)
#endif
    for (int i = 0; i < numCoords; i++){
        new[0] = 0.0;
        new[1] = 0.0;
        new[2] = 0.0;
        for (int j = 0; j < 3; j++){
            for (int k = 0; k < 3; k++){
                new[j] += coords[i][k] * box[k][j];
            }
        }
        coords[i][0] = new[0];
        coords[i][1] = new[1];
        coords[i][2] = new[2];
    }
}

static void _calc_bond_distance(coordinate* atom1, coordinate* atom2,
                                int numatom, float* box, ePBC pbc_type,
                                double* distances)
{
    double dx[3];
    float inverse_box[3];

    switch(pbc_type) {
        case PBCortho:
            inverse_box[0] = 1.0 / box[0];
            inverse_box[1] = 1.0 / box[1];
            inverse_box[2] = 1.0 / box[2];
            break;
        case PBCtriclinic:
            _triclinic_pbc(atom1, numatom, box);
            _triclinic_pbc(atom2, numatom, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel for private(dx) shared(distances)
#endif
    for (int i = 0; i < numatom; i++) {
        dx[0] = atom1[i][0] - atom2[i][0];
        dx[1] = atom1[i][1] - atom2[i][1];
        dx[2] = atom1[i][2] - atom2[i][2];
        switch(pbc_type) {
            case PBCortho:
                _minimum_image_ortho(dx, box, inverse_box);
                break;
            case PBCtriclinic:
                _minimum_image_triclinic_lazy(dx, box);
                break;
            default:
                break;
        };
        double rsq = (dx[0] * dx[0]) + (dx[1] * dx[1]) + (dx[2] * dx[2]);
        *(distances + i) = sqrt(rsq);
    }
}

static void _calc_angle(coordinate* atom1, coordinate* atom2,
                        coordinate* atom3, int numatom, float* box,
                        ePBC pbc_type, double* angles)
{
    double rji[3], rjk[3], xp[3];
    float inverse_box[3];

    switch(pbc_type) {
        case PBCortho:
            inverse_box[0] = 1.0 / box[0];
            inverse_box[1] = 1.0 / box[1];
            inverse_box[2] = 1.0 / box[2];
            break;
        case PBCtriclinic:
            _triclinic_pbc(atom1, numatom, box);
            _triclinic_pbc(atom2, numatom, box);
            _triclinic_pbc(atom3, numatom, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel for private(rji, rjk, xp) shared(angles)
#endif
    for (int i = 0; i < numatom; i++) {
        rji[0] = atom1[i][0] - atom2[i][0];
        rji[1] = atom1[i][1] - atom2[i][1];
        rji[2] = atom1[i][2] - atom2[i][2];

        rjk[0] = atom3[i][0] - atom2[i][0];
        rjk[1] = atom3[i][1] - atom2[i][1];
        rjk[2] = atom3[i][2] - atom2[i][2];

        switch(pbc_type) {
            case PBCortho:
                _minimum_image_ortho(rji, box, inverse_box);
                _minimum_image_ortho(rjk, box, inverse_box);
                break;
            case PBCtriclinic:
                _minimum_image_triclinic_lazy(rji, box);
                _minimum_image_triclinic_lazy(rjk, box);
                break;
            default:
                break;
        };

        double x = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];

        xp[0] =  rji[1] * rjk[2] - rji[2] * rjk[1];
        xp[1] = -rji[0] * rjk[2] + rji[2] * rjk[0];
        xp[2] =  rji[0] * rjk[1] - rji[1] * rjk[0];

        double y = sqrt(xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2]);

        *(angles + i) = atan2(y, x);
    }
}

static inline void _calc_dihedral_angle(double* va, double* vb, double* vc,
                                        double* result)
{
    // Returns atan2 from vectors va, vb, vc
    double n1[3], n2[3];
    double xp[3], vb_norm;
    double x, y;

    // n1 is normal vector to -va, vb
    // n2 is normal vector to -vb, vc
    n1[0] = -va[1] * vb[2] + va[2] * vb[1];
    n1[1] =  va[0] * vb[2] - va[2] * vb[0];
    n1[2] = -va[0] * vb[1] + va[1] * vb[0];

    n2[0] = -vb[1] * vc[2] + vb[2] * vc[1];
    n2[1] =  vb[0] * vc[2] - vb[2] * vc[0];
    n2[2] = -vb[0] * vc[1] + vb[1] * vc[0];

    // x = dot(n1,n2) = cos theta
    x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];

    // xp = cross(n1,n2)
    xp[0] =  n1[1] * n2[2] - n1[2] * n2[1];
    xp[1] = -n1[0] * n2[2] + n1[2] * n2[0];
    xp[2] =  n1[0] * n2[1] - n1[1] * n2[0];

    vb_norm = sqrt(vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]);

    y = (xp[0] * vb[0] + xp[1] * vb[1] + xp[2] * vb[2]) / vb_norm;

    if ((fabs(x) == 0.0) && (fabs(y) == 0.0)) // numpy consistency
    {
        *result = NAN;
        return;
    }
    *result = atan2(y, x); //atan2 is better conditioned than acos
}

static void _calc_dihedral(coordinate* atom1, coordinate* atom2,
                           coordinate* atom3, coordinate* atom4, int numatom,
                           float* box, ePBC pbc_type, double* angles)
{
    double va[3], vb[3], vc[3];
    float inverse_box[3];

    switch(pbc_type) {
        case PBCortho:
            inverse_box[0] = 1.0 / box[0];
            inverse_box[1] = 1.0 / box[1];
            inverse_box[2] = 1.0 / box[2];
            break;
        case PBCtriclinic:
            _triclinic_pbc(atom1, numatom, box);
            _triclinic_pbc(atom2, numatom, box);
            _triclinic_pbc(atom3, numatom, box);
            _triclinic_pbc(atom4, numatom, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel for private(va, vb, vc) shared(angles)
#endif
    for (int i = 0; i < numatom; i++) {
        // connecting vectors between all 4 atoms:
        // 1 -va-> 2 -vb-> 3 -vc-> 4
        va[0] = atom2[i][0] - atom1[i][0];
        va[1] = atom2[i][1] - atom1[i][1];
        va[2] = atom2[i][2] - atom1[i][2];

        vb[0] = atom3[i][0] - atom2[i][0];
        vb[1] = atom3[i][1] - atom2[i][1];
        vb[2] = atom3[i][2] - atom2[i][2];

        vc[0] = atom4[i][0] - atom3[i][0];
        vc[1] = atom4[i][1] - atom3[i][1];
        vc[2] = atom4[i][2] - atom3[i][2];

        switch(pbc_type) {
            case PBCortho:
                _minimum_image_ortho(va, box, inverse_box);
                _minimum_image_ortho(vb, box, inverse_box);
                _minimum_image_ortho(vc, box, inverse_box);
                break;
            case PBCtriclinic:
                _minimum_image_triclinic_lazy(va, box);
                _minimum_image_triclinic_lazy(vb, box);
                _minimum_image_triclinic_lazy(vc, box);
                break;
            default:
                break;
        };

        _calc_dihedral_angle(va, vb, vc, angles + i);
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

/**
 * @brief Arrange coordinates in blocks of _BLOCKSIZE positions
 *
 * This function takes an array @p coords containing @p numcoords xyz positions
 * and arranges them into a newly allocated(!) array returned as a @<float*@>
 * pointer. In this array, coordinates are block-aligned in blocks of
 * @c _BLOCKSIZE positions where each block contains first all x-, then y-, and
 * finally all z-coordinates. This layout improves memory locality and
 * guarantees 64-byte cachline-optimization. This helps to avoid "false sharing"
 * between OpenMP threads.
 */
static inline float* _get_coords_in_blocks(const coordinate* restrict coords,
                                           int numcoords)
{
    int nblocks = numcoords / _BLOCKSIZE;
    int nremaining = numcoords % _BLOCKSIZE;
    int nblocks_to_calloc = nblocks + (nremaining > 0);
    float* bcoords __attaligned = (float*) aligned_calloc( \
                                  nblocks_to_calloc * _3_BLOCKSIZE, \
                                  sizeof(float));
    // process full blocks
#ifdef PARALLEL
    #pragma omp parallel for shared(bcoords)
#endif
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < 3; j++) {
            float* _coords = ((float*) (coords + i * _BLOCKSIZE)) + j;
            float* _bcoords = (float*) __assaligned(bcoords + \
                                                     i * _3_BLOCKSIZE + \
                                                     j * _BLOCKSIZE);
            for (int k = 0; k < _BLOCKSIZE; k++) {
                _bcoords[k] = _coords[3 * k];
            }
        }
    }
    // process remaining partial block
    if (nremaining > 0) {
        for (int j = 0; j < 3; j++) {
            float* _coords = ((float*) (coords + nblocks * _BLOCKSIZE)) + j;
            float* _bcoords = (float*) __assaligned(bcoords + \
                                                    nblocks * _3_BLOCKSIZE + \
                                                    j * _BLOCKSIZE);
            for (int k = 0; k < nremaining; k++) {
                _bcoords[k] = _coords[3 * k];
            }
        }
    }
    return bcoords;
}

/**
 * @brief Moves block-aligned coordinates into the central periodic image
 *
 * This function takes an array @p coords of @p numcoords block-aligned
 * coordinates as well as an array @p box containing the box edge lengths of a
 * rectangular simulation box. Folds coordinates which lie outside the box back
 * into the box.
 */
static void _ortho_pbc_vectorized(float* restrict coords, int numcoords,
                                  const float* box)
{
    const int nblocks = numcoords / _BLOCKSIZE + ((numcoords % _BLOCKSIZE) > 0);
    float box_inverse[3];
    box_inverse[0] = ((box[0] > FLT_EPSILON) ?  1.0 / box[0] : 0.0);
    box_inverse[1] = ((box[1] > FLT_EPSILON) ?  1.0 / box[1] : 0.0);
    box_inverse[2] = ((box[2] > FLT_EPSILON) ?  1.0 / box[2] : 0.0);
    if ((box_inverse[0] == 0.0) && (box_inverse[1] == 0.0) && \
        (box_inverse[2] == 0.0)) {
        return;
    }
#ifdef PARALLEL
    #pragma omp parallel shared(coords)
#endif
    {
        float* s __attaligned = (float*) aligned_calloc(_BLOCKSIZE,
                                                        sizeof(float));
        __memaligned float bx;
        __memaligned float ibx;
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks; n++) {
            for (int i = 0; i < 3; i++) {
                if (box[i] > FLT_EPSILON) {
                    float* _coords __attaligned = \
                    (float*) __assaligned(coords + n * _3_BLOCKSIZE + \
                                          i * _BLOCKSIZE);
                    bx = box[i];
                    ibx = box_inverse[i];
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        s[j] = _coords[j] * ibx;
                    }
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        s[j] = floorf(s[j]);
                    }
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        _coords[j] -= bx * s[j];
                    }
                }
            }
        }
        free(s);
    }
}

/**
 * @brief Moves block-aligned coordinates into the central periodic image
 *
 * This function takes an array @p coords of @p numcoords block-aligned
 * coordinates as well as an array @p box_vectors containing the box vectors of
 * a triclinic simulation box. Folds coordinates which lie outside the box back
 * into the box.
 */
static void _triclinic_pbc_vectorized(float* restrict coords, int numcoords,
                                      const float* box_vectors)
{
    // Moves all coordinates to within the box boundaries for a triclinic box
    // Assumes box_vectors having zero values for box_vectors[1], box_vectors[2]
    // and box_vectors[5]
    const int nblocks = numcoords / _BLOCKSIZE + \
                        ((numcoords % _BLOCKSIZE) > 0);
    // Inverse bi of matrix box b (row-major indexing):
    //   [ 1/b0                      ,  0         , 0   ]
    //   [-b3/(b0*b4)                ,  1/b4      , 0   ]
    //   [ (b3*b7/(b0*b4) - b6/b0)/b8, -b7/(b4*b8), 1/b8]
    __memaligned float bi0 = ((box_vectors[0] > FLT_EPSILON) ? \
                             1.0 / box_vectors[0] : 0.0);
    __memaligned float bi4 = ((box_vectors[4] > FLT_EPSILON) ? \
                             1.0 / box_vectors[4] : 0.0);
    __memaligned float bi8 = ((box_vectors[8] > FLT_EPSILON) ? \
                             1.0 / box_vectors[8] : 0.0);
    if ((bi0 == 0.0) && (bi4 == 0.0) && (bi8 == 0.0)) {
        return;
    }
    __memaligned float bi3 = -box_vectors[3] * bi0 * bi4;
    __memaligned float bi7 = -box_vectors[7] * bi4 * bi8;
    __memaligned float bi6 = (box_vectors[3] * box_vectors[7] * bi4 - \
                              box_vectors[6]) * bi0 * bi8;
#ifdef PARALLEL
    #pragma omp parallel shared(coords)
#endif
    {
        float* s __attaligned = (float*) aligned_calloc(_BLOCKSIZE,
                                                        sizeof(float));
        __memaligned float bxv;
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks; n++) {
            float* x_coords __attaligned = \
            (float*) __assaligned(coords + n * _3_BLOCKSIZE);
            float* y_coords __attaligned = \
            (float*) __assaligned(coords + n * _3_BLOCKSIZE + _BLOCKSIZE);
            float* z_coords __attaligned = \
            (float*) __assaligned(coords + n * _3_BLOCKSIZE + _2_BLOCKSIZE);
            // translate x-, y-, and z-coordinates to central cell along c-axis
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = z_coords[i] * bi8;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[6];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[7];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                y_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[8];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                z_coords[i] -= s[i] * bxv;
            }
            // translate x- and y-coordinates to central cell along b-axis
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = y_coords[i] * bi4;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] += z_coords[i] * bi7;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[3];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[4];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                y_coords[i] -= s[i] * bxv;
            }
            // translate x-coordinates to central cell along a-axis
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = x_coords[i] * bi0;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] += y_coords[i] * bi3;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] += z_coords[i] * bi6;
            }
            for (int i = 0; i < _BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[0];
            for (int i = 0; i < _BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
        }
        free(s);
    }
}

/**
 * @brief Computes all distances within a block of @c _BLOCKSIZE positions
 *
 * This function takes an array @p refs of @c _BLOCKSIZE block-aligned
 * positions and computes all @<_BLOCKSIZE * _BLOCKSIZE@> pairwise distance
 * vectors, which are stored in the provided @p dxs array.
 * When SIMD-vectorized by the compiler, this routine should be faster than
 * computing only the unique @<_BLOCKSIZE * (_BLOCKSIZE - 1) / 2@> distances.
 */
static inline void _calc_self_distance_vectors_block(double* restrict dxs,
                                                     const float* restrict refs)
{
    dxs = (double*) __assaligned(dxs);
    refs = (float*) __assaligned(refs);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < _BLOCKSIZE; j++) {
            double* _dxs = (double*) __assaligned(dxs + i * _BLOCKSIZE_2 + \
                                                  j * _BLOCKSIZE);
            float* _refs = (float*) __assaligned(refs + i * _BLOCKSIZE);
            __memaligned float _ref = refs[i * _BLOCKSIZE + j];
            for (int k = 0; k < _BLOCKSIZE; k++) {
                _dxs[k] = _refs[k] - _ref;
            }
        }
    }
}

/**
 * @brief Computes all distances between two blocks of @c _BLOCKSIZE positions
 *
 * This function takes two arrays @p refs and @p confs, each containing
 * @c _BLOCKSIZE block-aligned positions. It computes all
 * @<_BLOCKSIZE * _BLOCKSIZE@> pairwise distance vectors between the two
 * arrays, which are stored in the provided @p dxs array.
 */
static inline void _calc_distance_vectors_block(double* restrict dxs,
                                                const float* restrict refs,
                                                const float* restrict confs)
{
    dxs = (double*) __assaligned(dxs);
    refs = (float*) __assaligned(refs);
    confs = (float*) __assaligned(confs);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < _BLOCKSIZE; j++) {
            double* _dxs = (double*) __assaligned(dxs + i * _BLOCKSIZE_2 + \
                                                  j * _BLOCKSIZE);
            float* _confs = (float*) __assaligned(confs + i * _BLOCKSIZE);
            __memaligned float _ref = refs[i * _BLOCKSIZE + j];
            for (int k = 0; k < _BLOCKSIZE; k++) {
                _dxs[k] = _confs[k] - _ref;
            }
        }
    }
}

/**
 * @brief Compute minimum image representations of distance vectors
 *
 * This function takes an array @p dxs containing @<_BLOCKSIZE * _BLOCKSIZE@>
 * distance vectors, an array @p box containing the box edge lengths of a
 * rectangular simulation box, and an array @p half_box containing the respective
 * half box edge lengths. It applies the minimum image convention on the
 * distance vectors with respect to the box.
 */
static inline void _minimum_image_ortho_lazy_block(double* restrict dxs,
                                                   const float* box,
                                                   const float* half_box)
{
    dxs = (double*) __assaligned(dxs);
    __memaligned double bx;
    __memaligned double hbx;
    __memaligned double nhbx;
    for (int i = 0; i < 3; i++) {
        if (box[i] > FLT_EPSILON) {
            bx = box[i];
            hbx = half_box[i];
            nhbx = -half_box[i];
            double* _dxs = (double*) __assaligned(dxs + i * _BLOCKSIZE_2);
            for (int j = 0; j < _BLOCKSIZE_2; j++) {
                _dxs[j] -= ((_dxs[j] > hbx) ? bx : 0.0);
            }
            for (int j = 0; j < _BLOCKSIZE_2; j++) {
                _dxs[j] += ((_dxs[j] <= nhbx) ? bx : 0.0);
            }
        }
    }
}

/**
 * @brief Compute minimum image representations of distance vectors
 *
 * This function takes an array @p dxs containing @<_BLOCKSIZE * _BLOCKSIZE@>
 * distance vectors, an array @p box_vectors containing the box vectors of
 * a triclinic simulation box. It applies the minimum image convention on the
 * distance vectors with respect to the box.
 * The parameter @p aux serves as a container to store intermediate values and
 * must provide enough space to store 11 * _BLOCKSIZE ^ 2 doubles (for
 * _BLOCKSIZE = 32 and sizeof(double) = 8 bytes that's exactly 88 kiB).
 * This avoids repeated memory allocations if the function is called in a loop.
 */
static inline void _minimum_image_triclinic_lazy_block(double* restrict dxs,
                                                       const float* box_vectors,
                                                       double* restrict aux)
{
    // pointers to x-, y-, and z-coordinates of distances:
    double* dx_0 __attaligned = (double*) __assaligned(dxs + 0 * _BLOCKSIZE_2);
    double* dx_1 __attaligned = (double*) __assaligned(dxs + 1 * _BLOCKSIZE_2);
    double* dx_2 __attaligned = (double*) __assaligned(dxs + 2 * _BLOCKSIZE_2);
    // pointers for auxiliary arrays:
    double* rx_0 __attaligned = (double*) __assaligned(aux + 0 * _BLOCKSIZE_2);
    double* ry_0 __attaligned = (double*) __assaligned(aux + 1 * _BLOCKSIZE_2);
    double* ry_1 __attaligned = (double*) __assaligned(aux + 2 * _BLOCKSIZE_2);
    double* rz_0 __attaligned = (double*) __assaligned(aux + 3 * _BLOCKSIZE_2);
    double* rz_1 __attaligned = (double*) __assaligned(aux + 4 * _BLOCKSIZE_2);
    double* rz_2 __attaligned = (double*) __assaligned(aux + 5 * _BLOCKSIZE_2);
    double* d    __attaligned = (double*) __assaligned(aux + 6 * _BLOCKSIZE_2);
    double* min  __attaligned = (double*) __assaligned(aux + 7 * _BLOCKSIZE_2);
    double* dmin_0 __attaligned = (double*) __assaligned(aux + \
                                                             8 * _BLOCKSIZE_2);
    double* dmin_1 __attaligned = (double*) __assaligned(aux + \
                                                             9 * _BLOCKSIZE_2);
    double* dmin_2 __attaligned = (double*) __assaligned(aux + \
                                                            10 * _BLOCKSIZE_2);
    // auxiliary variables:
    __memaligned double xb0;
    __memaligned double yb3;
    __memaligned double yb4;
    __memaligned double zb6;
    __memaligned double zb7;
    __memaligned double zb8;
    // initialize min, dmin_0, dmin_1, and dmin_2 in a single loop:
    __memaligned double flt_max = FLT_MAX;
    for (int i = 0; i < 4 * _BLOCKSIZE_2; i++) {
        min[i] = flt_max;
    }
    // now do the actual minimum image computation:
    for (int x = -1; x < 2; x++) {
        xb0 = x * box_vectors[0];
        for (int i = 0; i < _BLOCKSIZE_2; i++) {
            rx_0[i] = dx_0[i] + xb0;
        }
        for (int y = -1; y < 2; y++) {
            yb3 = y * box_vectors[3];
            yb4 = y * box_vectors[4];
            for (int i = 0; i < _BLOCKSIZE_2; i++) {
                ry_0[i] = rx_0[i] + yb3;
            }
            for (int i = 0; i < _BLOCKSIZE_2; i++) {
                ry_1[i] = dx_1[i] + yb4;
            }
            for (int z = -1; z < 2; z++) {
                zb6 = z * box_vectors[6];
                zb7 = z * box_vectors[7];
                zb8 = z * box_vectors[8];
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    rz_0[i] = ry_0[i] + zb6;
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    rz_1[i] = ry_1[i] + zb7;
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    rz_2[i] = dx_2[i] + zb8;
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    d[i] = rz_0[i] * rz_0[i];
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    d[i] += rz_1[i] * rz_1[i];
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    d[i] += rz_2[i] * rz_2[i];
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    dmin_0[i] = ((d[i] < min[i]) ? rz_0[i] : dmin_0[i]);
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    dmin_1[i] = ((d[i] < min[i]) ? rz_1[i] : dmin_1[i]);
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    dmin_2[i] = ((d[i] < min[i]) ? rz_2[i] : dmin_2[i]);
                }
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    min[i] = ((d[i] < min[i]) ? d[i] : min[i]);
                }
            }
        }
    }
    dxs = (double*) __assaligned(dxs);
    for (int i = 0; i < _3_BLOCKSIZE_2; i++) {
        dxs[i] = dmin_0[i];
    }
}

/**
 * @brief Compute squared distances from distance vectors
 *
 * This function takes an array @p dxs containing @<_BLOCKSIZE * _BLOCKSIZE@>
 * distance vectors and computes their squared Euclidean norms, which are
 * stored in the provided @p r2s array.
 */
static inline void _calc_squared_distances_block(double* restrict r2s,
                                                 const double* restrict dxs)
{
    r2s = (double*) __assaligned(r2s);
    r2s = (double*) memset((void*) r2s, 0, _BLOCKSIZE_2 * sizeof(double));
    for (int i = 0; i < 3; i++) {
        double* _dxs __attaligned = (double*) __assaligned(dxs + \
                                                           i * _BLOCKSIZE_2);
        double* _r2s __attaligned = (double*) __assaligned(r2s);
        for (int j = 0; j < _BLOCKSIZE_2; j++) {
            _r2s[j] += _dxs[j] * _dxs[j];
        }
    }
}

static void _calc_distance_array_vectorized(const coordinate* restrict ref,
                                            int numref,
                                            const coordinate* restrict conf,
                                            int numconf,
                                            const float* box, ePBC pbc_type,
                                            double* restrict distances)
{
    const int nblocks_ref = numref / _BLOCKSIZE;
    const int nblocks_conf = numconf / _BLOCKSIZE;
    const int partial_block_size_ref = numref % _BLOCKSIZE;
    const int partial_block_size_conf = numconf % _BLOCKSIZE;
    float* bref = _get_coords_in_blocks(ref, numref);
    float* bconf = _get_coords_in_blocks(conf, numconf);
    float half_box[3] = {0.0};

    switch (pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc_vectorized(bref, numref, box);
            _ortho_pbc_vectorized(bconf, numconf, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc_vectorized(bref, numref, box);
            _triclinic_pbc_vectorized(bconf, numconf, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(distances)
#endif
    {
        double* dxs = (double*) aligned_calloc(_3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(_BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * _BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks_ref; n++) {
            // process blocks of the n-th row
            // (_BLOCKSIZE x _BLOCKSIZE squares):
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bconf + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                double* _dists = distances + _BLOCKSIZE * (n * numconf + m);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        _dists[i * numconf + j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (_BLOCKSIZE x partial_block_size_conf rectangles):
            if (partial_block_size_conf > 0){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bconf + nblocks_conf * \
                                             _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                         _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                double* _dists = distances + \
                                 _BLOCKSIZE * (n * numconf + nblocks_conf);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        _dists[i * numconf + j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
        }
        // process remaining partial blocks after the last row
        // (partial_block_size_ref x _BLOCKSIZE rectangles):
        if (partial_block_size_ref > 0){
#ifdef PARALLEL
            #pragma omp for schedule(static, 1) nowait
#endif
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs,
                                             bref + nblocks_ref * _3_BLOCKSIZE,
                                             bconf + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                         _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                double* _dists = distances + \
                                 _BLOCKSIZE * (nblocks_ref * numconf + m);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        _dists[i * numconf + j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
        }
#ifdef PARALLEL
        #pragma omp single nowait
#endif
        {
            // process remaining bottom-right partial block
            // (partial_block_size_ref x partial_block_size_conf rectangle):
            if (partial_block_size_ref * partial_block_size_conf > 0) {
                _calc_distance_vectors_block(dxs, bref + \
                                             nblocks_ref * _3_BLOCKSIZE,
                                             bconf + \
                                             nblocks_conf * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                            _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                double* _dists = distances + _BLOCKSIZE * \
                                 (nblocks_ref * numconf + nblocks_conf);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        _dists[i * numconf + j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
    }
    free(bref);
    free(bconf);
}

static void _calc_self_distance_array_vectorized(const coordinate* restrict ref,
                                                 int numref,
                                                 const float* box,
                                                 ePBC pbc_type,
                                                 double* restrict distances)
{
    const int nblocks = numref / _BLOCKSIZE;
    const int partial_block_size = numref % _BLOCKSIZE;
    float* bref = _get_coords_in_blocks(ref, numref);
    float half_box[3] = {0.0};

    switch (pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc_vectorized(bref, numref, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc_vectorized(bref, numref, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };
#ifdef PARALLEL
    #pragma omp parallel
#endif
    {
        double* dxs = (double*) aligned_calloc(_3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(_BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * _BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (int n = 0; n < nblocks; n++) {
            // process first block of the n-th row
            // ((_BLOCKSIZE - 1) x (_BLOCKSIZE - 1) triangle):
            _calc_self_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE);
            switch (pbc_type) {
                case PBCortho:
                    _minimum_image_ortho_lazy_block(dxs, box, half_box);
                    break;
                case PBCtriclinic:
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    break;
                default:
                    break;
            };
            _calc_squared_distances_block(r2s, dxs);
            double* _distances = distances + n * (numref * _BLOCKSIZE - \
                                 (n * _BLOCKSIZE_2 + _BLOCKSIZE) / 2);
            for (int i = 0; i < _BLOCKSIZE - 1; i++) {
                double* __distances = _distances + \
                                      i * (numref - n * _BLOCKSIZE) - \
                                      (i + 1) * (i + 2) / 2;
                for (int j = i + 1; j < _BLOCKSIZE; j++) {
                    __distances[j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                }
            }
            // process the remaining blocks of the n-th row
            // (_BLOCKSIZE x _BLOCKSIZE squares):
            for (int m = n + 1; m < nblocks; m++){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bref + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                _distances += _BLOCKSIZE;
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    double* __distances = _distances + \
                                          i * (numref - n * _BLOCKSIZE) - \
                                          (i + 1) * (i + 2) / 2;
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        __distances[j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (_BLOCKSIZE x partial_block_size rectangle):
            if (partial_block_size > 0){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bref + nblocks * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                _distances += _BLOCKSIZE;
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    double* __distances = _distances + \
                                          i * (numref - n * _BLOCKSIZE) - \
                                          (i + 1) * (i + 2) / 2;
                    for (int j = 0; j < partial_block_size; j++) {
                        __distances[j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
        }
#ifdef PARALLEL
        #pragma omp single nowait
#endif
        {
            // process remaining bottom-right partial block:
            // ((partial_block_size - 1) x (partial_block_size - 1) triangle):
            if (partial_block_size > 0) {
                _calc_self_distance_vectors_block(dxs, bref + \
                                                  nblocks * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                            _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                double * _distances = distances + \
                                      nblocks * (numref * _BLOCKSIZE - \
                                      (nblocks * _BLOCKSIZE_2 + _BLOCKSIZE) \
                                      / 2);
                for (int i = 0; i < partial_block_size - 1; i++) {
                    double* __distances = _distances + i * (numref - \
                                          nblocks * _BLOCKSIZE) - \
                                          (i + 1) * (i + 2) / 2;
                    for (int j = i + 1; j < partial_block_size; j++) {
                        __distances[j] = sqrt(r2s[i * _BLOCKSIZE + j]);
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
    }
    free(bref);
}

static void _calc_distance_histogram_vectorized(const coordinate* restrict ref,
                                                int numref,
                                                const coordinate* restrict conf,
                                                int numconf,
                                                const float* box, ePBC pbc_type,
                                                double rmin, double rmax,
                                                histbin* restrict histo,
                                                int numhisto)
{
    assert((rmin >= 0.0) && \
           "Minimum distance must be greater than or equal to zero");
    assert(((rmax - rmin) > FLT_EPSILON) && \
           "Maximum distance must be greater than minimum distance.");
    const int nblocks_ref = numref / _BLOCKSIZE;
    const int nblocks_conf = numconf / _BLOCKSIZE;
    const int partial_block_size_ref = numref % _BLOCKSIZE;
    const int partial_block_size_conf = numconf % _BLOCKSIZE;
    float* bref = _get_coords_in_blocks(ref, numref);
    float* bconf = _get_coords_in_blocks(conf, numconf);
    float half_box[3] = {0.0};
    double inverse_binw = numhisto / (rmax - rmin);
    double r2_min = rmin * rmin;
    double r2_max = rmax * rmax;

    switch (pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc_vectorized(bref, numref, box);
            _ortho_pbc_vectorized(bconf, numconf, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc_vectorized(bref, numref, box);
            _triclinic_pbc_vectorized(bconf, numconf, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        histbin* local_histo = (histbin*) calloc(numhisto + 1, sizeof(histbin));
        double* dxs = (double*) aligned_calloc(_3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(_BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * _BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks_ref; n++) {
            // process blocks of the n-th row
            // (_BLOCKSIZE x _BLOCKSIZE squares):
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bconf + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    if ((r2s[i] >= r2_min) && (r2s[i] <= r2_max)) {
                        int k = (int) ((sqrt(r2s[i]) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k]++;
                        }
                    }
                }
            }
            // process remaining partial block of the n-th row
            // (_BLOCKSIZE x partial_block_size_conf rectangle):
            if (partial_block_size_conf > 0){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bconf + \
                                             nblocks_conf * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k]++;
                            }
                        }
                    }
                }
            }
        }
        // process remaining partial blocks after the last row
        // (partial_block_size_ref x _BLOCKSIZE rectangles):
        if (partial_block_size_ref > 0){
#ifdef PARALLEL
            #pragma omp for schedule(static, 1) nowait
#endif
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs,
                                             bref + nblocks_ref * _3_BLOCKSIZE,
                                             bconf + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k]++;
                            }
                        }
                    }
                }
            }
        }
#ifdef PARALLEL
        #pragma omp single nowait
#endif
        {
            // process remaining bottom-right partial block
            // (partial_block_size_ref x partial_block_size_conf rectangle):
            if (partial_block_size_ref * partial_block_size_conf > 0) {
                _calc_distance_vectors_block(dxs, bref + \
                                             nblocks_ref * _3_BLOCKSIZE,
                                             bconf + \
                                             nblocks_conf * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                            _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k]++;
                            }
                        }
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
        // gather local results from threads
        for (int i = 0; i < numhisto; i++) {
#ifdef PARALLEL
            #pragma omp atomic
#endif
            histo[i] += local_histo[i];
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        histo[numhisto - 1] += local_histo[numhisto]; // Numpy consistency
        free(local_histo);
    }
    free(bref);
    free(bconf);
}

static void _calc_self_distance_histogram_vectorized(const coordinate* \
                                                     restrict ref, int numref,
                                                     const float* box,
                                                     ePBC pbc_type,
                                                     double rmin, double rmax,
                                                     histbin* restrict histo,
                                                     int numhisto)
{
    assert((rmin >= 0.0) && \
           "Minimum distance must be greater than or equal to zero");
    assert(((rmax - rmin) > FLT_EPSILON) && \
           "Maximum distance must be greater than minimum distance.");
    const int nblocks = numref / _BLOCKSIZE;
    const int partial_block_size = numref % _BLOCKSIZE;
    float* bref = _get_coords_in_blocks(ref, numref);
    float half_box[3] = {0.0};
    double inverse_binw = numhisto / (rmax - rmin);
    double r2_min = rmin * rmin;
    double r2_max = rmax * rmax;

    switch (pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc_vectorized(bref, numref, box);
            break;
        case PBCtriclinic:
            _triclinic_pbc_vectorized(bref, numref, box);
            break;
        case PBCnone:
            break;
        default:
            //TODO: fatal error
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        histbin* local_histo = (histbin*) calloc(numhisto + 1, sizeof(histbin));
        double* dxs = (double*) aligned_calloc(_3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(_BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * _BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (int n = 0; n < nblocks; n++) {
            // process first block of the n-th row
            // (_BLOCKSIZE x (_BLOCKSIZE - 1) triangle)
            _calc_self_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE);
            switch (pbc_type) {
                case PBCortho:
                    _minimum_image_ortho_lazy_block(dxs, box, half_box);
                    break;
                case PBCtriclinic:
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    break;
                default:
                    break;
            };
            _calc_squared_distances_block(r2s, dxs);
            for (int i = 0; i < _BLOCKSIZE - 1; i++) {
                for (int j = i + 1; j < _BLOCKSIZE; j++) {
                    double r2 = r2s[i * _BLOCKSIZE + j];
                    if ((r2 >= r2_min) && (r2 <= r2_max)) {
                        int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k]++;
                        }
                    }
                }
            }
            // process remaining blocks of the n-th row
            // (_BLOCKSIZE x _BLOCKSIZE squares):
            for (int m = n + 1; m < nblocks; m++){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bref + m * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    if ((r2s[i] >= r2_min) && (r2s[i] <= r2_max)) {
                        int k = (int) ((sqrt(r2s[i]) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k]++;
                        }
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (_BLOCKSIZE x partial_block_size rectangle):
            if (partial_block_size > 0){
                _calc_distance_vectors_block(dxs, bref + n * _3_BLOCKSIZE,
                                             bref + nblocks * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                        _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k]++;
                            }
                        }
                    }
                }
            }
        }
#ifdef PARALLEL
        #pragma omp single nowait
#endif
        {
            // process remaining bottom-right partial block:
            // ((partial_block_size - 1) x (partial_block_size - 1) triangle):
            if (partial_block_size > 0) {
                _calc_self_distance_vectors_block(dxs, bref + \
                                                  nblocks * _3_BLOCKSIZE);
                switch (pbc_type) {
                    case PBCortho:
                        _minimum_image_ortho_lazy_block(dxs, box, half_box);
                        break;
                    case PBCtriclinic:
                            _minimum_image_triclinic_lazy_block(dxs, box, aux);
                        break;
                    default:
                        break;
                };
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size - 1; i++) {
                    for (int j = i + 1; j < partial_block_size; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k]++;
                            }
                        }
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
        // gather local results from threads
        for (int i = 0; i < numhisto; i++) {
#ifdef PARALLEL
            #pragma omp atomic
#endif
            histo[i] += local_histo[i];
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        histo[numhisto - 1] += local_histo[numhisto]; // Numpy consistency
        free(local_histo);
    }
    free(bref);
}


static void _calc_distance_histogram_grid_based(coordinate* restrict ref,
                                                int numref,
                                                coordinate* restrict conf,
                                                int numconf,
                                                float* box, ePBC pbc_type,
                                                double rmin, double rmax,
                                                histbin* restrict histo,
                                                int numhisto)
{
    assert((rmin >= 0.0) && \
           "Minimum distance must be greater than or equal to zero");
    assert(((rmax - rmin) > FLT_EPSILON) && \
           "Maximum distance must be greater than minimum distance.");
    float grid_box[3] = {0.0, 0.0, 0.0};
    double cell_edge_len = rmax;
    float cell[3] = {0.0, 0.0, 0.0};
    size_t numcells[3] = {0, 0, 0};
    size_t numcells_tot = 1;
    int do_brute_force = 0;
    int use_cell_octants = 1;
    double cell_edge_len_octant_threshold = 4.0 / 3.0 * rmax;

    switch (pbc_type) {
        case PBCortho:
            break;
        case PBCtriclinic:
            //TODO: _ortho_box_from_triclinic_box(grid_box, box);
            break;
        case PBCnone:
            //TODO: _ortho_box_from_coords(grid_box, ref, numref, conf, numconf);
            break;
        default:
            //TODO: fatal error
            break;
    };

    do {
        numcells_tot = 1;
        do_brute_force = 0;
        for (size_t i = 0; i < 3; i++) {
            numcells[i] = (size_t) (box[i] / cell_edge_len);
            if (numcells[i] < 4) {
                do_brute_force++;
                // Check if cell_edge_len is too large for grid search:
                if ((do_brute_force > 2) || (numcells[i] < 2)) {
                    if (numconf < _BLOCKSIZE) {
                        _calc_distance_histogram(ref, numref, conf, numconf,
                                                 box, pbc_type, rmin, rmax,
                                                 histo, numhisto);
                    }
                    else {
                        _calc_distance_histogram_vectorized(ref, numref,
                                                            conf, numconf,
                                                            box, pbc_type,
                                                            rmin, rmax,
                                                            histo, numhisto);
                    }
                    return;
                }
            }
            numcells_tot *= numcells[i];
            cell[i] = box[i] / numcells[i];
        }
        // If there are too many cells, we need to increase the cell edge
        // length. Edge lengths between 4*rmax/3 and 2*rmax are no-man's-land,
        // where it's better to go to 2*rmax and only search cells adjacent to
        // a reference particle's cell octant.
        if (numcells_tot > __MAX_NUM_GRID_CELLS) {
            cell_edge_len *= 1.1;
            if ((cell_edge_len > cell_edge_len_octant_threshold) && \
                (cell_edge_len < (2.0 * rmax))) {
                cell_edge_len = 2.0 * rmax;
            }
        }
    } while (numcells_tot > __MAX_NUM_GRID_CELLS);
    assert(((numcells_tot >= 16) && (numcells_tot <= __MAX_NUM_GRID_CELLS)) && \
            "(numcells_tot < 16) or (numcells_tot > __MAX_NUM_GRID_CELLS)");

    float half_box[3] = {0.0, 0.0, 0.0};
    double inverse_binw = numhisto / (rmax - rmin);
    double r2_min = rmin * rmin;
    double r2_max = rmax * rmax;
    float inv_cell[3] = {0.0, 0.0, 0.0};
    size_t max_neighbors[3] = {3, 3, 3};
    size_t num_neighbor_cells = 1;
    size_t cell_offset_y = 0;
    size_t cell_offset_z = 0;

    switch (pbc_type) {
        case PBCortho:
            half_box[0] = 0.5 * box[0];
            half_box[1] = 0.5 * box[1];
            half_box[2] = 0.5 * box[2];
            _ortho_pbc(ref, numref, box);
            _ortho_pbc(conf, numconf, box);
            break;
        case PBCtriclinic:
            box = grid_box;
            _triclinic_pbc(ref, numref, box);
            _triclinic_pbc(conf, numconf, box);
            //TODO: shift (!) particles to new ortho box
            break;
        case PBCnone:
            box = grid_box;
            //TODO: shift (!) particles to new pseudo box
        default:
            break;
    };

    cell_offset_y = numcells[0];
    cell_offset_z = numcells[0] * numcells[1];

    for (size_t i = 0; i < 3; i++) {
        if (cell[i] < (2.0 * rmax)) {
            use_cell_octants = 0;
            break;
        }
    }

    for (size_t i = 0; i < 3; i++){
        inv_cell[i] = 1.0 / cell[i];
        max_neighbors[i] -= (numcells[i] < 3);
        num_neighbor_cells *= max_neighbors[i];
    }

/*    printf("Grid dimensions: %lu x %lu x %lu\n", numcells[0], numcells[1], numcells[2]);*/
/*    printf("Maximum cell id: %lu\n", numcells_tot - 1);*/
/*    printf("Max neighbors: %lu x %lu x %lu\n", max_neighbors[0], max_neighbors[1], max_neighbors[2]);*/
/*    printf("Number of neighbor cells: %lu\n", num_neighbor_cells);*/

    size_t* cellid = (size_t*) calloc(numconf, sizeof(size_t));
    size_t* numconf_cell = (size_t*) calloc(numcells_tot, sizeof(size_t));
    size_t* offset = (size_t*) calloc(numcells_tot + 1, sizeof(size_t));
    coordinate* confgrid = (coordinate*) calloc(numconf, sizeof(coordinate));
    if (!cellid || !numconf_cell || !offset || !confgrid) {
        free(cellid);
        free(numconf_cell);
        free(offset);
        free(confgrid);
        //TODO: fatal error
        return;
    }

    for (size_t i = 0; i < numconf; i++) {
        size_t cid = (size_t) (conf[i][0] * inv_cell[0]) + \
                     (size_t) (conf[i][1] * inv_cell[1]) * cell_offset_y + \
                     (size_t) (conf[i][2] * inv_cell[2]) * cell_offset_z;
        assert((cid < numcells_tot) && "cid >= numcells_tot");
        cellid[i] = cid;
        numconf_cell[cid]++;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < numcells_tot; i++) {
        current_offset += numconf_cell[i];
        offset[i+1] = current_offset;
    }
    // DEBUG consistency check:
    assert((current_offset == numconf) && "current_offset != numconf");

    // Fill coordinates into grid:
#ifdef PARALLEL
    #pragma omp parallel shared(confgrid)
#endif
    {
        size_t cid;
        size_t coord_offset;
        size_t confgrid_offset;
#ifdef PARALLEL
        #pragma omp for nowait
#endif
        for (size_t i = 0; i < numconf; i++) {
            cid = cellid[i];
#ifdef PARALLEL
            #pragma omp critical
#endif
            {
                coord_offset = --numconf_cell[cid];
            }
            confgrid_offset = offset[cid] + coord_offset;
            assert((confgrid_offset < numconf) && "confgrid_offset >= numconf");
            confgrid[confgrid_offset][0] = conf[i][0];
            confgrid[confgrid_offset][1] = conf[i][1];
            confgrid[confgrid_offset][2] = conf[i][2];
        }
    }
    // consistency check (debug only):
    for (size_t i = 0; i < numcells_tot; i++) {
        assert((!numconf_cell[i]) && "numconf_cell != 0");
    }

    free(cellid);
    free(numconf_cell);

    if (use_cell_octants) {
      //TODO
      free(offset);
      free(confgrid);
      return;
    }

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        histbin* local_histo = (histbin*) calloc(numhisto + 1, sizeof(histbin));
        double dx[3];
        size_t ref_cid[3];
        size_t cid[27];
        size_t idx = 0;
#ifdef PARALLEL
        #pragma omp for nowait
#endif
        for (size_t n = 0; n < numref; n++) {
            ref_cid[0] = (size_t) (ref[n][0] * inv_cell[0]);
            ref_cid[1] = (size_t) (ref[n][1] * inv_cell[1]);
            ref_cid[2] = (size_t) (ref[n][2] * inv_cell[2]);
            // consistency check (debug only):
            assert((ref_cid[0] < numcells[0]) && "ref_cid[0] >= numcells[0]");
            assert((ref_cid[1] < numcells[1]) && "ref_cid[1] >= numcells[1]");
            assert((ref_cid[2] < numcells[2]) && "ref_cid[2] >= numcells[2]");
            for (size_t k = 0; k < max_neighbors[2]; k++) {
                size_t cid_z = ref_cid[2] + numcells[2] + k - 1;
                cid_z %= numcells[2];
                cid_z *= cell_offset_z;
                for (size_t j = 0; j < max_neighbors[1]; j++) {
                    size_t cid_y = ref_cid[1] + numcells[1] + j - 1;
                    cid_y %= numcells[1];
                    cid_y *= cell_offset_y;
                    for (size_t i = 0; i < max_neighbors[0]; i++) {
                        size_t cid_x = ref_cid[0] + numcells[0] + i - 1;
                        cid_x %= numcells[0];
                        cid[idx++] = cid_x + cid_y + cid_z;
                        // consistency check (debug only):
                        assert((cid[idx - 1] < numcells_tot) && "neighbor_cid >= numcells_tot");
                    }
                }
            }
            // consistency check (debug only):
            assert((idx == num_neighbor_cells) && "idx != num_neighbor_cells");
            idx = 0;
            for (size_t i = 0; i < num_neighbor_cells; i++) {
                for (size_t j = offset[cid[i]]; j < offset[cid[i] + 1]; j++) {
                    dx[0] = confgrid[j][0] - ref[n][0];
                    dx[1] = confgrid[j][1] - ref[n][1];
                    dx[2] = confgrid[j][2] - ref[n][2];
                    if (pbc_type != PBCnone) {
                        _minimum_image_ortho_lazy(dx, box, half_box);
                    }
                    double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                    if ((r2 >= r2_min) && (r2 <= r2_max)) {
                        int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                        assert((k <= numhisto) && "k > numhisto");
                        if (k >= 0) {
                            local_histo[k]++;
                        }
                    }
                }
            }
        }
        // Gather local results:
        for (int i = 0; i < numhisto; i++) {
#ifdef PARALLEL
            #pragma omp atomic
#endif
            histo[i] += local_histo[i];
        }
#ifdef PARALLEL
        #pragma omp atomic
#endif
        histo[numhisto - 1] += local_histo[numhisto]; // Numpy consistency
        free(local_histo);
    }
    free(offset);
    free(confgrid);
}
#endif /* __DISTANCES_H */
