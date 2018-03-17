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
 * Note that __BLOCKSIZE * sizeof(float) MUST be an integer multiple of
 * __MEMORY_ALIGNMENT!
 */
#define __BLOCKSIZE (__MEMORY_ALIGNMENT / 2)
#define __2_BLOCKSIZE (__BLOCKSIZE + __BLOCKSIZE)
#define __BLOCKSIZE_2 (__BLOCKSIZE * __BLOCKSIZE)
#define __2_BLOCKSIZE_2 (__BLOCKSIZE_2 + __BLOCKSIZE_2)
#define __3_BLOCKSIZE (3 * __BLOCKSIZE)
#define __3_BLOCKSIZE_2 (3 * __BLOCKSIZE_2)

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

// Assert that __BLOCKSIZE * sizeof(float) is divisible by __MEMORY_ALIGNMENT:
STATIC_ASSERT(!(__BLOCKSIZE * sizeof(float) % __MEMORY_ALIGNMENT), \
"__BLOCKSIZE*sizeof(float) is not an integer multiple of __MEMORY_ALIGNMENT!");

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
    for (int i=0; i < numcoords; i++){
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
        default:
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
            double rsq = (dx[0] * dx[0]) + (dx[1] * dx[1]) + (dx[2] * dx[2]);
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
        default:
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
            double rsq = (dx[0] * dx[0]) + (dx[1] * dx[1]) + (dx[2] * dx[2]);
            *(distances + distpos) = sqrt(rsq);
            distpos += 1;
        }
    }
}

static void _calc_distance_histogram(coordinate* ref, int numref,
                                     coordinate* conf, int numconf,
                                     float* box, ePBC pbc_type,
                                     double binw, histbin* histo, int numhisto)
{
    double inverse_binw = 1.0 / binw;
    double r2_max = (binw * numhisto) * (binw * numhisto);
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
        default:
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        int k;
        double dx[3];
        double r2;
#ifdef PARALLEL
        histbin* thread_local_histo = (histbin*) calloc((size_t) numhisto,
                                                        sizeof(histbin));
        assert(thread_local_histo != NULL);
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
                r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if (r2 < r2_max) {
                    k = (int) (sqrt(r2) * inverse_binw);
#ifdef PARALLEL
                    thread_local_histo[k] += 1;
#else
                    histo[k] += 1;
#endif
                }
            }
        }
#ifdef PARALLEL
        // gather local results from threads
        for (int i = 0; i < numhisto; i++) {
            #pragma omp atomic
            histo[i] += thread_local_histo[i];
        }
        free(thread_local_histo);
#endif
    }
}

static void _calc_self_distance_histogram(coordinate* ref, int numref,
                                          float* box, ePBC pbc_type,
                                          double binw, histbin* histo,
                                          int numhisto)
{
    double inverse_binw = 1.0 / binw;
    double r2_max = (binw * numhisto) * (binw * numhisto);
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
        default:
            break;
    };
#ifdef PARALLEL
    #pragma omp parallel shared(histo)
#endif
    {
        int k;
        double dx[3];
        double r2;
#ifdef PARALLEL
        histbin* thread_local_histo = (histbin*) calloc((size_t) numhisto,
                                                        sizeof(histbin));
        assert(thread_local_histo != NULL);
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
                r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if (r2 < r2_max) {
                    k = (int) (sqrt(r2) * inverse_binw);
#ifdef PARALLEL
                    thread_local_histo[k] += 1;
#else
                    histo[k] += 1;
#endif
                }
            }
        }
#ifdef PARALLEL
        // gather local results from threads
        for (int i = 0; i < numhisto; i++) {
            #pragma omp atomic
            histo[i] += thread_local_histo[i];
        }
        free(thread_local_histo);
#endif
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
        default:
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
        default:
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
        default:
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
 * own. Its interface is the same as for calloc(), but memory alignment and
 * padding is handled automatically. If C11 or POSIX features are not supported,
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
 * @brief Arrange coordinates in blocks of __BLOCKSIZE positions
 *
 * This function takes an array @p pos containing @p npos xyz positions and
 * arranges them into a newly allocated(!) array and returned as a @<float*@>.
 * In this array, coordinates are aligned in blocks of @c __BLOCKSIZE positions
 * where each block contains first all x-, then y-, and finally all
 * z-coordinates. This layout improves memory locality and guarantees 64-byte
 * cachline-optimization. Furthermore, it avoids possible "false sharing"
 * between OpenMP threads.
 */
static inline float* _get_coords_in_blocks(const coordinate* restrict coords,
                                           int numcoords)
{
    int nblocks = numcoords / __BLOCKSIZE;
    int nremaining = numcoords % __BLOCKSIZE;
    int nblocks_to_calloc = nblocks + (nremaining > 0);
    float* bcoords __attaligned = (float*) aligned_calloc( \
                                  nblocks_to_calloc * __3_BLOCKSIZE, \
                                  sizeof(float));
    // process full blocks
#ifdef PARALLEL
    #pragma omp parallel for shared(bcoords)
#endif
    for (int i = 0; i < nblocks; i++) {
        for (int j = 0; j < 3; j++) {
            float* _coords = ((float*) (coords + i * __BLOCKSIZE)) + j;
            float* _bcoords = (float*) __assaligned(bcoords + \
                                                     i * __3_BLOCKSIZE + \
                                                     j * __BLOCKSIZE);
            for (int k = 0; k < __BLOCKSIZE; k++) {
                _bcoords[k] = _coords[3*k];
            }
        }
    }
    // process remaining partial block
    if (nremaining > 0) {
        for (int j = 0; j < 3; j++) {
            float* _coords = ((float*) (coords + nblocks * __BLOCKSIZE)) + j;
            float* _bcoords = (float*) __assaligned(bcoords + \
                                                    nblocks * __3_BLOCKSIZE + \
                                                    j * __BLOCKSIZE);
            for (int k = 0; k < nremaining; k++) {
                _bcoords[k] = _coords[3*k];
            }
        }
    }
    return bcoords;
}

/**
 * @brief Moves block-aligned coordinates into the central periodic image
 *
 * This function takes an array @p coords of @p numcoords block-aligned
 * positions as well as an array @p box containing the box edge lengths of a
 * rectangular simulation box. Folds coordinates which lie outside the box back
 * into the box.
 */
static void _ortho_pbc_vectorized(float* restrict coords, int numcoords,
                                  const float* box)
{
    const int nblocks = numcoords / __BLOCKSIZE + \
                        ((numcoords % __BLOCKSIZE) > 0);
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
        float* s __attaligned = (float*) aligned_calloc(__BLOCKSIZE,
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
                    (float*) __assaligned(coords + n * __3_BLOCKSIZE + \
                                          i * __BLOCKSIZE);
                    bx = box[i];
                    ibx = box_inverse[i];
                    for (int j = 0; j < __BLOCKSIZE; j++) {
                        s[j] = _coords[j] * ibx;
                    }
                    for (int j = 0; j < __BLOCKSIZE; j++) {
                        s[j] = floorf(s[j]);
                    }
                    for (int j = 0; j < __BLOCKSIZE; j++) {
                        _coords[j] -= bx * s[j];
                    }
                }
            }
        }
        free(s);
    }
}

static void _triclinic_pbc_vectorized(float* restrict coords, int numcoords,
                                      const float* box_vectors)
{
    // Moves all coordinates to within the box boundaries for a triclinic box
    // Assumes box_vectors having zero values for box_vectors[1], box_vectors[2]
    // and box_vectors[5]
    const int nblocks = numcoords / __BLOCKSIZE + \
                        ((numcoords % __BLOCKSIZE) > 0);
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
        float* s __attaligned = (float*) aligned_calloc(__BLOCKSIZE,
                                                        sizeof(float));
        __memaligned float bxv;
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks; n++) {
            float* x_coords __attaligned = \
            (float*) __assaligned(coords + n * __3_BLOCKSIZE);
            float* y_coords __attaligned = \
            (float*) __assaligned(coords + n * __3_BLOCKSIZE + __BLOCKSIZE);
            float* z_coords __attaligned = \
            (float*) __assaligned(coords + n * __3_BLOCKSIZE + __2_BLOCKSIZE);
            // translate x-, y-, and z-coordinates to central cell along c-axis
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = z_coords[i] * bi8;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[6];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[7];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                y_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[8];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                z_coords[i] -= s[i] * bxv;
            }
            // translate x- and y-coordinates to central cell along b-axis
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = y_coords[i] * bi4;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] += z_coords[i] * bi7;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[3];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
            bxv = box_vectors[4];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                y_coords[i] -= s[i] * bxv;
            }
            // translate x-coordinates to central cell along a-axis
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = x_coords[i] * bi0;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] += y_coords[i] * bi3;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] += z_coords[i] * bi6;
            }
            for (int i = 0; i < __BLOCKSIZE; i++) {
                s[i] = floorf(s[i]);
            }
            bxv = box_vectors[0];
            for (int i = 0; i < __BLOCKSIZE; i++) {
                x_coords[i] -= s[i] * bxv;
            }
        }
        free(s);
    }
}

/**
 * @brief Computes all distances within a block of @c __BLOCKSIZE positions
 *
 * This function takes an array @p refs of @c __BLOCKSIZE block-aligned
 * positions and computes all @<__BLOCKSIZE * __BLOCKSIZE@> pairwise distance
 * vectors, which are stored in the provided @p dxs array.
 * When SIMD-vectorized by the compiler, this routine should be faster than
 * computing only the unique @<__BLOCKSIZE * (__BLOCKSIZE - 1) / 2@> distances.
 */
static inline void _calc_self_distance_vectors_block(double* restrict dxs,
                                                     const float* restrict refs)
{
    dxs = (double*) __assaligned(dxs);
    refs = (float*) __assaligned(refs);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < __BLOCKSIZE; j++) {
            double* _dxs = (double*) __assaligned(dxs + i * __BLOCKSIZE_2 + \
                                                  j * __BLOCKSIZE);
            float* _refs = (float*) __assaligned(refs + i * __BLOCKSIZE);
            __memaligned float _ref = refs[i*__BLOCKSIZE+j];
            for (int k = 0; k < __BLOCKSIZE; k++) {
                _dxs[k] = _refs[k] - _ref;
            }
        }
    }
}

/**
 * @brief Computes all distances between two blocks of @c __BLOCKSIZE positions
 *
 * This function takes two arrays @p refs and @p confs, each containing
 * @c __BLOCKSIZE block-aligned positions. It computes all
 * @<__BLOCKSIZE * __BLOCKSIZE@> pairwise distance vectors between the two
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
        for (int j = 0; j < __BLOCKSIZE; j++) {
            double* _dxs = (double*) __assaligned(dxs + i * __BLOCKSIZE_2 + \
                                                  j * __BLOCKSIZE);
            float* _confs = (float*) __assaligned(confs + i * __BLOCKSIZE);
            __memaligned float _ref = refs[i*__BLOCKSIZE+j];
            for (int k = 0; k < __BLOCKSIZE; k++) {
                _dxs[k] = _confs[k] - _ref;
            }
        }
    }
}

/**
 * @brief Compute minimum image representations of distance vectors
 *
 * This function takes an array @p dxs containing @<__BLOCKSIZE * __BLOCKSIZE@>
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
            double* _dxs = (double*) __assaligned(dxs + i * __BLOCKSIZE_2);
            for (int j = 0; j < __BLOCKSIZE_2; j++) {
                _dxs[j] -= ((_dxs[j] > hbx) ? bx : 0.0);
            }
            for (int j = 0; j < __BLOCKSIZE_2; j++) {
                _dxs[j] += ((_dxs[j] <= nhbx) ? bx : 0.0);
            }
        }
    }
}

/**
 * @brief Compute minimum image representations of distance vectors
 *
 * This function takes an array @p dxs containing @<__BLOCKSIZE * __BLOCKSIZE@>
 * distance vectors, an array @p box_vectors containing the box vectors of
 * a triclinic simulation box. It applies the minimum image convention on the
 * distance vectors with respect to the box.
 * The parameter @p aux serves as a container to store intermediate values and
 * must provide enough space to store 11 * __BLOCKSIZE ^ 2 doubles (for
 * __BLOCKSIZE = 32 and sizeof(double) = 8 bytes that's exactly 88 kiB).
 * This avoids repeated memory allocations if the function is called in a loop.
 */
static inline void _minimum_image_triclinic_lazy_block(double* restrict dxs,
                                                       const float* box_vectors,
                                                       double* restrict aux)
{
    // pointers to x-, y-, and z-coordinates of distances:
    double* dx_0 __attaligned = (double*) __assaligned(dxs + 0 * __BLOCKSIZE_2);
    double* dx_1 __attaligned = (double*) __assaligned(dxs + 1 * __BLOCKSIZE_2);
    double* dx_2 __attaligned = (double*) __assaligned(dxs + 2 * __BLOCKSIZE_2);
    // pointers for auxiliary arrays:
    double* rx_0 __attaligned = (double*) __assaligned(aux + 0 * __BLOCKSIZE_2);
    double* ry_0 __attaligned = (double*) __assaligned(aux + 1 * __BLOCKSIZE_2);
    double* ry_1 __attaligned = (double*) __assaligned(aux + 2 * __BLOCKSIZE_2);
    double* rz_0 __attaligned = (double*) __assaligned(aux + 3 * __BLOCKSIZE_2);
    double* rz_1 __attaligned = (double*) __assaligned(aux + 4 * __BLOCKSIZE_2);
    double* rz_2 __attaligned = (double*) __assaligned(aux + 5 * __BLOCKSIZE_2);
    double* d    __attaligned = (double*) __assaligned(aux + 6 * __BLOCKSIZE_2);
    double* min  __attaligned = (double*) __assaligned(aux + 7 * __BLOCKSIZE_2);
    double* dmin_0 __attaligned = (double*) __assaligned(aux + \
                                                             8 * __BLOCKSIZE_2);
    double* dmin_1 __attaligned = (double*) __assaligned(aux + \
                                                             9 * __BLOCKSIZE_2);
    double* dmin_2 __attaligned = (double*) __assaligned(aux + \
                                                            10 * __BLOCKSIZE_2);
    // auxiliary variables:
    __memaligned double xb0;
    __memaligned double yb3;
    __memaligned double yb4;
    __memaligned double zb6;
    __memaligned double zb7;
    __memaligned double zb8;
    // initialize min, dmin_0, dmin_1, and dmin_2 in a single loop:
    __memaligned double flt_max = FLT_MAX;
    for (int i = 0; i < 4 * __BLOCKSIZE_2; i++) {
        min[i] = flt_max;
    }
    // now do the actual minimum image computation:
    for (int x = -1; x < 2; x++) {
        xb0 = x * box_vectors[0];
        for(int i = 0; i < __BLOCKSIZE_2; i++) {
            rx_0[i] = dx_0[i] + xb0;
        }
        for (int y = -1; y < 2; y++) {
            yb3 = y * box_vectors[3];
            yb4 = y * box_vectors[4];
            for(int i = 0; i < __BLOCKSIZE_2; i++) {
                ry_0[i] = rx_0[i] + yb3;
            }
            for(int i = 0; i < __BLOCKSIZE_2; i++) {
                ry_1[i] = dx_1[i] + yb4;
            }
            for (int z = -1; z < 2; z++) {
                zb6 = z * box_vectors[6];
                zb7 = z * box_vectors[7];
                zb8 = z * box_vectors[8];
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    rz_0[i] = ry_0[i] + zb6;
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    rz_1[i] = ry_1[i] + zb7;
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    rz_2[i] = dx_2[i] + zb8;
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    d[i] = rz_0[i] * rz_0[i];
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    d[i] += rz_1[i] * rz_1[i];
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    d[i] += rz_2[i] * rz_2[i];
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    dmin_0[i] = ((d[i] < min[i]) ? rz_0[i] : dmin_0[i]);
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    dmin_1[i] = ((d[i] < min[i]) ? rz_1[i] : dmin_1[i]);
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    dmin_2[i] = ((d[i] < min[i]) ? rz_2[i] : dmin_2[i]);
                }
                for(int i = 0; i < __BLOCKSIZE_2; i++) {
                    min[i] = ((d[i] < min[i]) ? d[i] : min[i]);
                }
            }
        }
    }
    dxs = (double*) __assaligned(dxs);
    for(int i = 0; i < __3_BLOCKSIZE_2; i++) {
        dxs[i] = dmin_0[i];
    }
}

/**
 * @brief Compute squared distances from distance vectors
 *
 * This function takes an array @p dxs containing @<__BLOCKSIZE * __BLOCKSIZE@>
 * distance vectors and computes their squared Euclidean norms, which are
 * stored in the provided @p r2s array.
 */
static inline void _calc_squared_distances_block(double* restrict r2s,
                                                 const double* restrict dxs)
{
    r2s = (double*) __assaligned(r2s);
    r2s = (double*) memset((void*) r2s, 0, __BLOCKSIZE_2 * sizeof(double));
    for (int i = 0; i < 3; i++) {
        double* _dxs __attaligned = (double*) __assaligned(dxs + \
                                                           i * __BLOCKSIZE_2);
        double* _r2s __attaligned = (double*) __assaligned(r2s);
        for (int j = 0; j < __BLOCKSIZE_2; j++) {
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
    const int nblocks_ref = numref / __BLOCKSIZE;
    const int nblocks_conf = numconf / __BLOCKSIZE;
    const int partial_block_size_ref = numref % __BLOCKSIZE;
    const int partial_block_size_conf = numconf % __BLOCKSIZE;
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
        default:
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(distances)
#endif
    {
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n = 0; n < nblocks_ref; n++) {
            // process blocks of the n-th row
            // (__BLOCKSIZE x __BLOCKSIZE squares):
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bconf + m * __3_BLOCKSIZE);
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
                double* _dists = distances + __BLOCKSIZE * (n * numconf + m);
                for (int i = 0; i < __BLOCKSIZE; i++) {
                    for (int j = 0; j < __BLOCKSIZE; j++) {
                        _dists[i*numconf+j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (__BLOCKSIZE x partial_block_size_conf rectangles):
            if (partial_block_size_conf > 0){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bconf + nblocks_conf * \
                                             __3_BLOCKSIZE);
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
                                 __BLOCKSIZE * (n * numconf + nblocks_conf);
                for (int i = 0; i < __BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        _dists[i*numconf+j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                    }
                }
            }
        }
        // process remaining partial blocks after the last row
        // (partial_block_size_ref x __BLOCKSIZE rectangles):
        if (partial_block_size_ref > 0){
#ifdef PARALLEL
            #pragma omp for schedule(static, 1) nowait
#endif
            for (int m = 0; m < nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs,
                                             bref + nblocks_ref * __3_BLOCKSIZE,
                                             bconf + m * __3_BLOCKSIZE);
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
                                 __BLOCKSIZE * (nblocks_ref * numconf + m);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j=0; j<__BLOCKSIZE; j++) {
                        _dists[i*numconf+j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
    }
    // process remaining bottom-right partial block
    // (partial_block_size_ref x partial_block_size_conf rectangle):
    if (partial_block_size_ref * partial_block_size_conf > 0) {
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        _calc_distance_vectors_block(dxs, bref + nblocks_ref * __3_BLOCKSIZE,
                                     bconf + nblocks_conf * __3_BLOCKSIZE);
        switch (pbc_type) {
            case PBCortho:
                _minimum_image_ortho_lazy_block(dxs, box, half_box);
                break;
            case PBCtriclinic:
                {
                    double* aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2,
                                                           sizeof(double));
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    free(aux);
                }
                break;
            default:
                break;
        };
        _calc_squared_distances_block(r2s, dxs);
        double* _dists = distances + \
                         __BLOCKSIZE * (nblocks_ref * numconf + nblocks_conf);
        for (int i=0; i<partial_block_size_ref; i++) {
            for (int j=0; j<partial_block_size_conf; j++) {
                _dists[i*numconf+j] = sqrt(r2s[i*__BLOCKSIZE+j]);
            }
        }
        free(dxs);
        free(r2s);
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
    const int nblocks = numref / __BLOCKSIZE;
    const int partial_block_size = numref % __BLOCKSIZE;
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
        default:
            break;
    };
#ifdef PARALLEL
    #pragma omp parallel
#endif
    {
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (int n=0; n<nblocks; n++) {
            // process first block of the n-th row
            // ((__BLOCKSIZE - 1) x (__BLOCKSIZE - 1) triangle):
            _calc_self_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE);
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
            double* _distances = distances + n * (numref * __BLOCKSIZE - \
                                 (n * __BLOCKSIZE_2 + __BLOCKSIZE) / 2);
            for (int i=0; i<__BLOCKSIZE-1; i++) {
                double* __distances = _distances + \
                                      i * (numref - n * __BLOCKSIZE) - \
                                      (i + 1) * (i + 2) / 2;
                for (int j=i+1; j<__BLOCKSIZE; j++) {
                    __distances[j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                }
            }
            // process the remaining blocks of the n-th row
            // (__BLOCKSIZE x __BLOCKSIZE squares):
            for (int m=n+1; m<nblocks; m++){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bref + m * __3_BLOCKSIZE);
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
                _distances += __BLOCKSIZE;
                for (int i=0; i<__BLOCKSIZE; i++) {
                    double* __distances = _distances + \
                                          i * (numref - n * __BLOCKSIZE) - \
                                          (i + 1) * (i + 2) / 2;
                    for (int j=0; j<__BLOCKSIZE; j++) {
                        __distances[j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (__BLOCKSIZE x partial_block_size rectangle):
            if (partial_block_size > 0){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bref + nblocks * __3_BLOCKSIZE);
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
                _distances += __BLOCKSIZE;
                for (int i=0; i<__BLOCKSIZE; i++) {
                    double* __distances = _distances + \
                                          i * (numref - n * __BLOCKSIZE) - \
                                          (i + 1) * (i + 2) / 2;
                    for (int j=0; j<partial_block_size; j++) {
                        __distances[j] = sqrt(r2s[i*__BLOCKSIZE+j]);
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
    }
    // process remaining bottom-right partial block:
    // ((partial_block_size - 1) x (partial_block_size - 1) triangle):
    if (partial_block_size > 0) {
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        _calc_self_distance_vectors_block(dxs, bref + nblocks * __3_BLOCKSIZE);
        switch (pbc_type) {
            case PBCortho:
                _minimum_image_ortho_lazy_block(dxs, box, half_box);
                break;
            case PBCtriclinic:
                {
                    double* aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2,
                                                           sizeof(double));
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    free(aux);
                }
                break;
            default:
                break;
        };
        _calc_squared_distances_block(r2s, dxs);
        double * _distances = distances + nblocks * (numref * __BLOCKSIZE - \
                              (nblocks * __BLOCKSIZE_2 + __BLOCKSIZE) / 2);
        for (int i=0; i<partial_block_size-1; i++) {
            double* __distances = _distances + \
                                  i * (numref - nblocks * __BLOCKSIZE) - \
                                  (i + 1) * (i + 2) / 2;
            for (int j=i+1; j<partial_block_size; j++) {
                __distances[j] = sqrt(r2s[i*__BLOCKSIZE+j]);
            }
        }
        free(dxs);
        free(r2s);
    }
    free(bref);
}

static void _calc_distance_histogram_vectorized(const coordinate* restrict ref,
                                                int numref,
                                                const coordinate* restrict conf,
                                                int numconf,
                                                const float* box, ePBC pbc_type,
                                                double binw,
                                                histbin* restrict histo,
                                                int numhisto)
{
    double inverse_binw = 1.0 / binw;
    double r2_max = (binw * numhisto) * (binw * numhisto);
    const int nblocks_ref = numref / __BLOCKSIZE;
    const int nblocks_conf = numconf / __BLOCKSIZE;
    const int partial_block_size_ref = numref % __BLOCKSIZE;
    const int partial_block_size_conf = numconf % __BLOCKSIZE;
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
        default:
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
    {
        histbin* thread_local_histo = \
        (histbin*) aligned_calloc(numhisto + 1, sizeof(histbin));
#endif
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(static, 1) nowait
#endif
        for (int n=0; n<nblocks_ref; n++) {
            int k;
            // process blocks of the n-th row
            // (__BLOCKSIZE x __BLOCKSIZE squares):
            for (int m=0; m<nblocks_conf; m++) {
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bconf + m * __3_BLOCKSIZE);
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
                for (int i=0; i<__BLOCKSIZE_2; i++) {
                    if (r2s[i] < r2_max) {
                        k = (int) (sqrt(r2s[i]) * inverse_binw);
#ifdef PARALLEL
                        thread_local_histo[k] += 1;
#else
                        histo[k] += 1;
#endif
                    }
                }
            }
            // process remaining partial block of the n-th row
            // (__BLOCKSIZE x partial_block_size_conf rectangle):
            if (partial_block_size_conf > 0){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bconf + \
                                             nblocks_conf * __3_BLOCKSIZE);
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
                for (int i=0; i<__BLOCKSIZE; i++) {
                    for (int j=0; j<partial_block_size_conf; j++) {
                        if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                            k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * \
                                      inverse_binw);
#ifdef PARALLEL
                            thread_local_histo[k] += 1;
#else
                            histo[k] += 1;
#endif
                        }
                    }
                }
            }
        }
        // process remaining partial blocks after the last row
        // (partial_block_size_ref x __BLOCKSIZE rectangles):
        if (partial_block_size_ref > 0){
#ifdef PARALLEL
            #pragma omp for schedule(static, 1) nowait
#endif
            for (int m=0; m<nblocks_conf; m++) {
                int k;
                _calc_distance_vectors_block(dxs,
                                             bref + nblocks_ref * __3_BLOCKSIZE,
                                             bconf + m * __3_BLOCKSIZE);
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
                for (int i=0; i<partial_block_size_ref; i++) {
                    for (int j=0; j<__BLOCKSIZE; j++) {
                        if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                            k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * \
                                      inverse_binw);
#ifdef PARALLEL
                            thread_local_histo[k] += 1;
#else
                            histo[k] += 1;
#endif
                        }
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
#ifdef PARALLEL
        // gather local results from threads
        #pragma omp critical
        {
            for (int i=0; i<numhisto; i++) {
                histo[i] += thread_local_histo[i];
            }
        }
        free(thread_local_histo);
    }
#endif
    // process remaining bottom-right partial block
    // (partial_block_size_ref x partial_block_size_conf rectangle):
    if (partial_block_size_ref * partial_block_size_conf > 0) {
        int k;
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        _calc_distance_vectors_block(dxs, bref + nblocks_ref * __3_BLOCKSIZE,
                                     bconf + nblocks_conf * __3_BLOCKSIZE);
        switch (pbc_type) {
            case PBCortho:
                _minimum_image_ortho_lazy_block(dxs, box, half_box);
                break;
            case PBCtriclinic:
                {
                    double* aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2,
                                                           sizeof(double));
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    free(aux);
                }
                break;
            default:
                break;
        };
        _calc_squared_distances_block(r2s, dxs);
        for (int i=0; i<partial_block_size_ref; i++) {
            for (int j=0; j<partial_block_size_conf; j++) {
                if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                    k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * inverse_binw);
                    histo[k] += 1;
                }
            }
        }
        free(dxs);
        free(r2s);
    }
    free(bref);
    free(bconf);
}

static void _calc_self_distance_histogram_vectorized(const coordinate* \
                                                     restrict ref, int numref,
                                                     const float* box,
                                                     ePBC pbc_type, double binw,
                                                     histbin* restrict histo,
                                                     int numhisto)
{
    double inverse_binw = 1.0 / binw;
    double r2_max = (binw * numhisto) * (binw * numhisto);
    const int nblocks = numref / __BLOCKSIZE;
    const int partial_block_size = numref % __BLOCKSIZE;
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
        default:
            break;
    };

#ifdef PARALLEL
    #pragma omp parallel shared(histo)
    {
        histbin* thread_local_histo = \
        (histbin*) aligned_calloc(numhisto + 1, sizeof(histbin));
#endif
        int k;
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        double* aux = NULL;
        if (pbc_type == PBCtriclinic) {
            aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2, sizeof(double));
        }
#ifdef PARALLEL
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (int n=0; n<nblocks; n++) {
            // process first block of the n-th row
            // (__BLOCKSIZE x (__BLOCKSIZE - 1) triangle)
            _calc_self_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE);
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
            for (int i=0; i<__BLOCKSIZE-1; i++) {
                for (int j=i+1; j<__BLOCKSIZE; j++) {
                    if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                        k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * inverse_binw);
#ifdef PARALLEL
                        thread_local_histo[k] += 1;
#else
                        histo[k] += 1;
#endif
                    }
                }
            }
            // process remaining blocks of the n-th row
            // (__BLOCKSIZE x __BLOCKSIZE squares):
            for (int m=n+1; m<nblocks; m++){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bref + m * __3_BLOCKSIZE);
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
                for (int i=0; i<__BLOCKSIZE_2; i++) {
                    if (r2s[i] < r2_max) {
                        k = (int) (sqrt(r2s[i]) * inverse_binw);
#ifdef PARALLEL
                        thread_local_histo[k] += 1;
#else
                        histo[k] += 1;
#endif
                    }
                }
            }
            // process the remaining partial block of the n-th row
            // (__BLOCKSIZE x partial_block_size rectangle):
            if (partial_block_size > 0){
                _calc_distance_vectors_block(dxs, bref + n * __3_BLOCKSIZE,
                                             bref + nblocks * __3_BLOCKSIZE);
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
                for (int i=0; i<__BLOCKSIZE; i++) {
                    for (int j=0; j<partial_block_size; j++) {
                        if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                            k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * \
                                      inverse_binw);
#ifdef PARALLEL
                            thread_local_histo[k] += 1;
#else
                            histo[k] += 1;
#endif
                        }
                    }
                }
            }
        }
        free(dxs);
        free(r2s);
        free(aux);
#ifdef PARALLEL
        // gather local results from threads
        #pragma omp critical
        {
            for (int i=0; i<numhisto; i++) {
                histo[i] += thread_local_histo[i];
            }
        }
        free(thread_local_histo);
    }
#endif
    // process remaining bottom-right partial block:
    // ((partial_block_size - 1) x (partial_block_size - 1) triangle):
    if (partial_block_size > 0) {
        int k;
        double* dxs = (double*) aligned_calloc(__3_BLOCKSIZE_2, sizeof(double));
        double* r2s = (double*) aligned_calloc(__BLOCKSIZE_2, sizeof(double));
        _calc_self_distance_vectors_block(dxs, bref + nblocks * __3_BLOCKSIZE);
        switch (pbc_type) {
            case PBCortho:
                _minimum_image_ortho_lazy_block(dxs, box, half_box);
                break;
            case PBCtriclinic:
                {
                    double* aux = (double*) aligned_calloc(11 * __BLOCKSIZE_2,
                                                           sizeof(double));
                    _minimum_image_triclinic_lazy_block(dxs, box, aux);
                    free(aux);
                }
                break;
            default:
                break;
        };
        _calc_squared_distances_block(r2s, dxs);
        for (int i=0; i<partial_block_size-1; i++) {
            for (int j=i+1; j<partial_block_size; j++) {
                if (r2s[i*__BLOCKSIZE+j] < r2_max) {
                    k = (int) (sqrt(r2s[i*__BLOCKSIZE+j]) * inverse_binw);
                    histo[k] += 1;
                }
            }
        }
        free(dxs);
        free(r2s);
    }
    free(bref);
}
#endif /* __DISTANCES_H */
