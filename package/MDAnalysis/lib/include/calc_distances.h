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
 *   - math.h: for square roots, rounding, trigonometric functions, ...
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


/**
 * @brief Minimum image convention for distances in orthorhombic boxes
 */
static inline void minimum_image(double* x, float* box, float* inverse_box)
{
  int i;
  double s;
  for (i=0; i<3; i++) {
    if (box[i] > FLT_EPSILON) {
      s = inverse_box[i] * x[i];
      x[i] = box[i] * (s - round(s));
    }
  }
}

/**
 * @brief Minimum image convention for orthorhombic boxes.
 *
 * Assumes that the maximum separation is less than 1.5 times the box length
 * (enforced in distance_array functions by packing all particles into the
 * box before calculating separations). For O(n) algorithms it should be faster
 * to use _minimum_image_ortho() without prior packing!
 */
static inline void _minimum_image_ortho_lazy(double* dx, float* box,
                                             float* half_box)
{
    for (int i = 0; i < 3; i++) {
        dx[i] -= (dx[i] > half_box[i]) * box[i];
        dx[i] += (dx[i] <= -half_box[i]) * box[i];
    }
}

/**
 * @brief Minimum image convention for triclinic boxes.
 *
 * Modelled after domain.cpp in LAMMPS. Assumes that there is a maximum
 * separation of 1 box length (enforced in distance functions by moving all
 * particles into the box before calculating separations). Requires a box
 * @p box_vectors corresponding to the flattened output of
 * MDAnalysis.lib.mdamath.triclinic_vectors()). Assumes that box_vectors[1],
 * box_vectors[2] and box_vectors[5] are zero.
 */
static inline void minimum_image_triclinic(double* dx, float* box)
{
   /*
    * Assumes box having zero values for box[1], box[2] and box[5]:
    *   /  a_x   0    0   \                 /  0    1    2  \
    *   |  b_x  b_y   0   |       indices:  |  3    4    5  |
    *   \  c_x  c_y  c_z  /                 \  6    7    8  /
    */
    double dx_min[3] = {0.0, 0.0, 0.0};
    double dsq_min = FLT_MAX;
    double dsq;
    double rx;
    double ry[2];
    double rz[3];
    int ix, iy, iz;
    for (ix = -1; ix < 2; ++ix) {
        rx = dx[0] + box[0] * ix;
        for (iy = -1; iy < 2; ++iy) {
            ry[0] = rx + box[3] * iy;
            ry[1] = dx[1] + box[4] * iy;
            for (iz = -1; iz < 2; ++iz) {
                rz[0] = ry[0] + box[6] * iz;
                rz[1] = ry[1] + box[7] * iz;
                rz[2] = dx[2] + box[8] * iz;
                dsq = rz[0] * rz[0] + rz[1] * rz[1] + rz[2] * rz[2];
                if (dsq < dsq_min) {
                    dsq_min = dsq;
                    dx_min[0] = rz[0];
                    dx_min[1] = rz[1];
                    dx_min[2] = rz[2];
                }
            }
        }
    }
    dx[0] = dx_min[0];
    dx[1] = dx_min[1];
    dx[2] = dx_min[2];
}

/**
 * Moves all coordinates to within the box boundaries for an orthogonal box.
 *
 * This routine first shifts coordinates by at most one box if necessary.
 * If that is not enough, the number of required box shifts is computed and
 * a multi-box shift is applied instead. The single shift is faster, usually
 * enough and more accurate since the estimation of the number of required
 * box shifts is error-prone if particles reside exactly on a box boundary.
 * In order to guarantee that coordinates lie strictly within the primary
 * image, multi-box shifts are always checked for accuracy and a subsequent
 * single-box shift is applied where necessary.
 */
static void _ortho_pbc(coordinate* coords, int numcoords, float* box)
{
    // nothing to do if the box is all-zeros:
    if (!box[0] && !box[1] && !box[2]) {
        return;
    }

    int i, j, s;
    float crd;
    // inverse box for multi-box shifts:
    const double inverse_box[3] = {1.0 / (double) box[0], \
                                   1.0 / (double) box[1], \
                                   1.0 / (double) box[2]};

   /*
    * NOTE FOR DEVELOPERS:
    * The order of operations matters due to numerical precision. A coordinate
    * residing just below the lower bound of the box might get shifted exactly
    * to the upper bound!
    * Example: -0.0000001 + 10.0 == 10.0 (in single precision)
    * It is therefore important to *first* check for the lower bound and
    * afterwards *always* for the upper bound.
    */

#ifdef PARALLEL
#pragma omp parallel for private(i, j, s, crd) shared(coords)
#endif
    for (i=0; i < numcoords; i++) {
        for (j=0; j < 3; j++) {
            crd = coords[i][j];
            if (crd < 0.0f) {
                crd += box[j];
                // check if multi-box shifts are required:
                if (crd < 0.0f) {
                    s = floor(coords[i][j] * inverse_box[j]);
                    coords[i][j] -= s * box[j];
                    // multi-box shifts might be inexact, so check again:
                    if (coords[i][j] < 0.0f) {
                        coords[i][j] += box[j];
                    }
                }
                else{
                    coords[i][j] = crd;
                }
            }
            // Don't put an "else" before this! (see note)
            if (crd >= box[j]) {
                crd -= box[j];
                // check if multi-box shifts are required:
                if (crd >= box[j]) {
                    s = floor(coords[i][j] * inverse_box[j]);
                    coords[i][j] -= s * box[j];
                    // multi-box shifts might be inexact, so check again:
                    if (coords[i][j] >= box[j]) {
                        coords[i][j] -= box[j];
                    }
                }
                else{
                    coords[i][j] = crd;
                }
            }
        }
    }
}

/**
 * @brief Wraps all coordinates into a triclinic box.
 * 
 * Folds all @p coords into a triclinic box given by @p box_vectors according to
 * periodic boundary conditions in all three dimensions. Requires that
 * @p box_vectors corresponds to the flattened output of
 * MDAnalysis.lib.mdamath.triclinic_vectors()). Assumes that box_vectors[1],
 * box_vectors[2] and box_vectors[5] are zero.
 */
static void _triclinic_pbc(coordinate* coords, int numcoords, float* box)
{
   /* Moves all coordinates to within the box boundaries for a triclinic box.
    * Assumes that the box has zero values for box[1], box[2] and box[5]:
    *   [ a_x,   0,   0 ]                 [ 0, 1, 2 ]
    *   [ b_x, b_y,   0 ]       indices:  [ 3, 4, 5 ]
    *   [ c_x, c_y, c_z ]                 [ 6, 7, 8 ]
    *
    * Inverse of matrix box (here called "m"):
    *   [                       1/m0,           0,    0 ]
    *   [                -m3/(m0*m4),        1/m4,    0 ]
    *   [ (m3*m7/(m0*m4) - m6/m0)/m8, -m7/(m4*m8), 1/m8 ]
    *
    * This routine first shifts coordinates by at most one box if necessary.
    * If that is not enough, the number of required box shifts is computed and
    * a multi-box shift is applied instead. The single shift is faster, usually
    * enough and more accurate since the estimation of the number of required
    * box shifts is error-prone if particles reside exactly on a box boundary.
    * In order to guarantee that coordinates lie strictly within the primary
    * image, multi-box shifts are always checked for accuracy and a subsequent
    * single-box shift is applied where necessary.
    */

    // nothing to do if the box diagonal is all-zeros:
    if (!box[0] && !box[4] && !box[8]) {
        return;
    }

    int i, s, msr;
    float crd[3];
    // constants for multi-box shifts:
    const double bi0 = 1.0 / (double) box[0];
    const double bi4 = 1.0 / (double) box[4];
    const double bi8 = 1.0 / (double) box[8];
    const double bi3 = -box[3] * bi0 * bi4;
    const double bi6 = (-bi3 * box[7] - box[6] * bi0) * bi8;
    const double bi7 = -box[7] * bi4 * bi8;
    // variables and constants for single box shifts:
    double lbound;
    double ubound;
    const double a_ax_yfactor = (double) box[3] * bi4;;
    const double a_ax_zfactor = (double) box[6] * bi8;
    const double b_ax_zfactor = (double) box[7] * bi8;


   /*
    * NOTE FOR DEVELOPERS:
    * The order of operations matters due to numerical precision. A coordinate
    * residing just below the lower bound of the box might get shifted exactly
    * to the upper bound!
    * Example: -0.0000001 + 10.0 == 10.0 (in single precision)
    * It is therefore important to *first* check for the lower bound and
    * afterwards *always* for the upper bound.
    */

#ifdef PARALLEL
#pragma omp parallel for private(i, s, msr, crd, lbound, ubound) shared(coords)
#endif
    for (i = 0; i < numcoords; i++){
        msr = 0;
        crd[0] = coords[i][0];
        crd[1] = coords[i][1];
        crd[2] = coords[i][2];
        // translate coords[i] to central cell along c-axis
        if (crd[2] < 0.0f) {
            crd[0] +=  box[6];
            crd[1] +=  box[7];
            crd[2] +=  box[8];
            // check if multi-box shifts are required:
            if (crd[2] < 0.0f) {
                msr = 1;
            }
        }
        // Don't put an "else" before this! (see note)
        if (crd[2] >= box[8]) {
            crd[0] -=  box[6];
            crd[1] -=  box[7];
            crd[2] -=  box[8];
            // check if multi-box shifts are required:
            if (crd[2] >= box[8]) {
               msr = 1;
            }
        }
        if (!msr) {
            // translate remainder of crd to central cell along b-axis
            lbound = crd[2] * b_ax_zfactor;
            ubound = lbound + box[4];
            if (crd[1] < lbound) {
                crd[0] += box[3];
                crd[1] += box[4];
                // check if multi-box shifts are required:
                if (crd[1] < lbound) {
                    msr = 1;
                }
            }
            // Don't put an "else" before this! (see note)
            if (crd[1] >= ubound) {
                crd[0] -= box[3];
                crd[1] -= box[4];
                // check if multi-box shifts are required:
                if (crd[1] >= ubound) {
                    msr = 1;
                }
            }
            if (!msr) {
                // translate remainder of crd to central cell along a-axis
                lbound = crd[1] * a_ax_yfactor + crd[2] * a_ax_zfactor;
                ubound = lbound + box[0];
                if (crd[0] < lbound) {
                    crd[0] += box[0];
                    // check if multi-box shifts are required:
                    if (crd[0] < lbound) {
                        msr = 1;
                    }
                }
                // Don't put an "else" before this! (see note)
                if (crd[0] >= ubound) {
                    crd[0] -= box[0];
                    // check if multi-box shifts are required:
                    if (crd[0] >= ubound) {
                        msr = 1;
                    }
                }
            }
        }
        // multi-box shifts required?
        if (msr) {
            // translate coords[i] to central cell along c-axis
            s = floor(coords[i][2] * bi8);
            coords[i][2] -= s * box[8];
            coords[i][1] -= s * box[7];
            coords[i][0] -= s * box[6];
            // translate remainder of coords[i] to central cell along b-axis
            s = floor(coords[i][1] * bi4 + coords[i][2] * bi7);
            coords[i][1] -= s * box[4];
            coords[i][0] -= s * box[3];
            // translate remainder of coords[i] to central cell along a-axis
            s = floor(coords[i][0] * bi0 + coords[i][1] * bi3 + \
                      coords[i][2] * bi6);
            coords[i][0] -= s * box[0];
            // multi-box shifts might be inexact, so check again:
            crd[0] = coords[i][0];
            crd[1] = coords[i][1];
            crd[2] = coords[i][2];
            // translate coords[i] to central cell along c-axis
            if (crd[2] < 0.0f) {
                crd[0] +=  box[6];
                crd[1] +=  box[7];
                crd[2] +=  box[8];
            }
            // Don't put an "else" before this! (see note)
            if (crd[2] >= box[8]) {
                crd[0] -=  box[6];
                crd[1] -=  box[7];
                crd[2] -=  box[8];
            }
            // translate remainder of crd to central cell along b-axis
            lbound = crd[2] * b_ax_zfactor;
            ubound = lbound + box[4];
            if (crd[1] < lbound) {
                crd[0] += box[3];
                crd[1] += box[4];
            }
            // Don't put an "else" before this! (see note)
            if (crd[1] >= ubound) {
                crd[0] -= box[3];
                crd[1] -= box[4];
            }
            // translate remainder of crd to central cell along a-axis
            lbound = crd[1] * a_ax_yfactor + crd[2] * a_ax_zfactor;
            ubound = lbound + box[0];
            if (crd[0] < lbound) {
                crd[0] += box[0];
            }
            // Don't put an "else" before this! (see note)
            if (crd[0] >= ubound) {
                crd[0] -= box[0];
            }
            coords[i][0] = crd[0];
            coords[i][1] = crd[1];
            coords[i][2] = crd[2];
        }
        // single shift was sufficient, apply the result:
        else {
            coords[i][0] = crd[0];
            coords[i][1] = crd[1];
            coords[i][2] = crd[2];
        }
    }
}

static void _calc_distance_array(coordinate* ref, int numref, coordinate* conf,
                                 int numconf, double* distances)
{
  int i, j;
  double dx[3];
  double rsq;

#ifdef PARALLEL
#pragma omp parallel for private(i, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++) {
    for (j=0; j<numconf; j++) {
      dx[0] = conf[j][0] - ref[i][0];
      dx[1] = conf[j][1] - ref[i][1];
      dx[2] = conf[j][2] - ref[i][2];
      rsq = (dx[0]*dx[0]) + (dx[1]*dx[1]) + (dx[2]*dx[2]);
      *(distances+i*numconf+j) = sqrt(rsq);
    }
  }
}

static void _calc_distance_array_ortho(coordinate* ref, int numref, coordinate* conf,
                                       int numconf, float* box, double* distances)
{
  int i, j;
  double dx[3];
  float inverse_box[3];
  double rsq;

  inverse_box[0] = 1.0 / box[0];
  inverse_box[1] = 1.0 / box[1];
  inverse_box[2] = 1.0 / box[2];
#ifdef PARALLEL
#pragma omp parallel for private(i, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++) {
    for (j=0; j<numconf; j++) {
      dx[0] = conf[j][0] - ref[i][0];
      dx[1] = conf[j][1] - ref[i][1];
      dx[2] = conf[j][2] - ref[i][2];
      // Periodic boundaries
      minimum_image(dx, box, inverse_box);
      rsq = (dx[0]*dx[0]) + (dx[1]*dx[1]) + (dx[2]*dx[2]);
      *(distances+i*numconf+j) = sqrt(rsq);
    }
  }
}

static void _calc_distance_array_triclinic(coordinate* ref, int numref,
                                           coordinate* conf, int numconf,
                                           float* box, double* distances)
{
  int i, j;
  double dx[3];
  double rsq;

  // Move coords to inside box
  _triclinic_pbc(ref, numref, box);
  _triclinic_pbc(conf, numconf, box);

#ifdef PARALLEL
#pragma omp parallel for private(i, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++){
    for (j=0; j<numconf; j++){
      dx[0] = conf[j][0] - ref[i][0];
      dx[1] = conf[j][1] - ref[i][1];
      dx[2] = conf[j][2] - ref[i][2];
      minimum_image_triclinic(dx, box);
      rsq = (dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
      *(distances + i*numconf + j) = sqrt(rsq);
    }
  }
}

static void _calc_self_distance_array(coordinate* ref, int numref,
                                      double* distances)
{
  int i, j, distpos;
  double dx[3];
  double rsq;

  distpos = 0;

#ifdef PARALLEL
#pragma omp parallel for private(i, distpos, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++) {
#ifdef PARALLEL
    distpos = i * (2 * numref - i - 1) / 2;  // calculates the offset into distances
#endif
    for (j=i+1; j<numref; j++) {
      dx[0] = ref[j][0] - ref[i][0];
      dx[1] = ref[j][1] - ref[i][1];
      dx[2] = ref[j][2] - ref[i][2];
      rsq = (dx[0]*dx[0]) + (dx[1]*dx[1]) + (dx[2]*dx[2]);
      *(distances+distpos) = sqrt(rsq);
      distpos += 1;
    }
  }
}

static void _calc_self_distance_array_ortho(coordinate* ref, int numref,
                                            float* box, double* distances)
{
  int i, j, distpos;
  double dx[3];
  float inverse_box[3];
  double rsq;

  inverse_box[0] = 1.0 / box[0];
  inverse_box[1] = 1.0 / box[1];
  inverse_box[2] = 1.0 / box[2];
  distpos = 0;

#ifdef PARALLEL
#pragma omp parallel for private(i, distpos, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++) {
#ifdef PARALLEL
    distpos = i * (2 * numref - i - 1) / 2;  // calculates the offset into distances
#endif
    for (j=i+1; j<numref; j++) {
      dx[0] = ref[j][0] - ref[i][0];
      dx[1] = ref[j][1] - ref[i][1];
      dx[2] = ref[j][2] - ref[i][2];
      // Periodic boundaries
      minimum_image(dx, box, inverse_box);
      rsq = (dx[0]*dx[0]) + (dx[1]*dx[1]) + (dx[2]*dx[2]);
      *(distances+distpos) = sqrt(rsq);
      distpos += 1;
    }
  }
}

static void _calc_self_distance_array_triclinic(coordinate* ref, int numref,
                                                float* box, double *distances)
{
  int i, j, distpos;
  double dx[3];
  double rsq;

  _triclinic_pbc(ref, numref, box);

  distpos = 0;

#ifdef PARALLEL
#pragma omp parallel for private(i, distpos, j, dx, rsq) shared(distances)
#endif
  for (i=0; i<numref; i++){
#ifdef PARALLEL
    distpos = i * (2 * numref - i - 1) / 2;  // calculates the offset into distances
#endif
    for (j=i+1; j<numref; j++){
      dx[0] = ref[j][0] - ref[i][0];
      dx[1] = ref[j][1] - ref[i][1];
      dx[2] = ref[j][2] - ref[i][2];
      minimum_image_triclinic(dx, box);
      rsq = (dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
      *(distances + distpos) = sqrt(rsq);
      distpos += 1;
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
        default:
            break;
    }

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
                        minimum_image_triclinic(dx, box);
                        break;
                    default:
                        break;
                }
                double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if ((r2 >= r2_min) && (r2 <= r2_max)) {
                    int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                    if (k >= 0) {
                        local_histo[k] += 1;
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
        default:
            break;
    }
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
                        minimum_image_triclinic(dx, box);
                        break;
                    default:
                        break;
                }
                double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if ((r2 >= r2_min) && (r2 <= r2_max)) {
                    int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                    if (k >= 0) {
                        local_histo[k] += 1;
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

void _coord_transform(coordinate* coords, int numCoords, double* box)
{
  int i, j, k;
  float newpos[3];
  // Matrix multiplication inCoords * box = outCoords
  // Multiplication done in place using temp array 'new'
  // Used to transform coordinates to/from S/R space in trilinic boxes
#ifdef PARALLEL
#pragma omp parallel for private(i, j, k, newpos) shared(coords)
#endif
  for (i=0; i < numCoords; i++){
    newpos[0] = 0.0;
    newpos[1] = 0.0;
    newpos[2] = 0.0;
    for (j=0; j<3; j++){
      for (k=0; k<3; k++){
        newpos[j] += coords[i][k] * box[3 * k + j];
      }
    }
    coords[i][0] = newpos[0];
    coords[i][1] = newpos[1];
    coords[i][2] = newpos[2];
  }
}

static void _calc_bond_distance(coordinate* atom1, coordinate* atom2,
                                int numatom, double* distances)
{
  int i;
  double dx[3];
  double rsq;

#ifdef PARALLEL
#pragma omp parallel for private(i, dx, rsq) shared(distances)
#endif
  for (i=0; i<numatom; i++) {
    dx[0] = atom1[i][0] - atom2[i][0];
    dx[1] = atom1[i][1] - atom2[i][1];
    dx[2] = atom1[i][2] - atom2[i][2];
    rsq = (dx[0]*dx[0])+(dx[1]*dx[1])+(dx[2]*dx[2]);
    *(distances+i) = sqrt(rsq);
  }
}

static void _calc_bond_distance_ortho(coordinate* atom1, coordinate* atom2,
                                      int numatom, float* box, double* distances)
{
  int i;
  double dx[3];
  float inverse_box[3];
  double rsq;

  inverse_box[0] = 1.0/box[0];
  inverse_box[1] = 1.0/box[1];
  inverse_box[2] = 1.0/box[2];

#ifdef PARALLEL
#pragma omp parallel for private(i, dx, rsq) shared(distances)
#endif
  for (i=0; i<numatom; i++) {
    dx[0] = atom1[i][0] - atom2[i][0];
    dx[1] = atom1[i][1] - atom2[i][1];
    dx[2] = atom1[i][2] - atom2[i][2];
    // PBC time!
    minimum_image(dx, box, inverse_box);
    rsq = (dx[0]*dx[0])+(dx[1]*dx[1])+(dx[2]*dx[2]);
    *(distances+i) = sqrt(rsq);
  }
}
static void _calc_bond_distance_triclinic(coordinate* atom1, coordinate* atom2,
                                          int numatom, float* box,
                                          double* distances)
{
  int i;
  double dx[3];
  double rsq;

  _triclinic_pbc(atom1, numatom, box);
  _triclinic_pbc(atom2, numatom, box);

#ifdef PARALLEL
#pragma omp parallel for private(i, dx, rsq) shared(distances)
#endif
  for (i=0; i<numatom; i++) {
    dx[0] = atom1[i][0] - atom2[i][0];
    dx[1] = atom1[i][1] - atom2[i][1];
    dx[2] = atom1[i][2] - atom2[i][2];
    // PBC time!
    minimum_image_triclinic(dx, box);
    rsq = (dx[0]*dx[0])+(dx[1]*dx[1])+(dx[2]*dx[2]);
    *(distances+i) = sqrt(rsq);
  }
}

static void _calc_angle(coordinate* atom1, coordinate* atom2,
                        coordinate* atom3, int numatom, double* angles)
{
  int i;
  double rji[3], rjk[3];
  double x, y, xp[3];

#ifdef PARALLEL
#pragma omp parallel for private(i, rji, rjk, x, xp, y) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    rji[0] = atom1[i][0] - atom2[i][0];
    rji[1] = atom1[i][1] - atom2[i][1];
    rji[2] = atom1[i][2] - atom2[i][2];

    rjk[0] = atom3[i][0] - atom2[i][0];
    rjk[1] = atom3[i][1] - atom2[i][1];
    rjk[2] = atom3[i][2] - atom2[i][2];

    x = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];

    xp[0] = rji[1]*rjk[2] - rji[2]*rjk[1];
    xp[1] =-rji[0]*rjk[2] + rji[2]*rjk[0];
    xp[2] = rji[0]*rjk[1] - rji[1]*rjk[0];

    y = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2]);

    *(angles+i) = atan2(y,x);
  }
}

static void _calc_angle_ortho(coordinate* atom1, coordinate* atom2,
                              coordinate* atom3, int numatom,
                              float* box, double* angles)
{
  // Angle is calculated between two vectors
  // pbc option ensures that vectors are constructed between atoms in the same image as eachother
  // ie that vectors don't go across a boxlength
  // it doesn't matter if vectors are from different boxes however
  int i;
  double rji[3], rjk[3];
  double x, y, xp[3];
  float inverse_box[3];

  inverse_box[0] = 1.0/box[0];
  inverse_box[1] = 1.0/box[1];
  inverse_box[2] = 1.0/box[2];

#ifdef PARALLEL
#pragma omp parallel for private(i, rji, rjk, x, xp, y) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    rji[0] = atom1[i][0] - atom2[i][0];
    rji[1] = atom1[i][1] - atom2[i][1];
    rji[2] = atom1[i][2] - atom2[i][2];
    minimum_image(rji, box, inverse_box);

    rjk[0] = atom3[i][0] - atom2[i][0];
    rjk[1] = atom3[i][1] - atom2[i][1];
    rjk[2] = atom3[i][2] - atom2[i][2];
    minimum_image(rjk, box, inverse_box);

    x = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];

    xp[0] = rji[1]*rjk[2] - rji[2]*rjk[1];
    xp[1] =-rji[0]*rjk[2] + rji[2]*rjk[0];
    xp[2] = rji[0]*rjk[1] - rji[1]*rjk[0];

    y = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2]);

    *(angles+i) = atan2(y,x);
  }
}

static void _calc_angle_triclinic(coordinate* atom1, coordinate* atom2,
                                  coordinate* atom3, int numatom,
                                  float* box, double* angles)
{
  // Triclinic version of min image aware angle calculate, see above
  int i;
  double rji[3], rjk[3];
  double x, y, xp[3];

  _triclinic_pbc(atom1, numatom, box);
  _triclinic_pbc(atom2, numatom, box);
  _triclinic_pbc(atom3, numatom, box);

#ifdef PARALLEL
#pragma omp parallel for private(i, rji, rjk, x, xp, y) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    rji[0] = atom1[i][0] - atom2[i][0];
    rji[1] = atom1[i][1] - atom2[i][1];
    rji[2] = atom1[i][2] - atom2[i][2];
    minimum_image_triclinic(rji, box);

    rjk[0] = atom3[i][0] - atom2[i][0];
    rjk[1] = atom3[i][1] - atom2[i][1];
    rjk[2] = atom3[i][2] - atom2[i][2];
    minimum_image_triclinic(rjk, box);

    x = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];

    xp[0] = rji[1]*rjk[2] - rji[2]*rjk[1];
    xp[1] =-rji[0]*rjk[2] + rji[2]*rjk[0];
    xp[2] = rji[0]*rjk[1] - rji[1]*rjk[0];

    y = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2]);

    *(angles+i) = atan2(y,x);
  }
}

static void _calc_dihedral_angle(double* va, double* vb, double* vc, double* result)
{
  // Returns atan2 from vectors va, vb, vc
  double n1[3], n2[3];
  double xp[3], vb_norm;
  double x, y;

  //n1 is normal vector to -va, vb
  //n2 is normal vector to -vb, vc
  n1[0] =-va[1]*vb[2] + va[2]*vb[1];
  n1[1] = va[0]*vb[2] - va[2]*vb[0];
  n1[2] =-va[0]*vb[1] + va[1]*vb[0];

  n2[0] =-vb[1]*vc[2] + vb[2]*vc[1];
  n2[1] = vb[0]*vc[2] - vb[2]*vc[0];
  n2[2] =-vb[0]*vc[1] + vb[1]*vc[0];

  // x = dot(n1,n2) = cos theta
  x = (n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]);

  // xp = cross(n1,n2)
  xp[0] = n1[1]*n2[2] - n1[2]*n2[1];
  xp[1] =-n1[0]*n2[2] + n1[2]*n2[0];
  xp[2] = n1[0]*n2[1] - n1[1]*n2[0];

  vb_norm = sqrt(vb[0]*vb[0] + vb[1]*vb[1] + vb[2]*vb[2]);

  y = (xp[0]*vb[0] + xp[1]*vb[1] + xp[2]*vb[2]) / vb_norm;

  if ( (fabs(x) == 0.0) && (fabs(y) == 0.0) ) // numpy consistency
  {
    *result = NAN;
    return;
  }

  *result = atan2(y, x); //atan2 is better conditioned than acos
}

static void _calc_dihedral(coordinate* atom1, coordinate* atom2,
                           coordinate* atom3, coordinate* atom4,
                           int numatom, double* angles)
{
  int i;
  double va[3], vb[3], vc[3];

#ifdef PARALLEL
#pragma omp parallel for private(i, va, vb, vc) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    // connecting vectors between all 4 atoms: 1 -va-> 2 -vb-> 3 -vc-> 4
    va[0] = atom2[i][0] - atom1[i][0];
    va[1] = atom2[i][1] - atom1[i][1];
    va[2] = atom2[i][2] - atom1[i][2];

    vb[0] = atom3[i][0] - atom2[i][0];
    vb[1] = atom3[i][1] - atom2[i][1];
    vb[2] = atom3[i][2] - atom2[i][2];

    vc[0] = atom4[i][0] - atom3[i][0];
    vc[1] = atom4[i][1] - atom3[i][1];
    vc[2] = atom4[i][2] - atom3[i][2];

    _calc_dihedral_angle(va, vb, vc, angles + i);
  }
}

static void _calc_dihedral_ortho(coordinate* atom1, coordinate* atom2,
                                 coordinate* atom3, coordinate* atom4,
                                 int numatom, float* box, double* angles)
{
  int i;
  double va[3], vb[3], vc[3];
  float inverse_box[3];

  inverse_box[0] = 1.0/box[0];
  inverse_box[1] = 1.0/box[1];
  inverse_box[2] = 1.0/box[2];

#ifdef PARALLEL
#pragma omp parallel for private(i, va, vb, vc) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    // connecting vectors between all 4 atoms: 1 -va-> 2 -vb-> 3 -vc-> 4
    va[0] = atom2[i][0] - atom1[i][0];
    va[1] = atom2[i][1] - atom1[i][1];
    va[2] = atom2[i][2] - atom1[i][2];
    minimum_image(va, box, inverse_box);

    vb[0] = atom3[i][0] - atom2[i][0];
    vb[1] = atom3[i][1] - atom2[i][1];
    vb[2] = atom3[i][2] - atom2[i][2];
    minimum_image(vb, box, inverse_box);

    vc[0] = atom4[i][0] - atom3[i][0];
    vc[1] = atom4[i][1] - atom3[i][1];
    vc[2] = atom4[i][2] - atom3[i][2];
    minimum_image(vc, box, inverse_box);

    _calc_dihedral_angle(va, vb, vc, angles + i);
  }
}

static void _calc_dihedral_triclinic(coordinate* atom1, coordinate* atom2,
                                     coordinate* atom3, coordinate* atom4,
                                     int numatom, float* box, double* angles)
{
  int i;
  double va[3], vb[3], vc[3];

  _triclinic_pbc(atom1, numatom, box);
  _triclinic_pbc(atom2, numatom, box);
  _triclinic_pbc(atom3, numatom, box);
  _triclinic_pbc(atom4, numatom, box);

#ifdef PARALLEL
#pragma omp parallel for private(i, va, vb, vc) shared(angles)
#endif
  for (i=0; i<numatom; i++) {
    // connecting vectors between all 4 atoms: 1 -va-> 2 -vb-> 3 -vc-> 4
    va[0] = atom2[i][0] - atom1[i][0];
    va[1] = atom2[i][1] - atom1[i][1];
    va[2] = atom2[i][2] - atom1[i][2];
    minimum_image_triclinic(va, box);

    vb[0] = atom3[i][0] - atom2[i][0];
    vb[1] = atom3[i][1] - atom2[i][1];
    vb[2] = atom3[i][2] - atom2[i][2];
    minimum_image_triclinic(vb, box);

    vc[0] = atom4[i][0] - atom3[i][0];
    vc[1] = atom4[i][1] - atom3[i][1];
    vc[2] = atom4[i][2] - atom3[i][2];
    minimum_image_triclinic(vc, box);

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
static inline float* _get_coords_in_blocks(const coordinate* __restrict__ coords,
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
static void _ortho_pbc_vectorized(float* __restrict__ coords, int numcoords,
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
static void _triclinic_pbc_vectorized(float* __restrict__ coords, int numcoords,
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
static inline void _calc_self_distance_vectors_block(double* __restrict__ dxs,
                                                     const float* __restrict__ refs)
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
static inline void _calc_distance_vectors_block(double* __restrict__ dxs,
                                                const float* __restrict__ refs,
                                                const float* __restrict__ confs)
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
static inline void _minimum_image_ortho_lazy_block(double* __restrict__ dxs,
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
static inline void _minimum_image_triclinic_lazy_block(double* __restrict__ dxs,
                                                       const float* box_vectors,
                                                       double* __restrict__ aux)
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
static inline void _calc_squared_distances_block(double* __restrict__ r2s,
                                                 const double* __restrict__ dxs)
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

static void _calc_distance_array_vectorized(const coordinate* __restrict__ ref,
                                            int numref,
                                            const coordinate* __restrict__ conf,
                                            int numconf,
                                            const float* box, ePBC pbc_type,
                                            double* __restrict__ distances)
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
        default:
            break;
    }

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
                }
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
                }
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
                }
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
                }
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

static void _calc_self_distance_array_vectorized(const coordinate* __restrict__ ref,
                                                 int numref,
                                                 const float* box,
                                                 ePBC pbc_type,
                                                 double* __restrict__ distances)
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
        default:
            break;
    }
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
            }
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
                }
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
                }
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
                }
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

static void _calc_distance_histogram_vectorized(const coordinate* __restrict__ ref,
                                                int numref,
                                                const coordinate* __restrict__ conf,
                                                int numconf,
                                                const float* box, ePBC pbc_type,
                                                double rmin, double rmax,
                                                histbin* __restrict__ histo,
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
        default:
            break;
    }

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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    if ((r2s[i] >= r2_min) && (r2s[i] <= r2_max)) {
                        int k = (int) ((sqrt(r2s[i]) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < _BLOCKSIZE; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size_ref; i++) {
                    for (int j = 0; j < partial_block_size_conf; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k] += 1;
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
                                                     __restrict__ ref, int numref,
                                                     const float* box,
                                                     ePBC pbc_type,
                                                     double rmin, double rmax,
                                                     histbin* __restrict__ histo,
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
        default:
            break;
    }

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
            }
            _calc_squared_distances_block(r2s, dxs);
            for (int i = 0; i < _BLOCKSIZE - 1; i++) {
                for (int j = i + 1; j < _BLOCKSIZE; j++) {
                    double r2 = r2s[i * _BLOCKSIZE + j];
                    if ((r2 >= r2_min) && (r2 <= r2_max)) {
                        int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE_2; i++) {
                    if ((r2s[i] >= r2_min) && (r2s[i] <= r2_max)) {
                        int k = (int) ((sqrt(r2s[i]) - rmin) * inverse_binw);
                        if (k >= 0) {
                            local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < _BLOCKSIZE; i++) {
                    for (int j = 0; j < partial_block_size; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k] += 1;
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
                }
                _calc_squared_distances_block(r2s, dxs);
                for (int i = 0; i < partial_block_size - 1; i++) {
                    for (int j = i + 1; j < partial_block_size; j++) {
                        double r2 = r2s[i * _BLOCKSIZE + j];
                        if ((r2 >= r2_min) && (r2 <= r2_max)) {
                            int k = (int) ((sqrt(r2) - rmin) * inverse_binw);
                            if (k >= 0) {
                                local_histo[k] += 1;
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
#endif /* __DISTANCES_H */
