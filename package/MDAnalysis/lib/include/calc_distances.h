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

#include <assert.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <stdlib.h>

typedef float coordinate[3];
typedef int64_t histbin;

typedef enum ePBC {
    PBCortho,
    PBCtriclinic,
    PBCnone,
    PBCunknown
} ePBC;

#ifdef PARALLEL
    #include <omp.h>
    #define USED_OPENMP 1
#else
    #define USED_OPENMP 0
#endif

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
    box_inverse[0] = 1.0 / box[0];
    box_inverse[1] = 1.0 / box[1];
    box_inverse[2] = 1.0 / box[2];
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
    float bi0 = 1.0 / box_vectors[0];
    float bi4 = 1.0 / box_vectors[4];
    float bi8 = 1.0 / box_vectors[8];
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
#endif /* __DISTANCES_H */
