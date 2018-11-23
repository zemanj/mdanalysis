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
 #ifndef __CALC_DISTANCES_H
 #define __CALC_DISTANCES_H
 
#include "memalign.h"
#include "typedefs.h"

#include <math.h>
#include <float.h>

#ifdef PARALLEL
  #include <omp.h>
  #define USED_OPENMP 1
#else
  #define USED_OPENMP 0
#endif

// Process coordinates in blocks of 32 to support auto-vectorization
#define BLOCKSIZE 32

// Assert that BLOCKSIZE * sizeof(float) is divisible by MEMORY_ALIGNMENT:
static_assert(!(BLOCKSIZE * sizeof(float) % MEMORY_ALIGNMENT), \
"BLOCKSIZE * sizeof(float) is not an integer multiple of MEMORY_ALIGNMENT!");

// parallel code is only faster for a large number of coordinates (~5000)
#define ORTHO_PBC_PARALLEL_THRESHOLD 5000

#ifdef __cplusplus
extern "C" {
#endif

static inline void minimum_image(double *dx,
                                 float *box,
                                 float *inverse_box)
{
    int i;
    double s[3];
    for (i=0; i<3; i++) {
        if (box[i] > FLT_EPSILON) {
            s[i] = inverse_box[i] * dx[i];
            dx[i] = box[i] * (s[i] - round(s[i]));
        }
    }
}

static inline void minimum_image_triclinic(double *dx,
                                           coordinate* box)
{
  // Minimum image convention for triclinic systems, modelled after domain.cpp in LAMMPS
  // Assumes that there is a maximum separation of 1 box length (enforced in dist functions
  // by moving all particles to inside the box before calculating separations)
  // Requires a box
  // Assumes box having zero values for box[0][1], box[0][2] and box [1][2]
  double dmin[3] = {0.0}, rx[3], ry[3], rz[3];
  double min = FLT_MAX, d;
  int i, x, y, z;

  for (x = -1; x < 2; ++x) {
    rx[0] = dx[0] + box[0][0] * x;
    rx[1] = dx[1];
    rx[2] = dx[2];
    for (y = -1; y < 2; ++y) {
      ry[0] = rx[0] + box[1][0] * y;
      ry[1] = rx[1] + box[1][1] * y;
      ry[2] = rx[2];
      for (z = -1; z < 2; ++z) {
        rz[0] = ry[0] + box[2][0] * z;
        rz[1] = ry[1] + box[2][1] * z;
        rz[2] = ry[2] + box[2][2] * z;
        d = rz[0]*rz[0] + rz[1]*rz[1] + rz[2] * rz[2];
        if (d < min) {
          min = d;
          for (i=0; i<3; ++i){
            dmin[i] = rz[i];
          }
        }
      }
    }
  }
  for (i =0; i<3; ++i) {
    dx[i] = dmin[i];
  }
}

void _ortho_pbc(coordinate* coords, int numcoords, float* box,
                float* box_inverse);

void _triclinic_pbc(coordinate* coords, int numcoords, coordinate* box,
                    float* box_inverse);

void _calc_distance_array(coordinate* ref, int numref, coordinate* conf,
                          int numconf, double* distances);

void _calc_distance_array_ortho(coordinate* ref, int numref, coordinate* conf,
                                int numconf, float* box, double* distances);

void _calc_distance_array_triclinic(coordinate* ref, int numref,
                                    coordinate* conf, int numconf,
                                    coordinate* box, double* distances);

void _calc_self_distance_array(coordinate* ref, int numref, double* distances);

void _calc_self_distance_array_ortho(coordinate* ref, int numref, float* box,
                                     double* distances);

void _calc_self_distance_array_triclinic(coordinate* ref, int numref,
                                         coordinate* box, double *distances);

void _coord_transform(coordinate* coords, int numCoords, coordinate* box);

void _calc_bond_distance(coordinate* atom1, coordinate* atom2, int numatom,
                         double* distances);

void _calc_bond_distance_ortho(coordinate* atom1, coordinate* atom2,
                               int numatom, float* box, double* distances);

void _calc_bond_distance_triclinic(coordinate* atom1, coordinate* atom2,
                                   int numatom, coordinate* box,
                                   double* distances);

void _calc_angle(coordinate* atom1, coordinate* atom2, coordinate* atom3,
                 int numatom, double* angles);

void _calc_angle_ortho(coordinate* atom1, coordinate* atom2, coordinate* atom3,
                       int numatom, float* box, double* angles);

void _calc_angle_triclinic(coordinate* atom1, coordinate* atom2,
                           coordinate* atom3, int numatom, coordinate* box,
                           double* angles);

void _calc_dihedral_angle(double* va, double* vb, double* vc, double* result);

void _calc_dihedral(coordinate* atom1, coordinate* atom2, coordinate* atom3,
                    coordinate* atom4, int numatom, double* angles);

void _calc_dihedral_ortho(coordinate* atom1, coordinate* atom2,
                          coordinate* atom3, coordinate* atom4, int numatom,
                          float* box, double* angles);

void _calc_dihedral_triclinic(coordinate* atom1, coordinate* atom2,
                              coordinate* atom3, coordinate* atom4, int numatom,
                              coordinate* box, double* angles);

#ifdef __cplusplus
}
#endif

#endif /*__CALC_DISTANCES_H*/
