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
#include "calc_distances.h"

/*
 * With AVX, floorf() vectorizes nicely but without AVX, integer casting
 * and subtracting the sign is a lot faster. this function will always yield
 * the faster implementation.
 */
static inline int _floorf(float x) {
#ifdef __AVX__
            return (int) floorf(x);
#else
            return (int) x - (x < 0);
#endif
}

void _ortho_pbc(coordinate* restrict coords, size_t numcoords,
                float* restrict box, float* restrict box_inverse)
{
    /*
     * The coords array is not guaranteed to be memory-aligned, so we divide it
     * into three regions:
     *   1. One region extending from the beginning up to (but not including)
     *      the first memory-aligned coordinate
     *   2. One region starting at the first aligned coordinate and comprising
     *      as many memory-aligned blocks of BLOCKSIZE coordinates as possible
     *   3. One region containing the last partial block at the end of the
     *      coords array
     */
    // compute end of first region (nbefore):
    int64_t nbefore = first_aligned_index((void*) coords, (size_t) numcoords,
                                          sizeof(coordinate));
    int64_t nblocks = 0L;
    int64_t nblocked = 0L;
    int64_t nremaining = 0L;
    // make sure the end of the first region is bound by the number of
    // coordinates:
    if (nbefore < numcoords) {
        // compute the number of blocks in the second region:
        nremaining = numcoords - nbefore;
        nblocks = nremaining / BLOCKSIZE;
        // and the number of coordinates in the third region:
        nblocked = nblocks * BLOCKSIZE;
        nremaining -= nblocked;
        // if there are no blocks in the second region, add the third to the
        // first region instead:
        if (nblocks == 0L) {
            nbefore += nremaining;
            nremaining = 0L;
        }
    }
#ifdef PARALLEL
    // parallel code is only faster for a large number of coordinates
    #pragma omp parallel if(numcoords > ORTHO_PBC_PARALLEL_THRESHOLD)
#endif
    {
        // Have each thread prepare memory-aligned auxiliary arrays with enough
        // elements to work with one coordinate block:
        float _memaligned _box[3*BLOCKSIZE] _attaligned;
        float _memaligned _box_inverse[3*BLOCKSIZE] _attaligned;
        float* restrict _coords = (float*) coords;
        int64_t i, n;
        for(i=0; i<BLOCKSIZE; ++i) {
            memcpy(&_box[3*i], box, 3*sizeof(float));
            memcpy(&_box_inverse[3*i], box_inverse, 3*sizeof(float));
        }

        // loop over first region
#ifdef PARALLEL
        #pragma omp single nowait  // never worth parallelizing
#endif
        for (i=0; i < 3*nbefore; i++) {
            _coords[i] -= _floorf(_coords[i] * _box_inverse[i]) * _box[i];
        }

        // loop over aligned blocks in second region
#ifdef PARALLEL
        #pragma omp for nowait
#endif
        for (n=0; n<nblocks; n++) {
            _coords = _assaligned((float*) (coords + nbefore + n * BLOCKSIZE));
            for (i=0; i<3*BLOCKSIZE; i++) {  // this loop vectorizes nicely
                _coords[i] -= _floorf(_coords[i] * _box_inverse[i]) * _box[i];
            }
        }

        // loop over third region
        _coords = _assaligned((float*) (coords + nbefore + nblocked));
#ifdef PARALLEL
        #pragma omp single nowait  // never worth parallelizing
#endif
        for (i=0; i < 3*nremaining; i++) {
            _coords[i] -= _floorf(_coords[i] * _box_inverse[i]) * _box[i];
        }
    }
}


void _triclinic_pbc(coordinate* coords, int numcoords, coordinate* box, float* box_inverse)
{
  int i, s;
  // Inverse of matrix box (here called "m")
  //   [           1/m00                 ,        0      ,   0  ]
  //   [        -m10/(m00*m11)           ,      1/m11    ,   0  ]
  //   [(m10*m21/(m00*m11) - m20/m00)/m22, -m21/(m11*m22), 1/m22]
  float bi00 = box_inverse[0];
  float bi11 = box_inverse[1];
  float bi22 = box_inverse[2];
  float bi01 = -box[1][0]*bi00*bi11;
  float bi02 = (box[1][0]*box[2][1]*bi11 - box[2][0])*bi00*bi22;
  float bi12 = -box[2][1]*bi11*bi22;
  // Moves all coordinates to within the box boundaries for a triclinic box
  // Assumes box having zero values for box[0][1], box[0][2] and box [1][2]
#ifdef PARALLEL
#pragma omp parallel for private(i, s) shared(coords)
#endif
  for (i=0; i < numcoords; i++){
    // translate coords[i] to central cell along c-axis
    s = floor(coords[i][2]*bi22);
    coords[i][2] -= s * box[2][2];
    coords[i][1] -= s * box[2][1];
    coords[i][0] -= s * box[2][0];
    // translate remainder of coords[i] to central cell along b-axis
    s = floor(coords[i][1]*bi11 + coords[i][2]*bi12);
    coords[i][1] -= s * box[1][1];
    coords[i][0] -= s * box[1][0];
    // translate remainder of coords[i] to central cell along a-axis
    s = floor(coords[i][0]*bi00 + coords[i][1]*bi01 + coords[i][2]*bi02);
    coords[i][0] -= s * box[0][0];
  }
}

void _calc_distance_array(coordinate* ref, int numref, coordinate* conf,
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

void _calc_distance_array_ortho(coordinate* ref, int numref, coordinate* conf,
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

void _calc_distance_array_triclinic(coordinate* ref, int numref,
                                           coordinate* conf, int numconf,
                                           coordinate* box, double* distances)
{
  int i, j;
  double dx[3];
  float box_inverse[3];
  double rsq;

  box_inverse[0] = 1.0 / box[0][0];
  box_inverse[1] = 1.0 / box[1][1];
  box_inverse[2] = 1.0 / box[2][2];
  // Move coords to inside box
  _triclinic_pbc(ref, numref, box, box_inverse);
  _triclinic_pbc(conf, numconf, box, box_inverse);

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

void _calc_self_distance_array(coordinate* ref, int numref, double* distances)
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

void _calc_self_distance_array_ortho(coordinate* ref, int numref, float* box,
                                            double* distances)
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

void _calc_self_distance_array_triclinic(coordinate* ref, int numref,
                                                coordinate* box, double *distances)
{
  int i, j, distpos;
  double dx[3];
  double rsq;
  float box_inverse[3];

  box_inverse[0] = 1.0 / box[0][0];
  box_inverse[1] = 1.0 / box[1][1];
  box_inverse[2] = 1.0 / box[2][2];

  _triclinic_pbc(ref, numref, box, box_inverse);

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

void _coord_transform(coordinate* coords, int numCoords, coordinate* box)
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
        newpos[j] += coords[i][k] * box[k][j];
      }
    }
    coords[i][0] = newpos[0];
    coords[i][1] = newpos[1];
    coords[i][2] = newpos[2];
  }
}

void _calc_bond_distance(coordinate* atom1, coordinate* atom2,
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

void _calc_bond_distance_ortho(coordinate* atom1, coordinate* atom2,
                                      int numatom, float* box, double* distances)
{
  int i;
  double dx[3];
  float inverse_box[3];
  double rsq;

  inverse_box[0] = 1.0 / box[0];
  inverse_box[1] = 1.0 / box[1];
  inverse_box[2] = 1.0 / box[2];

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
void _calc_bond_distance_triclinic(coordinate* atom1, coordinate* atom2,
                                          int numatom, coordinate* box,
                                          double* distances)
{
  int i;
  double dx[3];
  float box_inverse[3];
  double rsq;

  box_inverse[0] = 1.0/box[0][0];
  box_inverse[1] = 1.0/box[1][1];
  box_inverse[2] = 1.0/box[2][2];

  _triclinic_pbc(atom1, numatom, box, box_inverse);
  _triclinic_pbc(atom2, numatom, box, box_inverse);

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

void _calc_angle(coordinate* atom1, coordinate* atom2,
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

void _calc_angle_ortho(coordinate* atom1, coordinate* atom2,
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

  inverse_box[0] = 1.0 / box[0];
  inverse_box[1] = 1.0 / box[1];
  inverse_box[2] = 1.0 / box[2];

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

void _calc_angle_triclinic(coordinate* atom1, coordinate* atom2,
                                  coordinate* atom3, int numatom,
                                  coordinate* box, double* angles)
{
  // Triclinic version of min image aware angle calculate, see above
  int i;
  double rji[3], rjk[3];
  double x, y, xp[3];
  float box_inverse[3];

  box_inverse[0] = 1.0/box[0][0];
  box_inverse[1] = 1.0/box[1][1];
  box_inverse[2] = 1.0/box[2][2];

  _triclinic_pbc(atom1, numatom, box, box_inverse);
  _triclinic_pbc(atom2, numatom, box, box_inverse);
  _triclinic_pbc(atom3, numatom, box, box_inverse);

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

void _calc_dihedral_angle(double* va, double* vb, double* vc, double* result)
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

void _calc_dihedral(coordinate* atom1, coordinate* atom2,
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

void _calc_dihedral_ortho(coordinate* atom1, coordinate* atom2,
                                 coordinate* atom3, coordinate* atom4,
                                 int numatom, float* box, double* angles)
{
  int i;
  double va[3], vb[3], vc[3];
  float inverse_box[3];

  inverse_box[0] = 1.0 / box[0];
  inverse_box[1] = 1.0 / box[1];
  inverse_box[2] = 1.0 / box[2];

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

void _calc_dihedral_triclinic(coordinate* atom1, coordinate* atom2,
                                     coordinate* atom3, coordinate* atom4,
                                     int numatom, coordinate* box, double* angles)
{
  int i;
  double va[3], vb[3], vc[3];
  float box_inverse[3];

  box_inverse[0] = 1.0/box[0][0];
  box_inverse[1] = 1.0/box[1][1];
  box_inverse[2] = 1.0/box[2][2];

  _triclinic_pbc(atom1, numatom, box, box_inverse);
  _triclinic_pbc(atom2, numatom, box, box_inverse);
  _triclinic_pbc(atom3, numatom, box, box_inverse);
  _triclinic_pbc(atom4, numatom, box, box_inverse);

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
