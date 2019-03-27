# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
#

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: embedsignature=False

"""
Parallel distance calculation library --- :mod:`MDAnalysis.lib.c_distances_openmp`
==================================================================================


Contains OpenMP versions of the contents of "calc_distances.h"
"""

cdef extern from "string.h":
    void* memcpy(void* dst, void* src, int len)

cdef extern from "calc_distances.h":
    ctypedef float coordinate[3]
    cdef bint USED_OPENMP
    void _calc_distance_array(coordinate* ref, int numref, coordinate* conf, int numconf, double* distances)
    void _calc_distance_array_ortho(coordinate* ref, int numref, coordinate* conf, int numconf, float* box, double* distances)
    void _calc_distance_array_triclinic(coordinate* ref, int numref, coordinate* conf, int numconf, float* box, double* distances)
    void _calc_self_distance_array(coordinate* ref, int numref, double* distances)
    void _calc_self_distance_array_ortho(coordinate* ref, int numref, float* box, double* distances)
    void _calc_self_distance_array_triclinic(coordinate* ref, int numref, float* box, double* distances)
    void _coord_transform(coordinate* coords, int numCoords, double* box)
    void _calc_bond_distance(coordinate* atom1, coordinate* atom2, int numatom, double* distances)
    void _calc_bond_distance_ortho(coordinate* atom1, coordinate* atom2, int numatom, float* box, double* distances)
    void _calc_bond_distance_triclinic(coordinate* atom1, coordinate* atom2, int numatom, float* box, double* distances)
    void _calc_angle(coordinate* atom1, coordinate* atom2, coordinate* atom3, int numatom, double* angles)
    void _calc_angle_ortho(coordinate* atom1, coordinate* atom2, coordinate* atom3, int numatom, float* box, double* angles)
    void _calc_angle_triclinic(coordinate* atom1, coordinate* atom2, coordinate* atom3, int numatom, float* box, double* angles)
    void _calc_dihedral(coordinate* atom1, coordinate* atom2, coordinate* atom3, coordinate* atom4, int numatom, double* angles)
    void _calc_dihedral_ortho(coordinate* atom1, coordinate* atom2, coordinate* atom3, coordinate* atom4, int numatom, float* box, double* angles)
    void _calc_dihedral_triclinic(coordinate* atom1, coordinate* atom2, coordinate* atom3, coordinate* atom4, int numatom, float* box, double* angles)
    void _ortho_pbc(coordinate* coords, int numcoords, float* box)
    void _triclinic_pbc(coordinate* coords, int numcoords, float* box)


OPENMP_ENABLED = True if USED_OPENMP else False

def calc_distance_array(float[:, ::1] ref, float[:, ::1] conf,
                        double[:, ::1] result):
    cdef int confnum, refnum
    confnum = conf.shape[0]
    refnum = ref.shape[0]

    _calc_distance_array(<coordinate*> &ref[0, 0], refnum,
                         <coordinate*> &conf[0, 0], confnum, &result[0, 0])

def calc_distance_array_ortho(float[:, ::1] ref, float[:, ::1] conf,
                              float[::1] box, double[:, ::1] result):
    cdef int confnum, refnum
    confnum = conf.shape[0]
    refnum = ref.shape[0]

    _calc_distance_array_ortho(<coordinate*> &ref[0, 0], refnum,
                               <coordinate*> &conf[0, 0], confnum, &box[0],
                               &result[0, 0])

def calc_distance_array_triclinic(float[:, ::1] ref, float[:, ::1] conf,
                                  float[:, ::1] box, double[:, ::1] result):
    cdef int confnum, refnum
    confnum = conf.shape[0]
    refnum = ref.shape[0]

    _calc_distance_array_triclinic(<coordinate*> &ref[0, 0], refnum,
                                   <coordinate*> &conf[0, 0], confnum,
                                   &box[0, 0], &result[0, 0])

def calc_self_distance_array(float[:, ::1] ref, double[::1] result):
    cdef int refnum
    refnum = ref.shape[0]

    _calc_self_distance_array(<coordinate*> &ref[0, 0], refnum, &result[0])

def calc_self_distance_array_ortho(float[:, ::1] ref, float[::1] box,
                                   double[::1] result):
    cdef int refnum
    refnum = ref.shape[0]

    _calc_self_distance_array_ortho(<coordinate*> &ref[0, 0], refnum, &box[0],
                                    &result[0])

def calc_self_distance_array_triclinic(float[:, ::1] ref, float[:, ::1] box,
                                       double[::1] result):
    cdef int refnum
    refnum = ref.shape[0]

    _calc_self_distance_array_triclinic(<coordinate*> &ref[0, 0], refnum,
                                        &box[0, 0], &result[0])

def coord_transform(float[:, ::1] coords, double[:, ::1] box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _coord_transform(<coordinate*> &coords[0, 0], numcoords, &box[0, 0])

def calc_bond_distance(float[:, ::1] coords1, float[:, ::1] coords2,
                       double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_bond_distance(<coordinate*> &coords1[0, 0],
                        <coordinate*> &coords2[0, 0], numcoords, &results[0])

def calc_bond_distance_ortho(float[:, ::1] coords1, float[:, ::1] coords2,
                             float[::1] box, double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_bond_distance_ortho(<coordinate*> &coords1[0, 0],
                              <coordinate*> &coords2[0, 0], numcoords, &box[0],
                              &results[0])

def calc_bond_distance_triclinic(float[:, ::1] coords1, float[:, ::1] coords2,
                                 float[:, ::1] box, double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_bond_distance_triclinic(<coordinate*> &coords1[0, 0],
                                  <coordinate*> &coords2[0, 0], numcoords,
                                  &box[0, 0], &results[0])

def calc_angle(float[:, ::1] coords1, float[:, ::1] coords2,
               float[:, ::1] coords3, double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_angle(<coordinate*> &coords1[0, 0], <coordinate*> &coords2[0, 0],
                <coordinate*> &coords3[0, 0], numcoords, &results[0])

def calc_angle_ortho(float[:, ::1] coords1, float[:, ::1] coords2,
                     float[:, ::1] coords3, float[::1] box,
                     double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_angle_ortho(<coordinate*> &coords1[0, 0],
                      <coordinate*> &coords2[0, 0],
                      <coordinate*> &coords3[0, 0], numcoords, &box[0],
                      &results[0])

def calc_angle_triclinic(float[:, ::1] coords1, float[:, ::1] coords2,
                         float[:, ::1] coords3, float[:, ::1] box,
                         double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_angle_triclinic(<coordinate*> &coords1[0, 0],
                          <coordinate*> &coords2[0, 0],
                          <coordinate*> &coords3[0, 0], numcoords,
                          &box[0, 0], &results[0])

def calc_dihedral(float[:, ::1] coords1, float[:, ::1] coords2,
                  float[:, ::1] coords3, float[:, ::1] coords4,
                  double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_dihedral(<coordinate*> &coords1[0, 0], <coordinate*> &coords2[0, 0],
                   <coordinate*> &coords3[0, 0], <coordinate*> &coords4[0, 0],
                   numcoords, &results[0])

def calc_dihedral_ortho(float[:, ::1] coords1, float[:, ::1] coords2,
                        float[:, ::1] coords3, float[:, ::1] coords4,
                        float[::1] box, double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_dihedral_ortho(<coordinate*> &coords1[0, 0],
                         <coordinate*> &coords2[0, 0],
                         <coordinate*> &coords3[0, 0],
                         <coordinate*> &coords4[0, 0], numcoords, &box[0],
                         &results[0])

def calc_dihedral_triclinic(float[:, ::1] coords1, float[:, ::1] coords2,
                            float[:, ::1] coords3, float[:, ::1] coords4,
                            float[:, ::1] box, double[::1] results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_dihedral_triclinic(<coordinate*> &coords1[0, 0],
                             <coordinate*> &coords2[0, 0],
                             <coordinate*> &coords3[0, 0],
                             <coordinate*> &coords4[0, 0], numcoords,
                             &box[0, 0], &results[0])

def ortho_pbc(float[:, ::1] coords, float[::1] box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _ortho_pbc(<coordinate*> &coords[0, 0], numcoords, &box[0])

def triclinic_pbc(float[:, ::1] coords, float[:, ::1] box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _triclinic_pbc(<coordinate*> &coords[0, 0], numcoords, &box[0, 0])
