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
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
#

"""
Parallel distance calculation library --- :mod:`MDAnalysis.lib.c_distances_openmp`
==================================================================================

Contains OpenMP versions of the contents of "calc_distances.h"
"""

import numpy
cimport numpy
cimport openmp

from libc.stdint cimport int64_t

cdef extern from "calc_distances.h":
    ctypedef float coordinate[3]
    ctypedef int64_t histbin
    cdef bint USED_OPENMP
    cdef int _BLOCKSIZE
    cdef enum PBCenum "ePBC":
        PBCortho, PBCtriclinic, PBCnone, PBCunknown
    void _calc_distance_array(coordinate* ref, int numref, coordinate* conf,
                              int numconf, float* box, PBCenum pbc_type,
                              double* distances)
    void _calc_distance_array_vectorized(coordinate* ref, int numref,
                                         coordinate* conf, int numconf,
                                         float* box, PBCenum pbc_type,
                                         double* distances)
    void _calc_self_distance_array(coordinate* ref, int numref, float* box,
                                   PBCenum pbc_type, double* distances)
    void _calc_self_distance_array_vectorized(coordinate* ref, int numref,
                                              float* box, PBCenum pbc_type,
                                              double* distances)
    void _calc_distance_histogram(coordinate* ref, int numref, coordinate* conf,
                                  int numconf, float* box, PBCenum pbc_type,
                                  double rmin, double rmax, histbin* histo,
                                  int numhisto)
    void _calc_distance_histogram_vectorized(coordinate* ref, int numref,
                                             coordinate* conf, int numconf,
                                             float* box, PBCenum pbc_type,
                                             double rmin, double rmax,
                                             histbin* histo, int numhisto)
    void _calc_distance_histogram_grid_based(coordinate* ref, int numref,
                                             coordinate* conf, int numconf,
                                             float* box, PBCenum pbc_type,
                                             double rmin, double rmax,
                                             histbin* histo, int numhisto)
    void _calc_self_distance_histogram(coordinate* ref, int numref, float* box,
                                       PBCenum pbc_type, double rmin,
                                       double rmax, histbin* histo,
                                       int numhisto)
    void _calc_self_distance_histogram_vectorized(coordinate* ref, int numref,
                                                  float* box, PBCenum pbc_type,
                                                  double rmin, double rmax,
                                                  histbin* histo, int numhisto)
    void _coord_transform(coordinate* coords, int numCoords, coordinate* box)
    void _calc_bond_distance(coordinate* atom1, coordinate* atom2, int numatom,
                             float* box, PBCenum pbc_type, double* distances)
    void _calc_angle(coordinate* atom1, coordinate* atom2, coordinate* atom3,
                     int numatom, float* box, PBCenum pbc_type, double* angles)
    void _calc_dihedral(coordinate* atom1, coordinate* atom2, coordinate* atom3,
                        coordinate* atom4, int numatom, float* box,
                        PBCenum pbc_type, double* angles)
    void _ortho_pbc(coordinate* coords, int numcoords, float* box)
    void _triclinic_pbc(coordinate* coords, int numcoords, float* box)


OPENMP_ENABLED = True if USED_OPENMP else False
BLOCKSIZE = _BLOCKSIZE

class PBCtype(object):
    # wrapper to expose the ePBC enumerator to Python
    ortho = PBCortho
    triclinic = PBCtriclinic
    none = PBCnone
    unknown = PBCunknown

def calc_distance_array(numpy.ndarray ref, numpy.ndarray conf,
                        numpy.ndarray box,
                        PBCenum pbc_type,
                        numpy.ndarray result):
    cdef int confnum, refnum, numthreads
    confnum = conf.shape[0]
    refnum = ref.shape[0]
    numthreads = openmp.omp_get_num_threads()

    if (refnum / numthreads) < BLOCKSIZE:
        _calc_distance_array(<coordinate*>ref.data, refnum,
                             <coordinate*>conf.data, confnum,
                             NULL if box is None else <float*>box.data,
                             pbc_type, <double*>result.data)
    else:
        _calc_distance_array_vectorized(<coordinate*>ref.data, refnum,
                                        <coordinate*>conf.data, confnum,
                                        NULL if box is None else \
                                        <float*>box.data, pbc_type,
                                        <double*>result.data)

def calc_self_distance_array(numpy.ndarray ref, numpy.ndarray box,
                             PBCenum pbc_type, numpy.ndarray result):
    cdef int refnum, numthreads
    refnum = ref.shape[0]
    numthreads = openmp.omp_get_num_threads()

    if (refnum / numthreads) < BLOCKSIZE:
        _calc_self_distance_array(<coordinate*>ref.data, refnum,
                                  NULL if box is None else <float*>box.data,
                                  pbc_type, <double*>result.data)
    else:
        _calc_self_distance_array_vectorized(<coordinate*>ref.data, refnum,
                                             NULL if box is None else \
                                             <float*>box.data, pbc_type,
                                             <double*>result.data)

def calc_distance_histogram(numpy.ndarray ref, numpy.ndarray conf,
                            numpy.ndarray box, PBCenum pbc_type,
                            numpy.ndarray histo, r_min, r_max):
    cdef int confnum, refnum, histonum, numthreads
    cdef double rmin, rmax
    confnum = conf.shape[0]
    refnum = ref.shape[0]
    histonum = histo.shape[0]
    numthreads = openmp.omp_get_num_threads()
    rmin = r_min
    rmax = r_max

    if (refnum / numthreads) < BLOCKSIZE:
        _calc_distance_histogram(<coordinate*>ref.data, refnum,
                                 <coordinate*>conf.data, confnum,
                                 NULL if box is None else <float*>box.data,
                                 pbc_type, rmin, rmax, <histbin*>histo.data,
                                 histonum)
    else:
        _calc_distance_histogram_grid_based(<coordinate*>ref.data, refnum,
                                            <coordinate*>conf.data, confnum,
                                            NULL if box is None else \
                                            <float*>box.data, pbc_type,
                                            rmin, rmax, <histbin*>histo.data,
                                            histonum)

def calc_self_distance_histogram(numpy.ndarray ref, numpy.ndarray box,
                                 PBCenum pbc_type, numpy.ndarray histo, r_min,
                                 r_max):
    cdef int refnum, histonum, numthreads
    cdef double rmin, rmax
    refnum = ref.shape[0]
    histonum = histo.shape[0]
    numthreads = openmp.omp_get_num_threads()
    rmin = r_min
    rmax = r_max

    if (refnum / numthreads) < BLOCKSIZE:
        _calc_self_distance_histogram(<coordinate*>ref.data, refnum,
                                      NULL if box is None else <float*>box.data,
                                      pbc_type, rmin, rmax,
                                      <histbin*>histo.data, histonum)
    else:
        _calc_self_distance_histogram_vectorized(<coordinate*>ref.data, refnum,
                                                 NULL if box is None else \
                                                 <float*>box.data, pbc_type,
                                                 rmin, rmax,
                                                 <histbin*>histo.data, histonum)

def coord_transform(numpy.ndarray coords,
                    numpy.ndarray box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _coord_transform(<coordinate*>coords.data, numcoords,
                     <coordinate*>box.data)

def calc_bond_distance(numpy.ndarray coords1, numpy.ndarray coords2,
                       numpy.ndarray box, PBCenum pbc_type,
                       numpy.ndarray results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_bond_distance(<coordinate*>coords1.data, <coordinate*>coords2.data,
                        numcoords, NULL if box is None else <float*>box.data,
                        pbc_type, <double*>results.data)

def calc_angle(numpy.ndarray coords1, numpy.ndarray coords2,
               numpy.ndarray coords3, numpy.ndarray box, PBCenum pbc_type,
               numpy.ndarray results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_angle(<coordinate*>coords1.data, <coordinate*>coords2.data,
                <coordinate*>coords3.data, numcoords,
                NULL if box is None else <float*>box.data, pbc_type,
                <double*>results.data)

def calc_dihedral(numpy.ndarray coords1, numpy.ndarray coords2,
                 numpy.ndarray coords3, numpy.ndarray coords4,
                 numpy.ndarray box, PBCenum pbc_type, numpy.ndarray results):
    cdef int numcoords
    numcoords = coords1.shape[0]

    _calc_dihedral(<coordinate*>coords1.data, <coordinate*>coords2.data,
                  <coordinate*>coords3.data, <coordinate*>coords4.data,
                  numcoords, NULL if box is None else <float*>box.data,
                  pbc_type, <double*>results.data)

def ortho_pbc(numpy.ndarray coords, numpy.ndarray box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _ortho_pbc(<coordinate*>coords.data, numcoords, <float*>box.data)

def triclinic_pbc(numpy.ndarray coords, numpy.ndarray box):
    cdef int numcoords
    numcoords = coords.shape[0]

    _triclinic_pbc(<coordinate*>coords.data, numcoords, <float*>box.data)
