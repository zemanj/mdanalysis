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
Charge grid calculation library --- :mod:`MDAnalysis.lib.c_qgrid`
=================================================================

Serial versions of all charge grid calculations
"""

import numpy
cimport numpy

from libc.stdint cimport int64_t

cdef extern from "calc_qgrid.h":
    ctypedef float coordinate[3]
    ctypedef int64_t histbin
    cdef bint USED_OPENMP
    cdef enum PBCenum "ePBC":
        PBCortho, PBCtriclinic, PBCnone, PBCunknown
    void _calc_charge_per_volume_histogram(coordinate* coords, int numcoords,
                                           float* charges, int numqgridx,
                                           int numqgridy, int numqgridz,
                                           float* box, PBCenum pbc_type,
                                           histbin* cube_counts,
                                           double* cube_volumes,
                                           histbin* histograms, int numhistos,
                                           int numhisto, double qmin,
                                           double qmax)

OPENMP_ENABLED = True if USED_OPENMP else False

class PBCtype(object):
    # wrapper to expose the ePBC enumerator to Python
    ortho = PBCortho
    triclinic = PBCtriclinic
    none = PBCnone
    unknown = PBCunknown

def calc_charge_per_volume_histogram(numpy.ndarray coords,
                                     numpy.ndarray charges, numpy.ndarray box,
                                     PBCenum pbc_type,
                                     numpy.ndarray cube_counts,
                                     numpy.ndarray cube_volumes,
                                     numpy.ndarray histograms,
                                     ngridx, ngridy, ngridz, q_min, q_max):
    cdef int ncoords, ngx, ngy, ngz, nhistos, nhisto
    cdef double qmin, qmax
    ncoords = coords.shape[0]
    ngx = ngridx
    ngy = ngridy
    ngz = ngridz
    nhistos = histograms.shape[0]
    nhisto = histograms.shape[1]
    qmin = q_min
    qmax = q_max
    _calc_charge_per_volume_histogram(<coordinate*>coords.data, ncoords,
                                      <float*>charges.data, ngx, ngy, ngz,
                                      <float*>box.data, pbc_type,
                                      <histbin*>cube_counts.data,
                                      <double*>cube_volumes.data,
                                      <histbin*>histograms.data, nhistos,
                                      nhisto, qmin, qmax)
