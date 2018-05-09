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

"""Fast charge grid computation --- qgrid
===================================================================

Fast C-routines to calculate charge grids and corresponding histograms
from coordinate and charge arrays. All functions also exist in parallel
versions providing higher performance than the serial code.
The boolean attribute USED_OPENMP can be checked to see if OpenMP was
used in the compilation of qgrid.

Selection of acceleration ("backend")
-------------------------------------

All functions take the optional keyword *backend*, which determines
the type of acceleration. Currently, the following choices are
implemented (*backend* is case-insensitive):

.. Table:: Available *backends* for accelerated distance functions.

   ========== ========================= ======================================
   *backend*  module                    description
   ========== ========================= ======================================
   "serial"   :mod:`c_qgrid`            serial implementation in C/Cython

   "OpenMP"   :mod:`c_qgrid_openmp`     parallel implementation in C/Cython
                                        with OpenMP
   ========== ========================= ======================================

.. versionadded:: 0.1.0

Functions
---------

.. autofunction:: calc_charge_per_volume_histogram(positions, charges, box, grid_shape, histo_range [, backend])

"""
from __future__ import division, absolute_import
from six.moves import range

import numpy as np

# hack to select backend with backend=<backend> kwarg.
import importlib
_qgrid = {}
_qgrid['serial'] = importlib.import_module(".c_qgrid",
                                           package="MDAnalysis.lib")
try:
    _qgrid['openmp'] = importlib.import_module(".c_qgrid_openmp",
                                               package="MDAnalysis.lib")
except ImportError:
    pass
del importlib

def _run(funcname, args=None, kwargs=None, backend="serial"):
    """Helper function to select a backend function *funcname*."""
    args = args if args is not None else tuple()
    kwargs = kwargs if kwargs is not None else dict()
    backend = backend.lower()
    try:
        func = getattr(_qgrid[backend], funcname)
    except KeyError:
        raise ValueError("Function {0} not available with backend {1}; try one of: {2}".format(
            funcname, backend, ", ".join(_qgrid.keys())))
    return func(*args, **kwargs)

# serial versions are always available
from .c_qgrid import (PBCtype,
                      calc_charge_per_volume_histogram)

from .c_qgrid_openmp import OPENMP_ENABLED as USED_OPENMP


def _box_check(box):
    """Take a box input and deduce what type of system it represents based
    on the shape of the array and whether all angles are 90.

    Parameters
    ----------
    box : array
        Box information of unknown format.

    Returns
    -------
    boxtype : str
        * ``ortho`` orthogonal box
        * ``tri_vecs`` triclinic box vectors
        * ``tri_box`` triclinic box lengths and angles

    Raises
    ------
    TypeError
        If box is not float32.
    ValueError
        If box type not detected.
    """
    if box.dtype != np.float32:
        raise TypeError("Box must be of type float32")

    boxtype = 'unknown'

    if box.shape == (3,):
        boxtype = 'ortho'
    elif box.shape == (3, 3):
        if np.all([box[0][1] == 0.0,  # Checks that tri box is properly formatted
                      box[0][2] == 0.0,
                      box[1][2] == 0.0]):
            boxtype = 'tri_vecs'
        else:
            boxtype = 'tri_vecs_bad'
    elif box.shape == (6,):
        if np.all(box[3:] == 90.):
            boxtype = 'ortho'
        else:
            boxtype = 'tri_box'

    if boxtype == 'unknown':
        raise ValueError("box input not recognised"
                         ", must be an array of box dimensions")

    return boxtype


def _check_array(coords, desc):
    """Check an array is a valid array of coordinates

    Must be:
       (n,3) in shape
       float32 data
    """
    if (coords.ndim != 2 or coords.shape[1] != 3):
        raise ValueError("{0} must be a sequence of 3 dimensional coordinates"
                         "".format(desc))
    _check_array_dtype(coords, desc)


def _check_array_dtype(coords, desc, dtype=np.float32):
    """Check whether an array contains values of correct dtype"""
    if coords.dtype != dtype:
        raise TypeError("{0} must be of type {}.".format(desc, dtype))


def charge_per_volume_histogram(positions, charges, box, grid_shape, max_charge,
                                histo_resolution=0.1, backend="serial"):
    """Calculate charge histograms for different cubic volumes.

    This routine takes an array of `positions` and corresponding `charges` and
    maps the charges onto a grid (binning only, no interpolation) of shape
    `grid_shape` covering the whole `box`.
    Thereafter, cubic volume slices with sizes ranging from 1 grid cell to the
    maximum possible are taken into account. For each cube size, each cell of
    the charge grid is taken as the origin of the cube once.
    For each origin, the charge contained in the respective cube is added to a
    charge histogram (each cube size has its own histogram).
    Thus, the returned `h` array is a 2d array with its first dimension going
    over the cube edge lengths, and the second going over the considered charge
    range (i.e., the second dimension is the actual histogram for each of the
    cube sizes). Along with the histograms, the used histogram resolution,
    the number of cubes for each cube edge length, and the corresponding cube
    volumes are returned.

    Parameters
    ----------
    positions : numpy.ndarray of ``dtype=numpy.float32``
        Coordinate array.
    charges : numpy.ndarray of ``dtype=numpy.float32``
        Charge array containing the charges corresponding to the particles at
        `positions`.
    box : numpy.array
        Dimensions of the cell according to which the minimum image convention
        is applied. The dimensions must be provided in the same format as
        returned by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.
        As of now, only orthogonal boxes are supported!
    grid_shape : tuple of int
        Tuple of three integers representing the number of grid points in each
        dimension of the `box`.
    max_charge: float
        A floating point number representing the maximum charge considered
        in the resulting charge histogram. The minimum charge considered will be
        its negative value.
    histo_resolution: float (default: 0.1)
        Charge histogram bin width (in units of elementary charge e).
    backend: str (default: "serial")
        Select the type of acceleration; "serial" is always available. Other
        possibilities are "OpenMP" (OpenMP).

    Returns
    -------
    h : numpy.ndarray
        2d numpy array of ``dtype=numpy.int64`` containing histograms of net
        charges present in cubic volumes sampled on all possible positions of
        `qgrid`.
        The indices of the first dimension are cube edge length minus 1, the
        second dimension contains the respective net charge histograms.
    r : float
        actually used histogram resolution
    c : numpy.ndarray
        1d numpy array of ``dtype=numpy.int64`` containing the number of cubes
        sampled for each cube edge length.
    v : numpy.ndarray
        1d numpy array of ``dtype=numpy.float64`` containing the volumes of the
        sampled cubes.

    .. versionadded:: 0.1.0
    """

    _check_array(positions, "positions")
    _check_array_dtype(charges, "charges")
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("First dimensions of positions and charges array do "
                         "not match.")
    if not positions.flags['C_CONTIGUOUS']:
        raise TypeError("positions array is not C contiguous.")
    if not charges.flags['C_CONTIGUOUS']:
        raise TypeError("charges array is not C contiguous.")
    if np.any(box[:3] <= 0.0):
        raise ValueError("box must not contain dimensions of zero or negative "
                         "length.")
    boxtype = _box_check(box)
    if boxtype == 'ortho':
        pbc_type = PBCtype.ortho
    else:
        raise NotImplementedError("Function charge_per_volume_histogram() is "
                                  "currently only implemented for orthogonal "
                                  "boxes.")
    if not isinstance(grid_shape, tuple) or len(grid_shape) != 3:
        raise TypeError("grid_shape must be a tuple of length 3.")
    for i in range(len(grid_shape)):
        if int(grid_shape[i]) != grid_shape[i] or grid_shape[i] <= 0:
            raise ValueError("Elements of grid_shape must be positive "
                             "integers.")
    if max_charge <= 0.0:
        raise ValueError("Maximum charge must be strictly positive, got {}."
                         "".format())
    if histo_resolution <= 0.0 or histo_resolution >= max_charge:
        raise ValueError("Invalid histogram resolution ({}): Must be a "
                         "positive value smaller than max_charge ({})."
                         "".format(histo_resolution, max_charge))
    min_charge = -max_charge
    max_cube_edge_length = min(grid_shape) // 2
    cube_counts = np.zeros((max_cube_edge_length,), dtype=np.int64)
    cube_volumes = np.zeros((max_cube_edge_length,), dtype=np.float64)
    n_histbins = int(2.0 * max_charge / histo_resolution + 0.5)
    if n_histbins % 2 == 0:
        n_histbins += 1
    histo_resolution = 2.0 * max_charge / n_histbins
    histograms = np.zeros((max_cube_edge_length, n_histbins), dtype=np.int64)

    _run("calc_charge_per_volume_histogram",
         args=(positions, charges, box, pbc_type, cube_counts, cube_volumes,
               histograms, grid_shape[0], grid_shape[1], grid_shape[2],
               min_charge, max_charge),
         backend=backend)

    return histograms, histo_resolution, cube_counts, cube_volumes
