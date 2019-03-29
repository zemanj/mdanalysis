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
# Warning: Sphinx chokes if embedsignature is True

from __future__ import division

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, INFINITY, NAN

from MDAnalysis import NoDataError

from libcpp.set cimport set as cset
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref


__all__ = ['coords_add_vector', 'unique_int_1d', 'unique_masks_int_1d',
           'iscontiguous_int_1d', 'argwhere_int_1d', 'make_whole',
           'find_fragments']

cdef extern from "calc_distances.h":
    ctypedef float coordinate[3]
    void minimum_image(double* x, float* box, float* inverse_box) nogil
    void minimum_image_triclinic(double* dx, float* box) nogil
    void _ortho_pbc(coordinate* coords, int numcoords, float* box) nogil
    void _triclinic_pbc(coordinate* coords, int numcoords, float* box) nogil

ctypedef cset[int] intset
ctypedef cmap[int, intset] intmap


cdef inline bint isinf(double x) nogil:
    """Check if a double is ``inf`` or ``-inf``.

    In contrast to ``libc.math.isfinite``, this function keeps working when
    compiled with ``-ffast-math``.


    .. versionadded:: 0.20.0
    """
    return (x == INFINITY) | (x == -INFINITY)


def coords_add_vector(float[:, :] coordinates not None,
                      np.ndarray vector not None):
    """Add `vector` to each position in `coordinates`.

    Equivalent to ``coordinates += vector`` but faster for C-contiguous
    coordinate arrays. Coordinates are modified in place.

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinate array of dtype ``numpy.float32`` and shape ``(n, 3)``.
    vector: numpy.ndarray
        Single coordinate vector of shape ``(3,)``.

    Raises
    ------
    ValueError
        If the shape of `coordinates` is not ``(n, 3)`` or if the shape of
        `vector` is not ``(3,)``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t cshape1 = coordinates.shape[1]
    cdef np.intp_t vndim = vector.ndim
    cdef np.intp_t vshape0 = vector.shape[0]
    if cshape1 != 3:
        raise ValueError("Wrong shape: positions.shape != (n, 3)")
    if vndim != 1 or vshape0 != 3:
        raise ValueError("Wrong shape: vector.shape != (3,)")
    if vector.dtype == np.float32:
        _coords_add_vector32(coordinates, vector)
    else:
        _coords_add_vector64(coordinates, vector.astype(np.float64, copy=False))


cdef inline void _coords_add_vector32(float[:, :]& coordinates,
                                      float[:]& vector) nogil:
    """Low-level implementation of :func:`coords_add_vector` for float32
    vectors.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i
    for i in range(coordinates.shape[0]):
        coordinates[i, 0] += vector[0]
        coordinates[i, 1] += vector[1]
        coordinates[i, 2] += vector[2]


cdef inline void _coords_add_vector64(float[:, :]& coordinates,
                                      double[:]& vector) nogil:
    """Low-level implementation of :func:`coords_add_vector` for non-float32
    vectors.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i
    for i in range(coordinates.shape[0]):
        # Don't use += here!
        coordinates[i, 0] = coordinates[i, 0] + vector[0]
        coordinates[i, 1] = coordinates[i, 1] + vector[1]
        coordinates[i, 2] = coordinates[i, 2] + vector[2]


def _coords_add_vectors(np.ndarray[np.float32_t, ndim=2] coordinates not None,
                        np.ndarray vectors not None,
                        object[::1] compound_masks=None):
    """Add `vectors` to `coordinates`.

    If `compound_masks` is supplied, the vectors are added to the coordinates
    on a per-compound basis. Coordinates are modified in place.

    Use the ``compound_masks`` keyword with extreme caution! Masks with invalid
    indices *will cause undefined behavior* and lead to either a segmentation
    fault or silent memory corruption!

    Parameters
    ----------
    coordinates: numpy.ndarray
        Coordinate array of dtype ``numpy.float32`` and shape ``(n, 3)``.
    vectors: numpy.ndarray
        Coordinate vectors of shape ``(n, 3)`` or ``(len(compound_masks), 3)``,
        dtype will be converted to ``np.float32``.
    compound_masks
        A one-dimensional array of dtype ``object`` containing index masks for
        each compound. See :func:`unique_masks_int_1d` for details.

    Raises
    ------
    ValueError
        If the shape of `coordinates` is not ``(n, 3)`` or if the shape of
        `vectors` is not ``(n, 3)`` or ``(len(compound_masks), 3)``.

    Notes
    -----
    * Each mask in `compound_masks` must be either a slice or an index array
      with positive indices in the half-open interval ``[0, len(coordinates))``.
      Masks with invalid indices **will** cause undefined behavior and lead to
      either a segmentation fault or silent memory corruption!
    * For ``compound_masks=None``, the following two lines are equivalent:

      >>> coords_add_vectors(coordinates, vectors)

      >>> coordinates += vectors

      However, the following two constructs are *only* equivalent if *none* of the
      masks is an index array with duplicate indices:

      >>> coords_add_vectors(coordinates, vectors, compound_masks=masks)

      >>> for i, mask in enumerate(masks):
      >>>     coordinates[mask] += vectors[i]

      The reason is that numpy contracts duplicate indices in masks to single
      values. Nevertheless, the following two constructs are *always* equivalent
      if all masks are index arrays (with or without duplicates):

      >>> coords_add_vectors(coordinates, vectors, compound_masks=masks)

      >>> for i, mask in enumerate(masks):
      >>>     for j in mask:
      >>>         coordinates[j] += vector[i]


    .. versionadded:: 0.20.0
    """
    cdef float[:, :] coords = coordinates
    cdef np.intp_t n = coords.shape[0]
    cdef np.intp_t cshape1 = coords.shape[1]
    cdef np.intp_t vndim = vectors.ndim
    cdef np.intp_t nvec = vectors.shape[0]
    cdef bint per_compound = compound_masks is not None
    cdef np.intp_t i, j, k, start, stop, step
    cdef np.intp_t[::1] mask
    cdef float[:, :] fvecs
    cdef double[:, :] dvecs
    cdef slice slc
    if cshape1 != 3:
        raise ValueError("Wrong shape: positions.shape != (n, 3)")
    if vndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Wrong shape: vector.shape != (n, 3)")
    if per_compound:
        if nvec != compound_masks.shape[0]:
            raise ValueError("Number of vectors doesn't match number of "
                             "compounds.")
    elif nvec != n:
            raise ValueError("Number of vectors doesn't match number of "
                             "coordinates.")
    if n == 0 or nvec == 0:
        return
    if vectors.dtype == np.float32:
        if per_compound:
            fvecs = vectors
            if isinstance(compound_masks[0], slice):
                for i in range(nvec):
                    slc = compound_masks[i]
                    start = slc.start
                    stop = slc.stop
                    step = slc.step
                    _coords_add_vector32(coords[start:stop:step], fvecs[i])
            else:
                for i in range(nvec):
                    mask = compound_masks[i]
                    for j in range(mask.shape[0]):
                        k = mask[j]
                        coords[k, 0] += fvecs[i, 0]
                        coords[k, 1] += fvecs[i, 1]
                        coords[k, 2] += fvecs[i, 2]
        else:
            # if there's one float32 vector per coordinate, numpy's SIMD loops
            # are faster than our plain C loops:
            coordinates += vectors
    else:
        dvecs = vectors.astype(np.float64, copy=False)
        if per_compound:
            if isinstance(compound_masks[0], slice):
                for i in range(nvec):
                    start = compound_masks[i].start
                    stop = compound_masks[i].stop
                    _coords_add_vector64(coords[start:stop], dvecs[i])
            else:
                for i in range(nvec):
                    mask = compound_masks[i]
                    for j in range(mask.shape[0]):
                        k = mask[j]
                        # Don't use += here!
                        coords[k, 0] = coords[k, 0] + dvecs[i, 0]
                        coords[k, 1] = coords[k, 1] + dvecs[i, 1]
                        coords[k, 2] = coords[k, 2] + dvecs[i, 2]
        else:
            for i in range(n):
                coords[i, 0] = coords[i, 0] + dvecs[i, 0]
                coords[i, 1] = coords[i, 1] + dvecs[i, 1]
                coords[i, 2] = coords[i, 2] + dvecs[i, 2]


def coords_center(float[:, :] coordinates not None, double[:] weights=None,
                  np.intp_t[:] compound_indices=None, bint check_weights=False,
                  bint return_compound_masks=False):
    """Compute the center of a coordinate array.

    If `weights` are supplied, the center will be computed as a weighted
    average of the `coordinates`.

    If `compound_indices` are supplied, the (weighted) centers per compound will
    be computed.

    If the weights (of a compound) sum up to zero, the weighted center (of that
    compound) will be all ``nan`` (not a number). If `check_weights` is set to
    ``True``, a :class:`ValueError` will be raised in that case.

    Parameters
    ----------
    coordinates : numpy.ndarray
        An array of dtype ``numpy.float32`` and shape ``(n, 3)`` containing the
        coordinates to average.
    weights : numpy.ndarray, optional
        An array of dtype ``np.float64`` and shape ``(n,)`` containing the
        weights for each coordinate.
    compound_indices : numpy.ndarray, optional
        An array of dtype ``numpy.intp`` and shape ``(n,)`` containing the
        compound indices for each coordinate.
    check_weights : bool, optional
        If ``True``, raises a :class:`ValueError` if the weights (of any
        compound) sum up to zero.
    return_compound_masks : bool, optional
        If ``True`` and `compound_indices` is not ``None``, an array of dtype
        ``object`` containing index masks for each compound will be returned as
        well. See :func:`unique_masks_int_1d` for details.

    Returns
    -------
    numpy.ndarray
        An array of dtype ``np.float64`` and shape ``(1, 3)`` or
        ``(n_compounds, 3)`` containing the (weighted) center(s).

    Raises
    ------
    ValueError
        If the coordinates array has an invalid shape, or if the number of
        coordinates, compound indices, or weights do not match.
    ValueError
        If `check_weights` is ``True`` and the weights (of any compound) sum up
        to zero.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t n = coordinates.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] center
    cdef object[::1] comp_masks
    cdef bint weighted = weights is not None
    cdef bint per_compound = compound_indices is not None
    cdef bint zero_weights = False
    if coordinates.shape[1] != 3:
        raise ValueError("coordinates.shape is not (n, 3)")
    if per_compound and n != compound_indices.shape[0]:
        raise ValueError("Length of coordinates and compound_indices don't "
                         "match.")
    if weighted and n != weights.shape[0]:
        raise ValueError("Length of coordinates and weights don't match.")
    if n < 2:
        center = np.zeros((n, 3), dtype=np.float64)
        if n == 1:
            center[0, 0] = coordinates[0, 0]
            center[0, 1] = coordinates[0, 1]
            center[0, 2] = coordinates[0, 2]
            if weighted:
                if isinf(1.0 / weights[0]):
                    zero_weights = True
                    center[:] = NAN
                if check_weights and zero_weights:
                    raise ValueError("Weight is zero.")
        if per_compound and return_compound_masks:
            comp_masks = _unique_masks_int_1d(compound_indices, 1)
    else:
        if per_compound:
            comp_masks = _unique_masks_int_1d(compound_indices, 1)
            center = np.zeros((comp_masks.shape[0], 3), dtype=np.float64)
            if weighted:
                zero_weights = _coords_weighted_center_per_compound(coordinates,
                                                                    comp_masks,
                                                                    weights,
                                                                    center)
                if check_weights and zero_weights:
                    raise ValueError("The weights of one or more compounds sum "
                                     "up to zero.")
            else:
               _coords_center_per_compound(coordinates, comp_masks, center)
        else:
            center = np.zeros((1, 3), dtype=np.float64)
            if weighted:
                zero_weights = _coords_weighted_center(coordinates, weights,
                                                       center[0])
                if check_weights and zero_weights:
                    raise ValueError("Weights sum up to zero.")
            else:
                _coords_center(coordinates, center[0])
    if per_compound and return_compound_masks:
        return center, np.asarray(comp_masks)
    return center


cdef inline void _coords_center(float[:, :]& coords,
                                double[::1]& center) nogil:
    """Low-level implementation of :func:`coords_center` with
    ``weights == None`` and ``compound_indices == None``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i
    cdef np.intp_t n = coords.shape[0]
    cdef double inv_n = 1.0 / n
    for i in range(n):
        center[0] += coords[i, 0]
        center[1] += coords[i, 1]
        center[2] += coords[i, 2]
    center[0] *= inv_n
    center[1] *= inv_n
    center[2] *= inv_n


cdef inline bint _coords_weighted_center(float[:, :]& coords,
                                         double[:]& weights,
                                         double[::1]& center):
    """Low-level implementation of :func:`coords_center` with
    ``weights != None`` and ``compound_indices == None``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i
    cdef np.intp_t n = coords.shape[0]
    cdef double inv_sum_weights = 0.0
    cdef bint zero_weights = False
    for i in range(n):
        center[0] += coords[i, 0] * weights[i]
        center[1] += coords[i, 1] * weights[i]
        center[2] += coords[i, 2] * weights[i]
        inv_sum_weights += weights[i]
    inv_sum_weights = 1.0 / inv_sum_weights
    zero_weights = isinf(inv_sum_weights)
    if zero_weights:
        center[0] = NAN
        center[1] = NAN
        center[2] = NAN
    else:
        center[0] *= inv_sum_weights
        center[1] *= inv_sum_weights
        center[2] *= inv_sum_weights
    return zero_weights


cdef inline void _coords_center_per_compound(float[:, :]& coords,
                                             object[::1]& comp_masks,
                                             double[:, ::1]& center):
    """Low-level implementation of :func:`coords_center` with
    ``weights == None`` and ``compound_indices != None``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i, j, k
    cdef np.intp_t n = comp_masks.shape[0]
    cdef np.intp_t csize
    cdef double inv_csize
    cdef np.intp_t[:] cmask
    for i in range(n):
        cmask = comp_masks[i]
        csize = cmask.shape[0]
        inv_csize = 1.0 / csize
        for j in range(csize):
            k = cmask[j]
            center[i, 0] += coords[k, 0]
            center[i, 1] += coords[k, 1]
            center[i, 2] += coords[k, 2]
        center[i, 0] *= inv_csize
        center[i, 1] *= inv_csize
        center[i, 2] *= inv_csize


cdef inline bint _coords_weighted_center_per_compound(float[:, :]& coords,
                                                      object[::1]& comp_masks,
                                                      double[:]& weights,
                                                      double[:, ::1]& center):
    """Low-level implementation of :func:`coords_center` with
    ``weights != None`` and ``compound_indices != None``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i, j, k
    cdef np.intp_t n = comp_masks.shape[0]
    cdef np.intp_t csize
    cdef double weight, inv_sum_weights
    cdef np.intp_t[:] cmask
    cdef bint zero_weights = False
    for i in range(n):
        cmask = comp_masks[i]
        csize = cmask.shape[0]
        inv_sum_weights = 0.0
        for j in range(csize):
            k = cmask[j]
            weight = weights[k]
            inv_sum_weights += weight
            center[i, 0] += coords[k, 0] * weight
            center[i, 1] += coords[k, 1] * weight
            center[i, 2] += coords[k, 2] * weight
        inv_sum_weights = 1.0 / inv_sum_weights
        if isinf(inv_sum_weights):
            zero_weights = True
            center[i, 0] = NAN
            center[i, 1] = NAN
            center[i, 2] = NAN
        else:
            center[i, 0] *= inv_sum_weights
            center[i, 1] *= inv_sum_weights
            center[i, 2] *= inv_sum_weights
    return zero_weights


def unique_int_1d(np.intp_t[:] values not None, bint return_counts=False,
                  bint return_masks=False, bint assume_unsorted=False):
    """Find the unique elements of a 1D array of integers.

    This function is optimal on sorted arrays.

    Parameters
    ----------
    values: numpy.ndarray
        1D array of dtype ``numpy.intp`` (or equivalent) in which to find the
        unique values.
    return_counts: bool, optional
        If ``True``, the number of occurrences of each unique value is returned
        as well.
    return_masks: bool, optional
        If ``True``, an array of masks (with one mask per unique value) will be
        returned as well.
    assume_unsorted: bool, optional
        If `values` is known to be unsorted (i.e., its values are not
        monotonically increasing), setting `assume_unsorted` to ``True`` can
        speed up the computation.

    Returns
    -------
    unique: numpy.ndarray
        A deduplicated copy of `values`.
    counts: numpy.ndarray, optional
        An array of the same length as `unique` containing the number of
        occurrences of each unique value in the original `values` array. Only
        returned if `return_counts` is ``True``.
    masks: numpy.ndarray, optional
        An array of dtype ``object`` containing a mask for each value in
        `unique`. Each of the masks allows accessing all occurrences of its
        corresponding unique value in `values` such that
        ``numpy.all(values[masks[i]] == unique[i]) == True``. Thus, the masks
        array is roughly equivalent to
        ``[numpy.where(values == i) for i in numpy.unique(values)]``. Only
        returned if `return_masks` is ``True``.

    Notes
    -----
    The dtype ``numpy.intp`` is usually equivalent to ``numpy.int32`` on a 32
    bit operating system, and, likewise, equivalent to ``numpy.int64`` on a 64
    bit operating system. The exact behavior is compiler-dependent and can be
    checked with ``print(numpy.intp)``.


    See Also
    --------
    :func:`numpy.unique`
    :func:`unique_masks_int_1d`


    .. versionadded:: 0.19.0
    .. versionchanged:: 0.20.0
       Added optional  `return_counts`, `return_masks`, and `assume_unsorted`
       parameters and changed dtype from ``np.int64`` to ``np.intp``
       (corresponds to atom indices).
    """
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t n_unique
    cdef np.ndarray[np.intp_t, ndim=1] counts
    cdef np.ndarray[object, ndim=1] masks
    cdef np.ndarray[np.intp_t, ndim=1] unique = np.empty(n_values,
                                                         dtype=np.intp)
    if return_counts:
        counts = np.empty(n_values, dtype=np.intp)
        if return_masks:
            masks = np.empty(n_values, dtype=object)
            n_unique = _unique_int_1d_counts_masks(values, counts, masks,
                                                   assume_unsorted, unique)
            return unique[:n_unique], counts[:n_unique], masks[:n_unique]
        else:
            n_unique = _unique_int_1d_counts(values, counts, assume_unsorted,
                                             unique)
            return unique[:n_unique], counts[:n_unique]
    if return_masks:
        masks = np.empty(n_values, dtype=object)
        n_unique = _unique_int_1d_masks(values, masks, assume_unsorted, unique)
        return unique[:n_unique], masks[:n_unique]
    n_unique = _unique_int_1d(values, assume_unsorted, unique)
    return unique[:n_unique]


cdef inline np.intp_t _unique_int_1d(np.intp_t[:]& values, bint assume_unsorted,
                                     np.intp_t[::1] unique):
    """Low-level implementation of :func:`unique_int_1d` with
    ``return_counts=False`` and ``return_masks=False``.


    .. versionadded:: 0.20.0
    """
    cdef bint monotonic = True
    cdef np.intp_t i = 0
    cdef np.intp_t n = 0
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t[::1] sorted_values
    if n_values > 0:
        if assume_unsorted and n_values > 1:
            sorted_values = np.sort(values)
            unique[0] = sorted_values[0]
            for i in range(1, n_values):
                if sorted_values[i] != unique[n]:
                    n += 1
                    unique[n] = sorted_values[i]
        else:
            unique[0] = values[0]
            if n_values > 1:
                for i in range(1, n_values):
                    if values[i] != unique[n]:
                        if monotonic and values[i] < unique[n]:
                            monotonic = False
                        n += 1
                        unique[n] = values[i]
                if not monotonic:
                    n_values = n + 1
                    n = 0
                    sorted_values = np.sort(unique[:n_values])
                    unique[0] = sorted_values[0]
                    for i in range(1, n_values):
                        if sorted_values[i] != unique[n]:
                            n += 1
                            unique[n] = sorted_values[i]
        n += 1
    return n


cdef inline np.intp_t _unique_int_1d_counts(np.intp_t[:]& values,
                                            np.intp_t[::1]& counts,
                                            bint assume_unsorted,
                                            np.intp_t[::1] unique):
    """Low-level implementation of :func:`unique_int_1d` with
    ``return_counts=True`` and ``return_masks=False``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i = 0
    cdef np.intp_t n = 0
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t[::1] sorted_values
    if n_values > 1:
        if not assume_unsorted:
            unique[0] = values[0]
            counts[0] = 1
            for i in range(1, n_values):
                if values[i] == unique[n]:
                    counts[n] += 1
                elif values[i] < unique[n]:
                    n = 0
                    break
                else:
                    n += 1
                    unique[n] = values[i]
                    counts[n] = 1
            else:
                return n + 1
        # values are unsorted or assume_unsorted == True:
        sorted_values = np.sort(values)
        unique[0] = sorted_values[0]
        counts[0] = 1
        for i in range(1, n_values):
            if sorted_values[i] == unique[n]:
                counts[n] += 1
            else:
                n += 1
                unique[n] = sorted_values[i]
                counts[n] = 1
        n += 1
    elif n_values == 1:
        n = 1
        unique[0] = values[0]
        counts[0] = 1
    return n


cdef inline np.intp_t _unique_int_1d_masks(np.intp_t[:]& values,
                                           object[::1]& masks,
                                           bint assume_unsorted,
                                           np.intp_t[::1] unique):
    """Low-level implementation of :func:`unique_int_1d` with
    ``return_counts=False`` and ``return_masks=True``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i = 0
    cdef np.intp_t n = 0
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t[::1] sort_ix
    cdef np.intp_t[::1] slice_ix
    if n_values > 1:
        slice_ix = np.empty(n_values + 1, dtype=np.intp)
        slice_ix[0] = 0
        if not assume_unsorted:
            unique[0] = values[0]
            for i in range(1, n_values):
                if values[i] != unique[n]:
                    if values[i] < unique[n]:
                        n = 0
                        break
                    n += 1
                    unique[n] = values[i]
                    slice_ix[n] = i
            else:
                n += 1
                slice_ix[n] = n_values
                for i in range(n):
                    masks[i] = slice(slice_ix[i], slice_ix[i + 1], 1)
                return n
        # values are unsorted or assume_unsorted == True:
        sort_ix = np.argsort(values)
        unique[0] = values[sort_ix[0]]
        for i in range(1, n_values):
            if values[sort_ix[i]] != unique[n]:
                n += 1
                unique[n] = values[sort_ix[i]]
                slice_ix[n] = i
        n += 1
        slice_ix[n] = n_values
        for i in range(n):
            masks[i] = sort_ix[slice_ix[i]:slice_ix[i + 1]]
    elif n_values == 1:
        n = 1
        unique[0] = values[0]
        masks[0] = slice(0, 1, 1)
    return n


cdef inline np.intp_t _unique_int_1d_counts_masks(np.intp_t[:]& values,
                                                  np.intp_t[::1]& counts,
                                                  object[::1]& masks,
                                                  bint assume_unsorted,
                                                  np.intp_t[::1] unique):
    """Low-level implementation of :func:`unique_int_1d` with
    ``return_counts=True`` and ``return_masks=True``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i = 0
    cdef np.intp_t n = 0
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t[::1] sort_ix
    cdef np.intp_t[::1] slice_ix
    if n_values > 1:
        slice_ix = np.empty(n_values + 1, dtype=np.intp)
        slice_ix[0] = 0
        counts[0] = 1
        if not assume_unsorted:
            unique[0] = values[0]
            for i in range(1, n_values):
                if values[i] == unique[n]:
                    counts[n] += 1
                elif values[i] < unique[n]:
                    n = 0
                    break
                else:
                    n += 1
                    unique[n] = values[i]
                    counts[n] = 1
                    slice_ix[n] = i
            else:
                n += 1
                slice_ix[n] = n_values
                for i in range(n):
                    masks[i] = slice(slice_ix[i], slice_ix[i + 1], 1)
                return n
        # values are unsorted or assume_unsorted == True:
        sort_ix = np.argsort(values)
        unique[0] = values[sort_ix[0]]
        counts[0] = 1
        for i in range(1, n_values):
            if values[sort_ix[i]] == unique[n]:
                counts[n] += 1
            else:
                n += 1
                unique[n] = values[sort_ix[i]]
                slice_ix[n] = i
                counts[n] = 1
        n += 1
        slice_ix[n] = n_values
        for i in range(n):
            masks[i] = sort_ix[slice_ix[i]:slice_ix[i + 1]]
    elif n_values == 1:
        n = 1
        unique[0] = values[0]
        counts[0] = 1
        masks[0] = slice(0, 1, 1)
    return n


def unique_masks_int_1d(np.intp_t[:] values not None,
                        bint assume_unsorted=False):
    """Find the indices of each unique element in a 1D array of integers and
    return them as an array of index masks or equivalent slices, similar to
    ``[numpy.where(values == i) for i in numpy.unique(values)]``.

    This function is optimal on sorted arrays.

    Parameters
    ----------
    values: numpy.ndarray
        1D array of dtype ``numpy.intp`` (or equivalent) in which to find the
        unique values.
    assume_unsorted: bool, optional
        If `values` is known to be unsorted (i.e., its values are not
        monotonically increasing), setting `assume_unsorted` to ``True`` can
        speed up the computation.

    Returns
    -------
    masks : numpy.ndarray
        An array of dtype ``object`` containing index masks, one for each unique
        element in `values`.

    Notes
    -----
    The masks in the returned object array can themselves be either slice
    objects or potentially unsorted (we don't use stable sorting) index
    memoryviews. Memoryviews can be used for indexing just like normal numpy
    index arrays. If one wishes to print the mask indices, this can be done by
    converting the masks to numpy arrays:

    >>> for mask in masks:
    >>>     print(numpy.asarray(mask))


    See Also
    --------
    :func:`unique_int_1d`


    .. versionadded:: 0.20.0
    """
    cdef object[::1] masks = _unique_masks_int_1d(values, assume_unsorted)
    return np.asarray(masks)


cdef inline object[::1] _unique_masks_int_1d(np.intp_t[:]& values,
                                             bint assume_unsorted):
    """Low-level implementation of :func:`unique_masks_int_1d`.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i = 0
    cdef np.intp_t n = 1
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t[::1] sort_ix
    cdef np.intp_t[::1] slice_ix
    cdef object[::1] masks = np.empty(n_values, dtype=object)
    if n_values > 1:
        slice_ix = np.empty(n_values + 1, dtype=np.intp)
        slice_ix[0] = 0
        if not assume_unsorted:
            for i in range(1, n_values):
                if values[i] != values[i - 1]:
                    if values[i] < values[i - 1]:
                        n = 1
                        break
                    slice_ix[n] = i
                    n += 1
            else:
                slice_ix[n] = n_values
                for i in range(n):
                    masks[i] = slice(slice_ix[i], slice_ix[i + 1], 1)
                return masks[:n]
        # values are unsorted or assume_unsorted == True:
        sort_ix = np.argsort(values)
        for i in range(1, n_values):
            if values[sort_ix[i]] != values[sort_ix[i - 1]]:
                slice_ix[n] = i
                n += 1
        slice_ix[n] = n_values
        for i in range(n):
            masks[i] = sort_ix[slice_ix[i]:slice_ix[i + 1]]
        masks = masks[:n]
    elif n_values == 1:
        masks[0] = slice(0, 1, 1)
    return masks


def iscontiguous_int_1d(np.intp_t[:] values):
    """Checks if an integer array is a contiguous range.

    Checks if ``values[i+1] == values[i] + 1`` holds for all elements of
    `values`.

    Parameters
    ----------
    values: numpy.ndarray
        1D array of dtype ``numpy.intp``.

    Returns
    -------
    bool
        ``True`` if `values` is a contiguous range of numbers, ``False``
        otherwise or if `values` is empty.

    Notes
    -----
    The dtype ``numpy.intp`` is usually equivalent to ``numpy.int32`` on a 32
    bit operating system, and, likewise, equivalent to ``numpy.int64`` on a 64
    bit operating system. The exact behavior is compiler-dependent and can be
    checked with ``print(numpy.intp)``.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t n_values = values.shape[0]
    cdef np.intp_t i
    if n_values > 1:
        for i in range(n_values - 1):
            if values[i] + 1 != values[i + 1]:
                return False
    elif n_values == 0:
        return False
    return True


def indices_to_slice_1d(np.ndarray[np.intp_t, ndim=1] indices):
    """Converts an index array to a slice if possible.

    Slice conversion is only performed if all indices are non-negative.

    Parameters
    ----------
    indices: numpy.ndarray
        1D array of dtype ``numpy.intp``.

    Returns
    -------
    slice or np.ndarray
        If `indices` can be represented by a slice and all its elements are
        non-negative, a slice object is returned. Otherwise, the `indices` array
        is returned.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t[:] ix = indices
    cdef np.intp_t n = ix.shape[0]
    cdef np.intp_t i
    cdef np.intp_t stride
    if n > 0 and ix[0] >= 0:
        if n > 1:
            stride = ix[1] - ix[0]
            if stride > 0:
                for i in range(1, n - 1):
                    if ix[i] + stride != ix[i + 1]:
                        return indices  # multiple strides
                return slice(ix[0], ix[n - 1] + 1, stride)
            elif stride < 0 and ix[n - 1] >= 0:
                for i in range(1, n - 1):
                    if ix[i] + stride != ix[i + 1]:
                        return indices  # multiple strides
                if ix[n - 1] == 0:
                    return slice(ix[0], None, stride)  # last index is 0
                return slice(ix[0], ix[n - 1] - 1, stride)
            return indices  # stride == 0
        return slice(ix[0], ix[0] + 1, 1)  # single index
    return indices  # empty array


def argwhere_int_1d(np.intp_t[:] arr, np.intp_t value):
    """Find the array indices where elements of `arr` are equal to `value`.

    This function is similar to calling `numpy.argwhere(arr == value).ravel()`
    but is a bit faster since it avoids creating an intermediate boolean array.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to search. Must be one-dimensional and of dtype ``numpy.intp``
        or equivalent.
    value : int
        The value to search for. Must be of a type compatible with
        ``numpy.intp``.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array of dtype ``numpy.intp`` containing all indices
        ``i`` which satisfy the condition ``arr[i] == value``.

    Notes
    -----
    The dtype ``numpy.intp`` is usually equivalent to ``numpy.int32`` on a 32
    bit operating system, and, likewise, equivalent to ``numpy.int64`` on a 64
    bit operating system. The exact behavior is compiler-dependent and can be
    checked with ``print(numpy.intp)``.

    See Also
    --------
    :func:`numpy.argwhere`


    .. versionadded:: 0.20.0
    """
    cdef np.ndarray[np.intp_t, ndim=1] result = np.empty(arr.shape[0], dtype=np.intp)
    cdef np.intp_t nargs = _argwhere_int_1d(arr, value, result)
    return result[:nargs]


cdef inline np.intp_t _argwhere_int_1d(np.intp_t[:] arr, np.intp_t value,
                                       np.intp_t[::1] result) nogil:
    """Low-level implementation of :func:`argwhere_int_1d`.


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i
    cdef np.intp_t nargs = 0
    cdef np.intp_t n = arr.shape[0]
    for i in range(n):
        if arr[i] == value:
            result[nargs] = i
            nargs += 1
    return nargs


cdef inline intset difference(intset a, intset b) nogil:
    """a.difference(b)

    Returns a set of values in `a` which are not in `b`.
    """
    cdef intset output
    for val in a:
        if b.count(val) == 0:
            output.insert(val)
    return output


def make_whole(atomgroup, reference_atom=None, bint inplace=True):
    """Move all atoms in a single molecule so that bonds don't split over
    images.

    This function is most useful when atoms have been packed into the primary
    unit cell, causing breaks mid molecule, with the molecule then appearing
    on either side of the unit cell. This is problematic for operations
    such as calculating the center of mass of the molecule. ::

       +-----------+             +-----------+
       |           |             |           |
       | 6       3 |           3 | 6         |
       | !       ! |           ! | !         |
       |-5-8   1-2-|   ==>   1-2-|-5-8       |
       | !       ! |           ! | !         |
       | 7       4 |           4 | 7         |
       |           |             |           |
       +-----------+             +-----------+

    Parameters
    ----------
    atomgroup : AtomGroup
        The :class:`MDAnalysis.core.groups.AtomGroup` to work with.
        The positions of this are modified in place.  All these atoms
        must belong in the same molecule or fragment.
    reference_atom : :class:`~MDAnalysis.core.groups.Atom`
        The atom around which all other atoms will be moved.
        Defaults to atom 0 in the atomgroup.
    inplace : bool, optional
        If ``True``, coordinates are modified in place.

    Returns
    -------
    coords : numpy.ndarray
        The unwrapped atom coordinates.

    Raises
    ------
    NoDataError
        If the supplied AtomGroup is too large to be unwrapped without bond
        information (i.e., the group spans across more than half of the box in
        at least one dimension) but the underlying topology has no bonds.
        (Note that bonds can be guessed, see
        :func:`MDAnalysis.topology.core.guess_bonds`)

    ValueError
        If the algorithm fails to work. This is usually caused by the atomgroup
        not being a single fragment (i.e., the molecule can't be traversed by
        following bonds).


    Example
    -------
    Make fragments whole::

        from MDAnalysis.lib.mdamath import make_whole

        # This algorithm requires bonds, these can be guessed!
        u = mda.Universe(......, guess_bonds=True)

        # MDAnalysis can split AtomGroups into their fragments
        # based on bonding information.
        # Note that this function will only handle a single fragment
        # at a time, necessitating a loop.
        for frag in u.atoms.fragments:
            make_whole(frag)

    Alternatively, to keep a single atom in place as the anchor::

        # This will mean that atomgroup[10] will NOT get moved,
        # and all other atoms will move (if necessary).
        make_whole(atomgroup, reference_atom=atomgroup[10])


    Note
    ----
    This function is exception-safe, i.e., in case of failure, even when
    `inplace` is ``True``, the atomgroup's positions will remain unchanged.

    See Also
    --------
    :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`


    .. versionadded:: 0.11.0
    .. versionchanged:: 0.20.0
        Inplace-modification of atom positions is now optional, and positions
        are returned as a numpy array.
    """
    cdef np.intp_t i, j, nloops, atom, other, natoms
    cdef np.intp_t ref
    cdef cmap[int, int] ix_to_rel
    cdef np.int32_t[:, ::1] bond_ix
    cdef float[:, ::1] pos
    cdef bint ortho = True
    cdef float[::1] box
    cdef float half_box[3]
    cdef float inverse_box[3]
    cdef float* hbox = &half_box[0]
    cdef float* ibox = &inverse_box[0]
    cdef np.intp_t[:] atom_ix
    cdef bint success

    atom_ix = atomgroup._ix
    natoms = atom_ix.shape[0]

    # Nothing to do for less than 2 atoms
    if natoms < 2:
        return atomgroup.positions

    box = atomgroup.dimensions

    for i in range(3):
        half_box[i] = 0.5 * box[i]
        if box[i] <= 0.0:
            raise ValueError("Invalid dimensions: At least one box dimension "
                             "is non-positive. You can set valid dimensions "
                             "using 'atomgroup.dimensions ='.")
    for i in range(3, 6):
        if box[i] != 90.0:
            ortho = False
            if box[i] >= 180.0:
                raise ValueError("Invalid dimensions: At least one box angle is "
                                 "greater than or equal to 180 degrees. You can "
                                 "set valid dimensions using "
                                 "'atomgroup.dimensions='.")

    positions = atomgroup.positions
    pos = positions

    if ortho:
        # If atomgroup is already unwrapped, bail out:
        if isunwrapped_ortho(pos, &half_box[0]):
            return np.asarray(pos)
        inverse_box[0] = 1.0 / box[0]
        inverse_box[1] = 1.0 / box[1]
        inverse_box[2] = 1.0 / box[2]
    else:
        from .mdamath import triclinic_vectors
        box = triclinic_vectors(box).ravel()
        # Silence compiler warnings:
        inverse_box[0] = 0.0
        inverse_box[1] = 0.0
        inverse_box[2] = 0.0

    if reference_atom is None:
        ref = 0
    else:
        # Sanity check:
        atom = reference_atom._ix
        with nogil:
            success = False
            for i in range(natoms):
                if atom == atom_ix[i]:
                    ref = i
                    success = True
                    break
        if not success:
            raise ValueError("Reference atom not in atomgroup")

    success = False
    if ortho:
        # Try to unwrap without bond information:
        success = _make_whole_ortho_nobonds(pos, &box[0], hbox, ibox, ref)
        # If unwrapping without bonds failed, get new positions:
        if not success:
            pos = atomgroup.positions
    # Unwrap with bond information:
    if not success:
        # map of global indices to local indices
        with nogil:
            for i in range(natoms):
                ix_to_rel[atom_ix[i]] = i
        try:
            bond_ix = atomgroup.bondindices
        except (AttributeError, NoDataError):
            raise NoDataError("Cannot make molecule whole without bond "
                              "information in the topology.")
        success = _make_whole_bonds(pos, ix_to_rel, bond_ix, &box[0], ibox,
                                    ortho, ref)
        if not success:
            raise ValueError("Cannot make molecule whole: AtomGroup is not "
                             "contiguous from bonds (i.e., probably not a "
                             "molecule).")
    if inplace:
        atomgroup.positions = pos
    return np.asarray(pos)


cdef bint _make_whole_bonds(float[:, ::1]& pos, cmap[int, int]& ix_to_rel,
                            np.int32_t[:, ::1]& bond_ix, float* box,
                            float* ibox, bint ortho, np.intp_t ref) nogil:
    """Low-Level implementation of :meth:`make_whole` using bonds. In contrast
    to its counterpart :func:`_make_whole_nobonds`, this function is capable of
    unwrapping molecules which are larger than half the box in any dimension.
    However, this advantage comes with a significant performance penalty.

    Parameters
    ----------
    pos : memoryview
        A C-contiguous 2d memoryview of type ``numpy.float32`` and shape
        ``(n, 3)`` holding the positions to work with. Will be modified in
        place.
    ix_to_rel : libcpp.map[int, int]
        A data structure mapping atom indices to relative (i.e., zero-based)
        indices.
    bond_ix : memoryview
        A C-contiguous 2d memoryview of type ``numpy.int32`` and shape
        ``(m, 2)`` holding the bond indices to work with.
    box : float*
        Pointer to an array holding either orthorhombic box information or a
        flattened triclinic box matrix.
    ortho : bint
        Boolean value indicating an orthorhombic (``True``) or triclinic
        (``False``) unit cell.
    ref : np.intp_t
        The *relative* index in `pos` of the reference atom around which all
        other atoms will be moved.

    Returns
    -------
    success : bool
        A boolean value indicating success (``True``) or failure (``False``).


    .. versionadded:: 0.20.0
    """
    cdef intset refpoints, todo, done
    cdef np.intp_t i, nloops, atom, other, natoms
    cdef intmap bondmap
    cdef double vec[3]

    natoms = <np.intp_t> ix_to_rel.size()

    # C++ dict of bonds
    for i in range(bond_ix.shape[0]):
        atom = bond_ix[i, 0]
        other = bond_ix[i, 1]
        # only add bonds if both atoms are in atoms set
        if ix_to_rel.count(atom) and ix_to_rel.count(other):
            atom = ix_to_rel[atom]
            other = ix_to_rel[other]
            bondmap[atom].insert(other)
            bondmap[other].insert(atom)

    refpoints = intset()  # Who is safe to use as reference point?
    done = intset()  # Who have I already searched around?
    # initially we have one starting atom whose position is in correct image
    refpoints.insert(ref)

    nloops = 0
    while <np.intp_t> refpoints.size() < natoms and nloops < natoms:
        # count iterations to prevent infinite loop here
        nloops += 1

        # We want to iterate over atoms that are good to use as reference
        # points, but haven't been searched yet.
        todo = difference(refpoints, done)
        for atom in todo:
            for other in bondmap[atom]:
                # If other is already a refpoint, leave alone
                if refpoints.count(other):
                    continue
                # Draw vector from atom to other
                for i in range(3):
                    vec[i] = pos[other, i] - pos[atom, i]
                # Apply periodic boundary conditions to this vector
                if ortho:
                    minimum_image(&vec[0], box, ibox)
                else:
                    minimum_image_triclinic(&vec[0], box)
                # Then define position of other based on this vector
                for i in range(3):
                    pos[other, i] = pos[atom, i] + vec[i]
                # This other atom can now be used as a reference point
                refpoints.insert(other)
            done.insert(atom)

    if <np.intp_t> refpoints.size() < natoms:
        return False
    return True


cdef bint _make_whole_ortho_nobonds(float[:, ::1]& pos, float* box, float* hbox,
                                    float* ibox, np.intp_t ref) nogil:
    """Low-Level implementation of :func:`make_whole` which unwraps molecules
    in an orthorhombic box without bond information.

    This routine will only be successful if the unwrapped molecule is smaller
    than half the box size in all dimensions. Nevertheless, it is much faster
    than its counterpart :func:`_make_whole_bonds`, which has to traverse across
    the molecule's bond network.

    Parameters
    ----------
    pos : memoryview
        A C-contiguous 2d memoryview of type ``numpy.float32`` and shape
        ``(n, 3)`` holding the positions to work with. Will be modified in
        place.
    box : float*
        Pointer to an array holding orthorhombic box information.
    hbox : float*
        Pointer to an array holding half orthorhombic box edge lengths in all
        three dimensions.
    ibox : float*
        Pointer to an array holding inverse orthorhombic box edge lengths in all
        three dimensions.
    ref : np.intp_t
        The *relative* index in `pos` so that all other atoms of the compound
        will be moved around ``pos[ref]``.

    Returns
    -------
    success : bool
        A boolean value indicating success (``True``) or failure (``False``).
        Failure indicates that the unwrapped molecule spans more than half the
        box in at least one dimension.


    .. versionadded:: 0.20.0
    """
    cdef bint shift_required = False
    cdef np.intp_t natoms = pos.shape[0]
    cdef np.intp_t i, j
    cdef float ref_pos[3]
    cdef float min_pos[3]
    cdef float max_pos[3]
    cdef double vec[3]
    cdef float dist

    ref_pos[0] = pos[ref, 0]
    ref_pos[1] = pos[ref, 1]
    ref_pos[2] = pos[ref, 2]

    min_pos[0] = ref_pos[0]
    min_pos[1] = ref_pos[1]
    min_pos[2] = ref_pos[2]

    max_pos[0] = ref_pos[0]
    max_pos[1] = ref_pos[1]
    max_pos[2] = ref_pos[2]

    # Unwrap:
    for i in range(natoms):
        if i == ref:
            continue
        for j in range(3):
            dist = pos[i, j] - ref_pos[j]
            # check if a shift is required:
            if dist > hbox[j]:
                shift_required = True
                break
            elif dist <= -hbox[j]:
                shift_required = True
                break
        if shift_required:
            for j in range(3):
                vec[j] = pos[i, j] - ref_pos[j]
            minimum_image(&vec[0], box, ibox)
            for j in range(3):
                pos[i, j] = ref_pos[j] + vec[j]
                # check for inexact multi-box shift:
                dist = pos[i, j] - ref_pos[j]
                if (dist > hbox[j]):
                    pos[i, j] -= box[j]
                elif (dist <= -hbox[j]):
                    pos[i, j] += box[j]
            shift_required = False
        # Update the molecule's minimum and maximum bounding position:
        for j in range(3):
            if pos[i, j] < min_pos[j]:
                min_pos[j] = pos[i, j]
            elif pos[i, j] > max_pos[j]:
                max_pos[j] = pos[i, j]

    # Check if unwrap failed (i.e., if molecule spans more than half the box):
    for i in range(3):
        if (max_pos[i] - min_pos[i]) > hbox[i]:
            return False
    return True


cdef bint _make_whole_ortho_nobonds_masked(float[:, ::1]& pos,
                                           np.intp_t[::1]& mask, float* box,
                                           float* hbox, float* ibox,
                                           np.intp_t ref) nogil:
    """Low-Level implementation of :func:`make_whole` which unwraps molecules
    in an orthorhombic box without bond information.

    In contrast to :func:`_make_whole_ortho_nobonds`, this function requires a
    `mask` containing relative indices so that only the positions in `pos`
    corresponding to the mask indices will be unwrapped.

    This routine will only be successful if the unwrapped molecule is smaller
    than half the box size in all dimensions. Nevertheless, it is much faster
    than its counterpart :func:`_make_whole_bonds`, which has to traverse across
    the molecule's bond network.

    Parameters
    ----------
    pos : memoryview
        A C-contiguous 2d memoryview of type ``numpy.float32`` and shape
        ``(n, 3)`` holding the positions to work with. Will be modified in
        place.
    mask : memoryview
        A contiguous 1d memoryview of type ``numpy.intp_t`` containing indices
        of a compound's positions in the `pos` array.
    box : float*
        Pointer to an array holding orthorhombic box information.
    hbox : float*
        Pointer to an array holding half orthorhombic box edge lengths in all
        three dimensions.
    ibox : float*
        Pointer to an array holding inverse orthorhombic box edge lengths in all
        three dimensions.
    ref : np.intp_t
        The *relative* index in `mask` so that all other atoms of the compound
        will be moved around ``pos[mask[ref]]``.

    Returns
    -------
    success : bool
        A boolean value indicating success (``True``) or failure (``False``).
        Failure indicates that the unwrapped molecule spans more than half the
        box in at least one dimension.


    .. versionadded:: 0.20.0
    """
    cdef bint shift_required = False
    cdef np.intp_t natoms = mask.shape[0]
    cdef np.intp_t i, j, k
    cdef float ref_pos[3]
    cdef float min_pos[3]
    cdef float max_pos[3]
    cdef double vec[3]
    cdef float dist

    i = mask[ref]
    ref_pos[0] = pos[i, 0]
    ref_pos[1] = pos[i, 1]
    ref_pos[2] = pos[i, 2]

    min_pos[0] = ref_pos[0]
    min_pos[1] = ref_pos[1]
    min_pos[2] = ref_pos[2]

    max_pos[0] = ref_pos[0]
    max_pos[1] = ref_pos[1]
    max_pos[2] = ref_pos[2]

    # Unwrap:
    for i in range(natoms):
        if i == ref:
            continue
        k = mask[i]
        for j in range(3):
            dist = pos[k, j] - ref_pos[j]
            # check if a shift is required:
            if dist > hbox[j]:
                shift_required = True
                break
            elif dist <= -hbox[j]:
                shift_required = True
                break
        if shift_required:
            for j in range(3):
                vec[j] = pos[k, j] - ref_pos[j]
            minimum_image(&vec[0], box, ibox)
            for j in range(3):
                pos[k, j] = ref_pos[j] + vec[j]
                # check for inexact multi-box shift:
                dist = pos[k, j] - ref_pos[j]
                if (dist > hbox[j]):
                    pos[k, j] -= box[j]
                elif (dist <= -hbox[j]):
                    pos[k, j] += box[j]
            shift_required = False
        # Update the molecule's minimum and maximum bounding position:
        for j in range(3):
            if pos[k, j] < min_pos[j]:
                min_pos[j] = pos[k, j]
            elif pos[k, j] > max_pos[j]:
                max_pos[j] = pos[k, j]

    # Check if unwrap failed (i.e., if molecule spans more than half the box):
    for i in range(3):
        if (max_pos[i] - min_pos[i]) > hbox[i]:
            return False
    return True


def _unwrap(ag not None, np.ndarray[np.float32_t, ndim=2] positions not None,
            bint have_bonds, bint ortho, float[::1] box not None,
            object[::1] comp_masks, double[:, ::1] refpos, double[::1] weights):
    """Move (compounds of) `atomgroup` so that its bonds aren't split across
    periodic boundaries.

    Low-level implementation of `MDanalysis.core.groups.AtomGroup.unwrap()`.

    Parameters
    ----------
    ag : MDanalysis.core.groups.AtomGroup
        The AtomGroup whose compounds to unwrap. *Must not be empty*!
    positions : numpy.ndarray
        A C-contiguous numpy array of shape ``(len(ag), 3)`` and dtype
        ``numpy.float32`` containing the AtomGroup's positions. Will be modified
        in place.
    have_bonds : bool
        Boolean value indicating whether the underlying topology contains bonds.
    ortho : bool
        If ``True``, the `box` is assumed to be orthorhombic, otherwise
        triclinic.
    box : numpy.ndarray
        A 1d numpy array of dtype ``numpy.float32`` carrying orthorhombic or
        flattened triclinic box information.
    comp_masks : numpy.ndarray, optional
        An array of shape ``(n_compounds,)`` and dtype ``object`` containing
        index masks for the individual compounds of the AtomGroup. If
        `comp_masks` is ``None`` or has length 1, the whole group is regarded as
        a single compound. Note that if compounds are larger than half the box
        in any dimension, all atoms within such compounds must be interconnected
        by bonds, i.e., compounds must correspond to (parts of) molecules.
    refpos : nupy.ndarray, optional
        An array of shape ``(len(comp_masks), 3)`` (or ``(1, 3)`` if
        `comp_masks` is ``None``) and dtype ``numpy.float64`` serving as a
        buffer to store and return the reference coordinates, i.e., the
        (weighted) centers of each compound. Will be modified in place.
        If provided, unwrapped compounds will be shifted so that their
        individual reference point lies within the primary unit cell. If
        ``None``, no such shift is performed.
    weights : nupy.ndarray, optional
        A contiguous 1d numpy array of shape ``(len(ag),)`` and dtype
        ``numpy.float64`` containing weights to compute weighted reference
        positions. If ``None`` and `refpos` is not ``None``, this indicates that
        centers of geometry will be used as reference positions. If `refpos` is
        ``None``, this parameter will be ignored.

    Returns
    -------
    zero_weights : bool
        A boolean indicating whether the weights (of any compound) sum up to
        zero. If `weights` is ``None``, this will always be ``False``.


    .. versionadded:: 0.20.0
    """
    cdef np.ndarray[np.float32_t, ndim=2] shiftsarr
    cdef float[:, ::1] shifts
    cdef float[:, ::1] targets
    cdef float[:, ::1] pos = positions
    cdef float half_box[3]
    cdef float inverse_box[3]
    cdef float* hbox = &half_box[0]
    cdef float* ibox = &inverse_box[0]
    cdef np.intp_t i, j
    cdef np.intp_t natoms = pos.shape[0]
    cdef np.intp_t natoms_comp = 0
    cdef np.intp_t ncomp = 0
    cdef np.intp_t[:] atom_ix
    cdef np.intp_t[::1] comp_mask
    cdef np.int32_t[:, ::1] bond_ix
    cdef cmap[int, int] ix_to_rel
    cdef bint per_compound = comp_masks is not None
    cdef bint ref = refpos is not None
    cdef bint weighted = weights is not None
    cdef bint zero_weights = False
    cdef bint success = False

    assert len(ag) == len(positions)
    if per_compound:
        assert len(comp_masks)

    if ortho:
        # Prepare inverse box for orthorhombic wrapping:
        inverse_box[0] = 1.0 / box[0]
        inverse_box[1] = 1.0 / box[1]
        inverse_box[2] = 1.0 / box[2]
        half_box[0] = 0.5 * box[0]
        half_box[1] = 0.5 * box[1]
        half_box[2] = 0.5 * box[2]
    else:
        # Silence compiler warnings:
        inverse_box[0] = 0.0
        inverse_box[1] = 0.0
        inverse_box[2] = 0.0
        half_box[0] = 0.0
        half_box[1] = 0.0
        half_box[2] = 0.0
        # TODO: Enable triclinic wrapping without bonds.
        # Until then, we just give up and cry. ;-(
        if not have_bonds:
            raise NoDataError("Unwrapping (compounds of) groups in triclinic "
                              "systems without having bond information in the "
                              "topology is not (yet) supported.")

    if per_compound:
        ncomp = comp_masks.shape[0]
    else:
        ncomp = 1

    if ncomp == 1:
        # we have only one compound, so we just wrap all positions:
        if ortho:
            # try possible shortcuts for orthorhombic boxes:
            success = isunwrapped_ortho(pos, hbox)
            if not success:
                # group is not already unwrapped, try unwrapping without bonds:
                success = _make_whole_ortho_nobonds(pos, &box[0], hbox, ibox, 0)
            if (not success) and (not have_bonds):
                # ortho wrapping w/o bonds failed and no bonds available,
                # cowardly bailing out:
                raise NoDataError("No bonds in topology: Cannot unwrap groups "
                                  "which span more than half the box in any "
                                  "direction without bond information.")
        if not success:
            # ortho wrapping w/o bonds failed or we have a triclinic box, so we
            # fall back to unwrapping by traversing bonds:
            atom_ix = ag._ix
#            print("6. ag._ix:", ag._ix)
            bond_ix = ag.bondindices
#            print("7. ag.bondindices:", ag.bondindices)
            # map absolute to relative atom indices:
            for i in range(natoms):
#                print("ix_to_rel: {} -> {}".format(atom_ix[i], i))
                ix_to_rel[<np.int32_t> atom_ix[i]] = <np.int32_t> i
            success = _make_whole_bonds(pos, ix_to_rel, bond_ix, &box[0], ibox,
                                        ortho, 0)
            if not success:
                raise ValueError("Cannot unwrap group: Group is not contiguous "
                                 "from bonds (i.e., probably not a molecule).")
        if ref:
            if weighted:
                zero_weights = _coords_weighted_center(pos, weights, refpos[0])
            else:
                _coords_center(pos, refpos[0])
            # only apply reference shift if reference position is valid:
            if not zero_weights:
                shifts = np.empty((1, 3), dtype=np.float32)
                targets = np.empty((1, 3), dtype=np.float32)
                # first, copy origin to target arrays:
                targets[0, 0] = refpos[0, 0]
                targets[0, 1] = refpos[0, 1]
                targets[0, 2] = refpos[0, 2]
                # wrap targets:
                if ortho:
                    _ortho_pbc(<coordinate*> &targets[0, 0], 1, &box[0])
                else:
                    _triclinic_pbc(<coordinate*> &targets[0, 0], 1, &box[0])
                # shift = target - origin:
                shifts[0, 0] = targets[0, 0] - refpos[0, 0]
                shifts[0, 1] = targets[0, 1] - refpos[0, 1]
                shifts[0, 2] = targets[0, 2] - refpos[0, 2]
                # apply the reference shift:
                _coords_add_vector32(pos, shifts[0])
                # copy target to refpos:
                refpos[0, 0] = targets[0, 0]
                refpos[0, 1] = targets[0, 1]
                refpos[0, 2] = targets[0, 2]
    else:
        # we have multiple compounds, handle them one by one:
        if ortho:
            for i in range(ncomp):
                success = False
                comp_mask = comp_masks[i]
                success = isunwrapped_ortho_masked(pos, comp_mask, hbox)
                if not success:
                    # compound is not unwrapped, try unwrapping without bonds:
                    success = _make_whole_ortho_nobonds_masked(pos, comp_mask,
                                                               &box[0], hbox,
                                                               ibox, 0)
                    if not success:
#                        print("#### comp {} ####".format(i))
#                        print("comp_mask:", np.asarray(comp_mask))
                        if not have_bonds:
                            # ortho wrapping w/o bonds failed and no bonds
                            # available, cowardly bailing out:
                            raise NoDataError("No bonds in topology: Cannot "
                                              "unwrap compounds which span "
                                              "more than half the box in any "
                                              "direction without bond "
                                              "information.")
                        # fall back to unwrapping by traversing bonds:
                        comp_ag = ag[comp_mask]
                        atom_ix = comp_ag._ix
#                        print("comp_ag._ix:", comp_ag._ix)
                        bond_ix = comp_ag.bondindices
#                        print("comp_ag.bondindices:", comp_ag.bondindices)
                        ix_to_rel.clear()
                        natoms_comp = comp_mask.shape[0]
                        # map absolute to relative atom indices:
                        for j in range(natoms_comp):
#                            print("ix_to_rel: {} -> {}".format(atom_ix[j], comp_mask[j]))
                            ix_to_rel[atom_ix[j]] = comp_mask[j]
#                        # get a fresh copy of positions:
#                        positions = ag.positions
#                        pos = positions
                        success = _make_whole_bonds(pos, ix_to_rel, bond_ix,
                                                    &box[0], ibox, ortho,
                                                    comp_mask[0])
                        if not success:
                            raise ValueError("Cannot unwrap compound: Compound "
                                             "is not contiguous from bonds "
                                             "(i.e., probably not a molecule).")
        else:
            # we have a triclinic box, so we currently have to fall back to
            # unwrapping by traversing bonds:
            for i in range(ncomp):
                success = False
                comp_mask = comp_masks[i]
                comp_ag = ag[comp_mask]
                atom_ix = comp_ag._ix
                bond_ix = comp_ag.bondindices
                ix_to_rel.clear()
                natoms_comp = comp_mask.shape[0]
                for j in range(natoms_comp):
                    ix_to_rel[atom_ix[j]] = comp_mask[j]
                success = _make_whole_bonds(pos, ix_to_rel, bond_ix, &box[0],
                                            ibox, ortho, comp_mask[0])
                if not success:
                    raise ValueError("Cannot unwrap compound: Compound is not "
                                     "contiguous from bonds (i.e., probably "
                                     "not a molecule).")
        if ref:
            #print("unwrapped unshifted comp positions:", positions)
            if weighted:
                zero_weights = _coords_weighted_center_per_compound(pos,
                                                                    comp_masks,
                                                                    weights,
                                                                    refpos)
            else:
                _coords_center_per_compound(pos, comp_masks, refpos)
            # only apply reference shifts if there are no invalid reference
            # positions:
            if not zero_weights:
                shiftsarr = np.empty((ncomp, 3), dtype=np.float32)
                targets = np.empty((ncomp, 3), dtype=np.float32)
                shifts = shiftsarr
                # first, copy origins to targets arrays:
                for i in range(ncomp):
                    targets[i, 0] = refpos[i, 0]
                    targets[i, 1] = refpos[i, 1]
                    targets[i, 2] = refpos[i, 2]
                # wrap the targets:
                if ortho:
                    _ortho_pbc(<coordinate*> &targets[0, 0], ncomp, &box[0])
                else:
                    _triclinic_pbc(<coordinate*> &targets[0, 0], ncomp, &box[0])
                # shifts = targets - origins:
                for i in range(ncomp):
                    shifts[i, 0] = targets[i, 0] - refpos[i, 0]
                    shifts[i, 1] = targets[i, 1] - refpos[i, 1]
                    shifts[i, 2] = targets[i, 2] - refpos[i, 2]
                # apply the reference shifts:
                _coords_add_vectors(positions, shiftsarr, comp_masks)
                # copy target from shifts to refpos:
                for i in range(ncomp):
                    refpos[i, 0] = targets[i, 0]
                    refpos[i, 1] = targets[i, 1]
                    refpos[i, 2] = targets[i, 2]
    return zero_weights


cdef inline bint isunwrapped_ortho(float[:, ::1]& pos, float* hbox) nogil:
    """Check if the positions of an atomgroup are definitely unwrapped in
    an orthorhombic unit cell.

    Parameters
    ----------
    pos : memoryview
        The positions to check. Must be of shape ``(n, 3)`` and type ``float``.
    hbox : float*
        A pointer to an array containing the half edge lengths of the system's
        unit cell.

    Returns
    -------
    bint
        A boolean value indicating if the positions are unwrapped (``True``) or
        if this cannot be decided (``False``).


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i, j
    cdef np.intp_t natoms = pos.shape[0]
    for i in range(1, natoms):
        for j in range(3):
            if fabs(pos[i, j] - pos[0, j]) >= hbox[j]:
                return False
    return True


cdef inline bint isunwrapped_ortho_masked(float[:, ::1]& pos,
                                          np.intp_t[::1]& mask,
                                          float* hbox) nogil:
    """Check if the positions of an atomgroup are definitely unwrapped in
    an orthorhombic unit cell.

    In contrast to :func:`isunwrapped_ortho`, this function requires a `mask`
    containing relative indices so that only the positions in `pos`
    corresponding to the mask indices will be checked.

    Parameters
    ----------
    pos : memoryview&
        Referance to a C-contiguous memoryview of the positions to check.
        Must be of shape ``(n, 3)`` and type ``float``.
    mask : memoryview&
        Reference to a contiguous 1d memoryview of type ``numpy.intp_t``
        containing indices of a compound's positions in the `pos` array.
    hbox : float*
        A pointer to an array containing the half edge lengths of the system's
        unit cell.

    Returns
    -------
    bint
        A boolean value indicating if the positions are unwrapped (``True``) or
        if this cannot be decided (``False``).


    .. versionadded:: 0.20.0
    """
    cdef np.intp_t i, j, k
    cdef np.intp_t natoms = mask.shape[0]
    cdef np.intp_t ref = mask[0]
    for i in range(1, natoms):
        k = mask[i]
        for j in range(3):
            if fabs(pos[k, j] - pos[ref, j]) >= hbox[j]:
                return False
    return True


cdef float _dot(float * a, float * b):
    """Return dot product of two 3d vectors"""
    cdef ssize_t n
    cdef float sum1

    sum1 = 0.0
    for n in range(3):
        sum1 += a[n] * b[n]
    return sum1


cdef void _cross(float * a, float * b, float * result):
    """
    Calculates the cross product between 3d vectors

    Note
    ----
    Modifies the result array
    """

    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = - a[0]*b[2] + a[2]*b[0]
    result[2] = a[0]*b[1] - a[1]*b[0]


cdef float _norm(float* a):
    """
    Calculates the magnitude of the vector
    """
    cdef float result
    cdef ssize_t n
    result = 0.0
    for n in range(3):
        result += a[n]*a[n]
    return sqrt(result)


cpdef np.float64_t _sarrus_det_single(np.float64_t[:, ::1] m) nogil:
    """Computes the determinant of a 3x3 matrix."""
    cdef np.float64_t det
    det = m[0, 0] * m[1, 1] * m[2, 2]
    det -= m[0, 0] * m[1, 2] * m[2, 1]
    det += m[0, 1] * m[1, 2] * m[2, 0]
    det -= m[0, 1] * m[1, 0] * m[2, 2]
    det += m[0, 2] * m[1, 0] * m[2, 1]
    det -= m[0, 2] * m[1, 1] * m[2, 0]
    return det


cpdef np.ndarray _sarrus_det_multiple(np.float64_t[:, :, ::1] m):
    """Computes all determinants of an array of 3x3 matrices."""
    cdef np.intp_t n
    cdef np.intp_t i
    cdef np.float64_t[:] det
    n = m.shape[0]
    det = np.empty(n, dtype=np.float64)
    for i in range(n):
        det[i] = m[i, 0, 0] * m[i, 1, 1] * m[i, 2, 2]
        det[i] -= m[i, 0, 0] * m[i, 1, 2] * m[i, 2, 1]
        det[i] += m[i, 0, 1] * m[i, 1, 2] * m[i, 2, 0]
        det[i] -= m[i, 0, 1] * m[i, 1, 0] * m[i, 2, 2]
        det[i] += m[i, 0, 2] * m[i, 1, 0] * m[i, 2, 1]
        det[i] -= m[i, 0, 2] * m[i, 1, 1] * m[i, 2, 0]
    return np.asarray(det)


def find_fragments(atoms, bondlist):
    """Calculate distinct fragments from nodes (atom indices) and edges (pairs
    of atom indices).

    Parameters
    ----------
    atoms : array_like
       1-D Array of atom indices (dtype will be converted to ``numpy.int64``
       internally)
    bonds : array_like
       2-D array of bonds (dtype will be converted to ``numpy.int32``
       internally), where ``bonds[i, 0]`` and ``bonds[i, 1]`` are the
       indices of atoms connected by the ``i``-th bond. Any bonds referring to
       atom indices not in `atoms` will be ignored.

    Returns
    -------
    fragments : list
       List of arrays, each containing the atom indices of a fragment.

    .. versionaddded:: 0.19.0
    """
    cdef intmap bondmap
    cdef intset todo, frag_todo, frag_done
    cdef vector[int] this_frag
    cdef int i, a, b
    cdef np.intp_t[:] atoms_view
    cdef np.int32_t[:, :] bonds_view

    atoms_view = np.asarray(atoms, dtype=np.intp)
    bonds_view = np.asarray(bondlist, dtype=np.int32)

    # grab record of which atoms I have to process
    # ie set of all nodes
    for i in range(atoms_view.shape[0]):
        todo.insert(atoms_view[i])
    # Process edges into map
    for i in range(bonds_view.shape[0]):
        a = bonds_view[i, 0]
        b = bonds_view[i, 1]
        # only include edges if both are known nodes
        if todo.count(a) and todo.count(b):
            bondmap[a].insert(b)
            bondmap[b].insert(a)

    frags = []

    while not todo.empty():  # While not all nodes have been done
        # Start a new fragment
        frag_todo.clear()
        frag_done.clear()
        this_frag.clear()
        # Grab a start point for next fragment
        frag_todo.insert(deref(todo.begin()))

        # Loop until fragment fully explored
        while not frag_todo.empty():
            # Pop next in this frag todo
            a = deref(frag_todo.begin())
            frag_todo.erase(a)
            if not frag_done.count(a):
                this_frag.push_back(a)
                frag_done.insert(a)
                todo.erase(a)
                for b in bondmap[a]:
                    if not frag_done.count(b):
                        frag_todo.insert(b)

        # Add fragment to output
        frags.append(np.asarray(this_frag))

    return frags
