# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
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

"""
Radial Distribution Functions --- :mod:`MDAnalysis.analysis.rdf`
================================================================

Tools for calculating pair distribution functions ("radial
distribution functions" or "RDF").

.. Not Implemented yet:
.. - Structure factor?
.. - Coordination number

"""
from __future__ import division, absolute_import
import numpy as np

from ..lib.util import blocks_of
from ..lib import distances
from .base import AnalysisBase
from ..exceptions import NoDataError, SelectionWarning, ConversionWarning


class InterRDF(AnalysisBase):
    """Intermolecular pair distribution function

    Parameters
    ----------
    g1 : AtomGroup
      First AtomGroup
    g2 : AtomGroup, optional
      Second AtomGroup
    nbins : int, optional
          Number of bins in the histogram [75]
    range : tuple or list, optional
          The size of the RDF [(0.0, 15.0)]
    exclusion_block : tuple (optional)
          A tuple representing the tile to exclude from the distance array.
          [None]
    start : int, optional
          The frame to start at (default is first)
    stop : int, optional
          The frame to end at (default is last)
    step : int, optional
          The step size through the trajectory in frames (default is every
          frame)
    verbose : bool (optional)
          Show detailed progress of the calculation if set to ``True``; the
          default is ``False``.

    Example
    -------
    First create the :class:`InterRDF` object, by supplying two AtomGroups, then
    use the :meth:`run` method ::

      rdf = InterRDF(ag1, ag2)
      rdf.run()

    Results are available through the :attr:`bins` and :attr:`rdf`
    attributes::

      plt.plot(rdf.bins, rdf.rdf)

    The `exclusion_block` keyword allows the exclusion of pairs from within the
    same molecule. For example, if there are 7 atoms in each molecule, the
    exclusion mask `(7, 7)` can be used.


    .. versionadded:: 0.13.0
    .. versionchanged:: 0.17.1
       Added `backend` keyword.
    """
    def __init__(self, g1, g2, nbins=75, range=(0.0, 15.0),
                 exclusion_block=None, backend="serial", **kwargs):
        super(InterRDF, self).__init__(g1.universe.trajectory, **kwargs)
        self.g1 = g1
        self.g2 = g2
        self.u = g1.universe

        self.rdf_settings = {'bins': nbins,
                             'range': range}
        self._exclusion_block = exclusion_block
        self._backend = backend
        self._check_rdf_settings()
        self._check_selections()

    def _check_rdf_settings(self):
        _nbins, _range = self.rdf_settings['bins'], self.rdf_settings['range']
        if _nbins <= 0:
            raise ValueError("nbins must be positive. Got {}"
                             "".format(_nbins))
        if int(_nbins) != _nbins:
            warnings.warn("Truncating non-integer value nbins={0} to nbins={0}"
                          " in instance of {2}.{3}"
                          "".format(_nbins, int(_nbins),
                                    self.__class__.__module__,
                                    self.__class__.__name__),
                          category=ConversionWarning)
            self.rdf_settings['bins'] = int(_nbins)
        if _range[0] < 0.0 or _range[0] >= _range[1]:
            raise ValueError("range must be strictly positive. Got {}"
                             "".format(_range))

    def _check_selections(self):
        if len(self.g1) == 0:
            raise NoDataError("{0}.{1} g1 in instance of {2}.{3} is empty." \
                              .format(self.g1.__class__.__module__,
                                      self.g1.__class__.__name__,
                                      self.__class__.__module__,
                                      self.__class__.__name__))
        if len(self.g2) == 0:
            raise NoDataError("{0}.{1} g2 in instance of {2}.{3} is \
                              empty.".format(self.g2.__class__.__module__,
                                             self.g2.__class__.__name__,
                                             self.__class__.__module__,
                                             self.__class__.__name__))
        self._identical_selections = False
        if not self.g1.isdisjoint(self.g2):
            # Test if both atom groups are identical (regardless of order):
            if (len(self.g1) == len(self.g2)) and \
               (len(self.g1.intersection(self.g2)) == len(self.g1)):
                self._identical_selections = True
            else:
                # If atom groups are neither identical nor disjoint,
                # they overlap partially and we throw a warning:
                warnings.warn("{0}.{1} g1 and g2 in instance of {2}.{3} \
                              overlap partially.".format(
                              self.g1.__class__.__module__,
                              self.g1.__class__.__name__,
                              self.__class__.__module__,
                              self.__class__.__name__),
                              category=SelectionWarning)

    def _prepare(self):
        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count = count.astype(np.int64)  # Use integers for exact summation
        count *= 0
        self.count = count
        self.edges = edges
        self.bins = 0.5 * (edges[:-1] + edges[1:])

        # Need to know average volume
        self.volume = 0.0

        # Check if we need to take exclusions into account:
        if self._exclusion_block is not None:
            # Allocate a results array which we will reuse
            self._result = np.zeros((len(self.g1), len(self.g2)),
                                    dtype=np.float64)
            # create a mask of _result to handle exclusions:
            self._exclusion_mask = blocks_of(self._result,
                                             *self._exclusion_block)
        else:
            self._exclusion_mask = None
            self._result = None

    def _single_frame(self):
        if self._exclusion_mask is not None:
            distances.distance_array(self.g1.positions, self.g2.positions,
                                     box=self.u.dimensions, result=self._result,
                                     backend=self._backend)
            # Exclude same molecule distances
            if self._exclusion_mask is not None:
                self._exclusion_mask[:] = -1.0

            count = np.histogram(self._result, **self.rdf_settings)[0]
            self.count += count
        else:
            if self._identical_selections:
                distances.self_distance_histogram(self.g1.positions,
                                                  self.rdf_settings['range'],
                                                  histogram=self.count,
                                                  n_bins=None,
                                                  box=self.u.dimensions,
                                                  backend=self._backend)
            else:
                distances.distance_histogram(self.g1.positions,
                                             self.g2.positions,
                                             self.rdf_settings['range'],
                                             histogram=self.count,
                                             n_bins=None,
                                             box=self.u.dimensions,
                                             backend=self._backend)
        self.volume += self._ts.volume

    def _conclude(self):
        # Number of positions in each selection
        nA = len(self.g1)
        nB = len(self.g2)
        if self._exclusion_block:
            # If we had exclusions, take these into account:
            xB = self._exclusion_block[1]
            #N = nA * (nB - xB + 1)  #TODO: This ONLY EVER makes sense if g1 == g2 and xA == xB!
            N = nA * nB
        elif self._identical_selections:
            # Normalization with respect to N-particle (!) ideal gas RDF:
            N = len(self.g1) ** 2 // 2
        else:
            N = nA * nB

        # Inverse Volumes of each radial shell
        inv_shell_vol = 3.0 / ((self.edges[1:] ** 3 - self.edges[:-1] ** 3) * \
                               4.0 * np.pi)

        # RDF normalization
        norm_factors = (self.volume / (N * self.n_frames ** 2)) * inv_shell_vol
        rdf = self.count * norm_factors

        self.rdf = rdf
