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
import MDAnalysis as mda
import numpy as np

from MDAnalysisTests.datafiles import TPR, TRR

u = mda.Universe(TPR, TRR)

nframes = u.trajectory.n_frames
frames = np.arange(nframes, dtype=np.int32)
# [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8, 9])]
parts = np.split(frames, [3, 6])
parts = [[]] + parts + [[]]
# extend each part by 1 overlapping frame
offset = 1
xparts = []
for prev, part, nxt in zip(parts[:-2], parts[1:-1], parts[2:]):
    xparts.append(
        np.concatenate((prev[-offset:], part, nxt[:offset])).astype(np.int32))

print("Will use frames: ", xparts)


# XTC
atom = u.atoms[:1]
atom.write('atom.gro')

fmt = 'xtc'
for i, frames in enumerate(xparts):
    fname = f'parts_{i}.{fmt}'
    with mda.Writer(fname, 1) as W:
        for f, ts in zip(frames, u.trajectory[frames]):
            ts.time = f
            W.write(atom)

# DCD
fmt = 'dcd'
with mda.Writer(f'parts_single_frame.{fmt}', 1) as W:
    ts = u.trajectory[0]
    ts.time = 0
    ts.dt = 1
    W.write(atom)

frames = xparts[0]
i = 0
fname = f'parts_{i}.{fmt}'
with mda.Writer(fname, 1) as W:
    for f, ts in zip(frames, u.trajectory[frames]):
        ts.time = f
        W.write(atom)
