import MDAnalysis as mda
import numpy as np

from MDAnalysisTests.datafiles import TPR, TRR

u = mda.Universe(TPR, TRR)

nframes = u.trajectory.n_frames
frames = np.arange(nframes, dtype=np.int32)
parts = np.split(frames, [3,6])   # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8, 9])]
parts = [[]] + parts + [[]]
# extend each part by 1 overlapping frame
offset = 1
xparts = []
for prev, part, nxt in zip(parts[:-2], parts[1:-1], parts[2:]):
    #print(prev, part, nxt)
    xparts.append( np.concatenate((prev[-offset:], part, nxt[:offset])).astype(np.int32) )

print("Will use frames: ", xparts)

with mda.Writer('parts_0frame.xtc', 1) as W:
    ts = u.trajectory[0]
    ts.time = 0
    W.write(u.atoms[:1])

twoatoms = u.atoms[:1]
twoatoms.write('twoatoms.gro')

fmt = 'xtc'
for i, frames in enumerate(xparts):
    fname = f'parts_{i}.{fmt}'
    with mda.Writer(fname, 1) as W:
        for f, ts in zip(frames, u.trajectory[frames]):
            ts.time = f
            print(ts.time, f)
            W.write(twoatoms)
