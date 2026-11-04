import numpy as np
from sg_jepa.data import open_trajectory_store
st = open_trajectory_store('/fs/scratch/PAS3151/qijia387/gwm/data/ball_smoke')
e = st.episode(0)
frac = [(f.astype(np.int32).sum(-1) < 720).mean() for f in e.pixels]
print(f'  object pixel fraction: min {min(frac):.4f} max {max(frac):.4f} (constant => no rotation/shape change)')
print(f'  theta range {e.state[:,4].min():.3f}..{e.state[:,4].max():.3f}  omega {e.state[:,5].min():.3f}..{e.state[:,5].max():.3f}')
# centroid of dark pixels vs recorded position: checks the render matches the label
ys, xs = np.mgrid[0:128, 0:128]
md = st.metadata; vh = md['scene']['view_height']; zc = md['scene']['z_center']
err = []
for i, f in enumerate(e.pixels):
    m = (f.astype(np.int32).sum(-1) < 720)
    cx, cy = xs[m].mean(), ys[m].mean()
    wx = (cx - 63.5) / 128 * vh
    wz = zc - (cy - 63.5) / 128 * vh
    err.append(abs(wx - e.state[i,0]) + abs(wz - e.state[i,1]))
print(f'  pixel-centroid vs label disagreement: max {max(err):.4f} world units = {max(err)*128/vh:.3f} px')
