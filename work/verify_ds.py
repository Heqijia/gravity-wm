"""Round-trip the generated dataset through the sg_jepa reader and audit it."""
import json, sys
import numpy as np
from pathlib import Path
from sg_jepa.data import open_trajectory_store, TrajectoryWindowDataset

path = sys.argv[1]
store = open_trajectory_store(path)
print("episodes", len(store.lengths), "lengths", np.unique(store.lengths), "action_dim", store.action_dim)
print("split counts  train", int((store.split_id == 0).sum()), " test", int((store.split_id == 1).sum()))
md = store.metadata
print("dt", md["dt"], " contact_free", md["contact_free"], " encoding", md["pixels_encoding"])
ep = store.episode(0)
print("pixels", ep.pixels.shape, ep.pixels.dtype, "state", ep.state.shape, "gravity", ep.gravity[:3])
print("action[0]", ep.action[0], " action[1]", ep.action[1])

dt = float(md["dt"])
# second difference of the stored positions must equal (0, -g) * dt^2 exactly
worst = 0.0
for i in range(len(store.lengths)):
    e = store.episode(i)
    pos = e.state[:, :2].astype(np.float64)
    sec = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    tgt = np.array([0.0, -float(e.gravity[0])]) * dt**2
    worst = max(worst, float(np.abs(sec - tgt).max()))
print("max |second-difference - g dt^2| over all episodes:", f"{worst:.2e}")
print("one-step gravity displacement at g=4:", 4 * dt**2, "world units =",
      4 * dt**2 / float(md["scene"]["view_height"]) * 128, "px")

ds = TrajectoryWindowDataset(path, num_steps=14, split="train")
item = ds[0]
print("window keys", sorted(k for k in item), "pixels", tuple(item["pixels"].shape))
print("windows(train, num_steps=14):", len(ds), " pixel range",
      float(item["pixels"].min()), float(item["pixels"].max()))
# image sanity: object must be visible (non-white pixels) in every frame
e = store.episode(0)
frac = [(f.astype(np.int32).sum(-1) < 720).mean() for f in e.pixels]
print("object pixel fraction per frame: min %.4f max %.4f" % (min(frac), max(frac)))
