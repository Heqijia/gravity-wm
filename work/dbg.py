import math, numpy as np
from gwm.freefall_data import SceneSpec, trajectory, _fits
spec = SceneSpec(dt=0.20, frames=16, view_height=12.0, z_center=6.0)
left, right, bottom, top = spec.safe_box()
print("safe_box", (left, right, bottom, top), "span", spec.span, "fill", spec.fill_fraction)
height = (top - bottom) * spec.fill_fraction
width = (right - left) * spec.fill_fraction
print("height", height, "width", width)
for g in (4.0, 8.0, 10.0):
    rise_cap = math.sqrt(8.0 * height / (abs(g) * spec.span**2))
    low, high = spec.rise_fraction_range
    rh, rl = min(high, rise_cap), min(low, min(high, rise_cap))
    print(f"g={g}: rise_cap={rise_cap:.3f} range=({rl:.3f},{rh:.3f})")
    rng = np.random.default_rng(0)
    for _ in range(3):
        rise = float(rng.uniform(rl, rh)); vz = 0.5*g*spec.span*rise
        vx = float(rng.uniform(-1.6, 1.6))
        jz = float(rng.uniform(-1,1))*0.5*max(0.0,(top-bottom)-height)
        jx = float(rng.uniform(-1,1))*0.5*max(0.0,(right-left)-width)
        path = trajectory(x0=0.0,z0=0.0,vx=vx,vz=vz,gravity=g,dt=spec.dt,frames=spec.frames)
        x0 = 0.5*(left+right) - 0.5*(path[:,0].min()+path[:,0].max()) + jx
        z0 = 0.5*(bottom+top) - 0.5*(path[:,1].min()+path[:,1].max()) + jz
        p2 = trajectory(x0=x0,z0=z0,vx=vx,vz=vz,gravity=g,dt=spec.dt,frames=spec.frames)
        print(f"   rise={rise:.3f} vz={vz:.2f} excursion_z={p2[:,1].max()-p2[:,1].min():.3f} "
              f"z=[{p2[:,1].min():.3f},{p2[:,1].max():.3f}] x=[{p2[:,0].min():.3f},{p2[:,0].max():.3f}] fits={_fits(p2,spec)}")
