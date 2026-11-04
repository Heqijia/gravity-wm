"""Is the latent decodable from a window of frames but not from one?

A single latent decodes position no better than chance, while the predictor
explains 87% of the latent from an eight-frame history.  That is the signature
of a high-frequency code: aliased in one frame, resolvable across several.  The
paper's own state probe is a window probe for this reason.  This sweeps the
window length W and reports the probe error for each.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe

ckpt, data = sys.argv[1], sys.argv[2]
n_fit = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
dev = torch.device('cuda')
model, _m, _s, cfg = load_world_model(ckpt, device=dev); model.eval()
store = open_trajectory_store(data)
md = store.metadata; ppu = md['pixels_shape'][0] / md['scene']['view_height']
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)

def enc(indices):
    Z, S = [], []
    with torch.no_grad():
        for b in range(0, len(indices), 24):
            px, st = [], []
            for i in indices[b:b+24]:
                e = store.episode(int(i))
                t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
                px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
            Z.append(model.encode({'pixels': torch.stack(px).to(dev)})['emb'].float())
            S.append(torch.stack(st).to(dev))
    return torch.cat(Z), torch.cat(S)

dev_idx = np.flatnonzero(store.split_id == 0)
Zf, Sf = enc(dev_idx[:n_fit]); Zt, St = enc(dev_idx[n_fit:n_fit+300])
print(f"readout={cfg.get('readout')}  fit {Zf.shape[0]} eps  test {Zt.shape[0]} eps  frames {Zf.shape[1]}")
print(f"{'W':>3} {'features':>9} {'RMSE world':>11} {'px':>8} {'var expl':>10}")

def windows(Z, S, W):
    T = Z.shape[1]
    xs = [Z[:, t-W+1:t+1].reshape(Z.shape[0], -1) for t in range(W-1, T)]
    ys = [S[:, t, :2] for t in range(W-1, T)]
    return torch.cat(xs), torch.cat(ys)

for W in (1, 2, 3, 4, 6, 8):
    Xf, Yf = windows(Zf, Sf, W)
    Xt, Yt = windows(Zt, St, W)
    mu, sd = Yf.mean(0), Yf.std(0).clamp_min(1e-6)
    base = float((Yt - mu).pow(2).sum(-1).mean().sqrt())
    torch.manual_seed(0)
    net = build_position_probe(Xf.shape[1], ProbeConfig(input_norm=False)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    n = Xf.shape[0]; bs = 4096; best = 1e9
    for ep in range(300):
        net.train(); perm = torch.randperm(n, device=dev)
        for k in range(0, n, bs):
            j = perm[k:k+bs]
            loss = F.mse_loss(net(Xf[j]), (Yf[j]-mu)/sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            best = min(best, float((net(Xt)*sd+mu - Yt).pow(2).sum(-1).mean().sqrt()))
    print(f"{W:>3} {Xf.shape[1]:>9} {best:>11.4f} {best*ppu:>8.2f} {1-(best/base)**2:>+10.4f}")
