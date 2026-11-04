"""Why does the MLP probe underperform linear ridge? Print its learning curve."""
import sys
import numpy as np, torch, torch.nn.functional as F
from sg_jepa.data import open_trajectory_store
from gwm.latents import load_world_model
from gwm.probe import ProbeConfig, build_position_probe, train_position_probes

ckpt, data = sys.argv[1], sys.argv[2]
dev = torch.device('cuda')
model, _c, _s, _r = load_world_model(ckpt, device=dev); model.eval()
store = open_trajectory_store(data)
im = torch.tensor((0.485,0.456,0.406)).reshape(1,3,1,1); istd = torch.tensor((0.229,0.224,0.225)).reshape(1,3,1,1)
idx = np.flatnonzero(store.split_id == 0)[:2000]
Z, S = [], []
with torch.no_grad():
    for b in range(0, len(idx), 24):
        px, st = [], []
        for i in idx[b:b+24]:
            e = store.episode(int(i))
            t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(e.pixels,-1,1))).float().div_(255.)
            px.append((t-im)/istd); st.append(torch.from_numpy(e.state.astype(np.float32)))
        Z.append(model.encode({'pixels': torch.stack(px).to(dev)})['emb'].float().cpu())
        S.append(torch.stack(st))
Z = torch.cat(Z); S = torch.cat(S); frames = Z.shape[1]
feat = Z.reshape(-1, Z.shape[-1]); pos = S.reshape(-1, 8)[:, :2]
eids = np.repeat(np.arange(Z.shape[0]), frames)
print(f"frames {feat.shape[0]}  feature std {feat.std():.3f}  mean|.| {feat.abs().mean():.3f}")

# episode-level split, then train a plain MLP by hand with a visible curve
rng = np.random.default_rng(0); u = np.unique(eids); rng.shuffle(u)
cut = int(0.8*len(u)); trm = np.isin(eids, u[:cut]); tem = ~trm
Xtr, Ytr = feat[trm].to(dev), pos[trm].to(dev)
Xte, Yte = feat[tem].to(dev), pos[tem].to(dev)
mu, sd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
base = float((Yte - mu).pow(2).sum(-1).mean().sqrt())
print(f"predict-mean baseline {base:.4f} world")

for norm, lr, wd in [(False, 1e-3, 1e-6), (False, 3e-3, 1e-6), (False, 1e-2, 1e-6), (True, 1e-3, 1e-6)]:
    torch.manual_seed(0)
    net = build_position_probe(feat.shape[1], ProbeConfig(input_norm=norm)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
    n = Xtr.shape[0]; bs = 4096; best = 1e9; curve = []
    for ep in range(400):
        net.train(); perm = torch.randperm(n, device=dev)
        for k in range(0, n, bs):
            j = perm[k:k+bs]
            loss = F.mse_loss(net(Xtr[j]), (Ytr[j]-mu)/sd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            rmse = float((net(Xte)*sd+mu - Yte).pow(2).sum(-1).mean().sqrt())
        best = min(best, rmse)
        if ep in (0, 9, 49, 99, 199, 399): curve.append((ep+1, round(rmse, 4)))
    print(f"  norm={norm!s:<5} lr={lr:<6} best held-out RMSE {best:.4f} world  curve {curve}")

# and the library path, for comparison
pt, _pe, _s = train_position_probes(feat.to(dev), pos.to(dev), eids,
                                    config=ProbeConfig(input_norm=False, max_epochs=400, patience=60,
                                                       batch_size=4096, seed=7), device=dev)
with torch.no_grad():
    lib = float((pt(Xte) - Yte).pow(2).sum(-1).mean().sqrt())
print(f"  train_position_probes (half the episodes, patience 60): {lib:.4f} world")
