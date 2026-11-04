import sys, torch
p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
h = p.get('history', [])
val = [r for r in h if 'split' in r]
tr = [r for r in h if 'split' not in r]
print(f"  step {p['step']}  completed_epochs {p['cursor']['completed_epochs']}  best_val {p.get('best_validation_loss')}")
for r in val:
    print("   ep %2d  val_loss %10.3f  rollout %10.3f  sigreg %7.2f" %
          (r['epoch'], r['validation_loss'], r['validation_rollout_loss'], r['validation_sigreg_loss']))
if tr:
    import numpy as np
    k = max(1, len(tr)//10)
    print("   train loss (every ~10%):", [round(float(np.mean([x['loss'] for x in tr[i:i+k]])),2) for i in range(0, len(tr), k)])
