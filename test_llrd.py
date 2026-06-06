"""
test_llrd.py  -  驗證 LLRD 分層 LR 是否正確建立
"""
import sys, os, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from train import build_llrd_optimizer
from model import MAEASTFineTune

print("=== Loading model (this takes ~10s) ===")
model = MAEASTFineTune(
    pretrained_ckpt = CFG.PRETRAINED_CKPT,
    mae_ast_root    = CFG.MAE_AST_PROJECT_ROOT,
    num_classes     = 13,
    freeze_encoder  = False,
)

print("\n=== Building LLRD optimizer ===")
opt = build_llrd_optimizer(
    model,
    base_lr      = CFG.LR,
    weight_decay = CFG.WEIGHT_DECAY,
    decay_rate   = CFG.LLRD_DECAY,
)

print(f"\n=== Param groups summary ({len(opt.param_groups)} groups) ===")
for g in opt.param_groups:
    n_params = sum(p.numel() for p in g["params"])
    print(f"  {g.get('name','?'):<20} lr={g['lr']:.2e}  params={n_params/1e6:.2f}M")

print("\nLLRD OK!")
