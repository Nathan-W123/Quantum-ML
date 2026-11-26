"""Train the milestone 3 inverse model (observables -> SELFIES).

Usage:
    python scripts/05_train_m3.py --config configs/m3_baseline.json --out runs/m3_baseline
    python scripts/05_train_m3.py --config configs/m3_tiny.json --out runs/m3_tiny   # CPU smoke test

One model is trained on a random mix of information conditions (formula and
dipole each shown with probability p_formula / p_dipole) and of noise levels
(relative noise on A, B, C drawn log-uniformly from [noise_min, noise_max],
also given to the model as an input). Each epoch draws fresh noise and a
random SELFIES variant per molecule.

Writes to --out: config.json, history.csv, best.pt (lowest validation loss), last.pt.
--resume continues from <out>/last.pt (model, optimizer, schedule, history, RNG),
which matters on Colab where sessions disconnect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as nnf

from rotml.ml import features as F
from rotml.ml.dataset import M3Data
from rotml.ml.decode import generate_candidates
from rotml.ml.metrics import recall_at_k
from rotml.ml.model import ModelConfig, Seq2Seq, count_parameters

DEFAULTS = {
    "data_dir": "data/processed/m3",
    "model": {},
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 4000,
    "epochs": 150,
    "label_smoothing": 0.1,
    "aux_weight": 0.1,
    "noise_min": 0.001,
    "noise_max": 0.01,
    "p_formula": 0.5,
    "p_dipole": 0.5,
    "random_variants": True,
    "grad_clip": 1.0,
    "eval_every": 5,
    "eval_recall_n": 256,
    "max_train": None,
    "seed": 0,
}


def save_atomic(obj, path: Path) -> None:
    """Write to a temporary file and rename, so an interrupted save never corrupts a checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_noise(n: int, lo: float, hi: float, g: torch.Generator) -> torch.Tensor:
    return torch.exp(torch.empty(n).uniform_(math.log(lo), math.log(hi), generator=g))


def make_batch(data: M3Data, idx: torch.Tensor, cfg: dict, g: torch.Generator, train: bool):
    noise = sample_noise(len(idx), cfg["noise_min"], cfg["noise_max"], g)
    present = F.random_condition_mask(len(idx), cfg["p_formula"], cfg["p_dipole"], g)
    values, present = data.features(idx, noise, present, generator=g)
    tokens = data.targets(idx, random_variant=train and cfg["random_variants"], generator=g)
    return values, present, tokens, data.count_targets(idx)


def lr_at(step: int, total: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def compute_loss(model, batch, cfg, device, amp_dtype):
    values, present, tokens, counts = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits, aux = model(values, present, tokens[:, :-1])
    target = tokens[:, 1:]
    ce = nnf.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                           ignore_index=model.pad_id, label_smoothing=cfg["label_smoothing"])
    aux_loss = nnf.mse_loss(aux.float(), counts)
    mask = target != model.pad_id
    tok_acc = ((logits.argmax(-1) == target) & mask).sum() / mask.sum()
    seq_acc = ((logits.argmax(-1) == target) | ~mask).all(1).float().mean()
    return ce + cfg["aux_weight"] * aux_loss, ce, aux_loss, tok_acc, seq_acc


@torch.no_grad()
def evaluate(model, data, idx, cfg, device, amp_dtype, g):
    model.eval()
    sums = torch.zeros(5)
    n = 0
    for s in range(0, len(idx), cfg["batch_size"]):
        b = idx[s:s + cfg["batch_size"]]
        out = compute_loss(model, make_batch(data, b, cfg, g, train=False), cfg, device, amp_dtype)
        sums += torch.tensor([float(x) for x in out]) * len(b)
        n += len(b)
    keys = ["val_loss", "val_ce", "val_aux", "val_tok_acc", "val_seq_acc"]
    return dict(zip(keys, (sums / n).tolist()))


@torch.no_grad()
def quick_recall(model, data, idx, cfg, device, g) -> dict:
    """Beam-10 recall on a fixed validation subset at 0.5% noise, per condition."""
    vocab = data.vocab
    sub = idx[: cfg["eval_recall_n"]]
    targets = [data.smiles[int(i)] for i in sub]
    out = {}
    for name, (formula, dipole) in (("ABC", (False, False)), ("ABC + formula", (True, False))):
        present = F.condition_mask(len(sub), formula, dipole)
        values, present = data.features(sub, torch.full((len(sub),), 0.005), present, generator=g)
        ranked = []
        for s in range(0, len(sub), 64):
            ranked += generate_candidates(model, values[s:s + 64].to(device),
                                          present[s:s + 64].to(device), vocab, beam=10)
        r = recall_at_k(ranked, targets, ks=(1, 10))
        out.update({f"{name} {k}": v for k, v in r.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (GPU)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop cleanly after the epoch that exceeds this many minutes (resume later)")
    args = ap.parse_args()

    started = time.time()
    cfg = DEFAULTS | json.loads(args.config.read_text())
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    eval_g = lambda: torch.Generator().manual_seed(1234)  # noqa: E731 - fixed eval noise

    data = M3Data.load(Path(cfg["data_dir"]))
    vocab = data.vocab
    train_idx, val_idx = data.indices("train"), data.indices("val")
    if cfg["max_train"]:
        train_idx = train_idx[torch.randperm(len(train_idx), generator=g)[: cfg["max_train"]]]
    val_idx = val_idx[torch.randperm(len(val_idx), generator=torch.Generator().manual_seed(0))]

    mcfg = ModelConfig(vocab_size=len(vocab), max_len=data.meta["max_len"], n_obs=F.N_OBS,
                       **cfg["model"])
    model = Seq2Seq(mcfg, pad_id=vocab.pad).to(device)
    n_params = count_parameters(model)
    fwd = torch.compile(model) if args.compile else model

    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"],
                            betas=(0.9, 0.98))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {"train": cfg, "model": mcfg.to_dict(), "n_params": n_params, "device": str(device),
         "n_train": len(train_idx), "n_val": len(val_idx)}, indent=1))
    print(f"device {device} | params {n_params / 1e6:.2f}M | train {len(train_idx)} | "
          f"val {len(val_idx)} | amp {amp_dtype}", flush=True)

    steps_per_epoch = math.ceil(len(train_idx) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    step, best, history, start_epoch = 0, float("inf"), [], 1
    if args.resume and (args.out / "last.pt").exists():
        ckpt = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        g.set_state(ckpt["rng"])
        step, best, history = ckpt["step"], ckpt["best"], ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {ckpt['epoch']} (step {step})", flush=True)
    t0 = time.time() - (history[-1]["time_s"] if history else 0.0)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        run = torch.zeros(5)
        for s in range(0, len(perm), cfg["batch_size"]):
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, cfg)
            batch = make_batch(data, perm[s:s + cfg["batch_size"]], cfg, g, train=True)
            loss, ce, aux_loss, tok_acc, seq_acc = compute_loss(fwd, batch, cfg, device, amp_dtype)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            run += torch.tensor([x.item() for x in (loss, ce, aux_loss, tok_acc, seq_acc)])
            step += 1
        run /= steps_per_epoch
        row = {"epoch": epoch, "step": step, "lr": lr_at(step, total, cfg), "time_s": time.time() - t0,
               "train_loss": run[0].item(), "train_ce": run[1].item(), "train_tok_acc": run[3].item(),
               "train_seq_acc": run[4].item()}
        row |= evaluate(model, data, val_idx, cfg, device, amp_dtype, eval_g())
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            row |= quick_recall(model, data, val_idx, cfg, device, eval_g())
        history.append(row)
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
        ckpt = {"model": model.state_dict(), "model_config": mcfg.to_dict(), "train_config": cfg,
                "epoch": epoch, "vocab": vocab.itos}
        if row["val_loss"] < best:
            best = row["val_loss"]
            save_atomic(ckpt, args.out / "best.pt")
        save_atomic(ckpt | {"optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                            "rng": g.get_state(), "step": step, "best": best, "history": history},
                    args.out / "last.pt")
        msg = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(msg, flush=True)
        if args.time_budget_min and epoch < cfg["epochs"] \
                and time.time() - started > 60 * args.time_budget_min:
            print(f"time budget reached after epoch {epoch}; continue with --resume", flush=True)
            return
    print(f"done in {time.time() - t0:.0f}s; best val_loss {best:.4f}")


if __name__ == "__main__":
    main()
