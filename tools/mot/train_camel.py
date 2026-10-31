#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-30): train CAMEL (common/mot/camel.py) on KITTI-MOTS train detections.

    # 1) overfit gate (lesson 2): one train sequence, no augmentation -> PASS if association accuracy >= 0.95
    python tools/mot/train_camel.py --gate --dets ~/mot_logs/dets/kittimots_train/x_reid.npz
    # 2) training (all train sequences except --holdout; threshold computed on --holdout)
    python tools/mot/train_camel.py --dets ~/mot_logs/dets/kittimots_train/{x,cascade_rcnn,v6}_reid.npz --epochs 20

Data: detections of the train split (tools/mot/dump_dets.py --split train, then dump_reid.py), score >= 0.4
(min_det_conf). Each detection gets the GT identity it overlaps (Hungarian per class, IoU >= 0.5), else -1.
Sample = (detections file, sequence, frame t, class): tracklets = earlier detections (frames t-50 .. t-1) of every
identity detected at t, detections = their detections at t; at most 32 identities per sample (official num_samples).
Boxes are normalised by a fixed 1242 x 375 (KITTI) here and in the tracker ([Unified]).
Output: <runs>/camel/camel_kittimots.pth (+ last.pth each epoch for --resume), log ~/mot_logs/camel_train.txt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.camel import APP_DIM, CAMEL, MAX_LEN, batch_loss, hungarian  # noqa: E402
from common.mot.core import assign, iou_matrix  # noqa: E402
from common.mot.evaluate import load_kittimots  # noqa: E402

WH = np.array([1242, 375], np.float32)
MIN_CONF = 0.4
N_PER_SAMPLE = 32
LOG = Path("~/mot_logs/camel_train.txt").expanduser()


def log(msg):
    line = f"{time.strftime('%F %T')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ data
def label_sequence(rows, feats, gt_frames):
    """rows [N,7] frame,x1,y1,x2,y2,score,cls -> dict of arrays with GT identity per detection (-1 = none)."""
    keep = rows[:, 5] >= MIN_CONF
    rows, feats = rows[keep], feats[keep].astype(np.float32)
    lab = np.full(len(rows), -1, np.int64)
    for f, g in enumerate(gt_frames):
        idx = np.where(rows[:, 0] == f)[0]
        for c in np.unique(rows[idx, 6]).astype(int):
            di = idx[rows[idx, 6] == c]
            gm = g["cls"] == c
            if not gm.any():
                continue
            sim = iou_matrix(g["boxes"][gm], rows[di, 1:5])
            m, _, _ = assign(-sim)
            ok = sim[m[:, 0], m[:, 1]] >= 0.5
            lab[di[m[ok, 1]]] = g["ids"][gm][m[ok, 0]]
    b = rows[:, 1:5]
    b5 = np.c_[b[:, 0] / WH[0], b[:, 1] / WH[1], (b[:, 2] - b[:, 0]) / WH[0], (b[:, 3] - b[:, 1]) / WH[1], rows[:, 5]]
    return {"frame": rows[:, 0].astype(int), "cls": rows[:, 6].astype(int), "lab": lab, "box": b,
            "b5": b5.astype(np.float32), "app": feats}


def build_index(seqdata):
    """samples = list of (key, frame, class); per key: identity -> sorted row indices."""
    samples, by_id = [], {}
    for key, d in seqdata.items():
        ids = {}
        for i in np.argsort(d["frame"], kind="stable"):
            if d["lab"][i] >= 0:
                ids.setdefault((d["cls"][i], d["lab"][i]), []).append(i)
        by_id[key] = {k: np.array(v) for k, v in ids.items()}
        for f in np.unique(d["frame"]):
            for c in np.unique(d["cls"][d["frame"] == f]):
                at = (d["frame"] == f) & (d["cls"] == c) & (d["lab"] >= 0)
                n_ok = sum(1 for i in np.where(at)[0]
                           if ((d["frame"][by_id[key][(c, d["lab"][i])]] < f)
                               & (d["frame"][by_id[key][(c, d["lab"][i])]] >= f - MAX_LEN)).any())
                if n_ok >= 1:
                    samples.append((key, int(f), int(c)))
    return samples, by_id


def make_sample(d, ids, f, c, rng, aug):
    """-> tracks [(row indices, identity)], dets [(row, identity or -1)].
    v1.1-b: besides the matched identities (official sample) the sample also holds what the online tracker sees:
    tracklets of identities seen in the last 50 frames but not detected at t, detections of new identities (no
    tracklet) and false positives (identity -1). They enter GAFFE as context and the loss as negatives."""
    win = (d["frame"] >= f - MAX_LEN) & (d["frame"] < f) & (d["cls"] == c) & (d["lab"] >= 0)
    at = np.where((d["frame"] == f) & (d["cls"] == c))[0]
    det_ids = {int(d["lab"][i]) for i in at if d["lab"][i] >= 0}
    tracks, dets = [], []
    for lab in np.unique(d["lab"][win]):
        rows = ids[(c, int(lab))]
        past = rows[(d["frame"][rows] < f) & (d["frame"][rows] >= f - MAX_LEN)]
        if len(past):
            tracks.append((list(past), int(lab)))
    dets = [(int(i), int(d["lab"][i])) for i in at]
    matched = [t for t in tracks if t[1] in det_ids]
    others = [t for t in tracks if t[1] not in det_ids]
    if len(matched) > N_PER_SAMPLE:
        matched = [matched[k] for k in rng.choice(len(matched), N_PER_SAMPLE, replace=False)]
    if len(others) > N_PER_SAMPLE // 2:
        others = [others[k] for k in rng.choice(len(others), N_PER_SAMPLE // 2, replace=False)]
    tracks = matched + others
    keep_ids = {t[1] for t in matched}
    fp = [x for x in dets if x[1] < 0]
    if len(fp) > N_PER_SAMPLE // 2:
        fp = [fp[k] for k in rng.choice(len(fp), N_PER_SAMPLE // 2, replace=False)]
    track_ids = {t[1] for t in tracks}
    dets = [x for x in dets if x[1] in keep_ids or (x[1] >= 0 and x[1] not in track_ids)] + fp
    if aug:
        tr = [t[0] for t in tracks]
        u = rng.random()
        if u < 0.4:                                   # SwapSporadic p 0.2
            tr = swap_same_frame(d, tr, rng, p=0.2)
        elif u < 0.8:                                 # DropoutSporadic p 0.2 (keep >= 1 observation)
            tr = [[r for r in t if rng.random() >= 0.2] or [t[-1]] for t in tr]
        elif u < 0.9:                                 # SwapOccluded p 0.5, IoU >= 0.5
            tr = swap_same_frame(d, tr, rng, p=0.5, min_iou=0.5)
        tracks = [(t, lab) for t, (_, lab) in zip(tr, tracks)]
    return tracks, dets


def swap_same_frame(d, tr, rng, p, min_iou=None):
    """exchange observations of two tracklets taken at the same frame (identity-switch noise in the tracklets)."""
    pos = {}
    for k, t in enumerate(tr):
        for j, r in enumerate(t):
            pos.setdefault(d["frame"][r], []).append((k, j))
    tr = [list(t) for t in tr]
    for f, lst in pos.items():
        if len(lst) < 2:
            continue
        for (k1, j1) in lst:
            if rng.random() >= p:
                continue
            k2, j2 = lst[rng.integers(len(lst))]
            if k2 == k1:
                continue
            r1, r2 = tr[k1][j1], tr[k2][j2]
            if min_iou is not None and iou_matrix(d["box"][[r1]], d["box"][[r2]])[0, 0] < min_iou:
                continue
            tr[k1][j1], tr[k2][j2] = r2, r1
    return tr


def collate(items, seqdata, rng, aug, device):
    """items: list of (key, tracks, dets) -> model inputs; false-positive detections get unique negative labels."""
    B = len(items)
    T = max(1, max(len(it[1]) for it in items))
    D = max(1, max(len(it[2]) for it in items))
    S = max(1, max((min(len(t), MAX_LEN) for it in items for t, _ in it[1]), default=1))
    tb, ta = np.zeros((B, T, S, 5), np.float32), np.zeros((B, T, S, APP_DIM), np.float32)
    tg, tm = np.zeros((B, T, S), np.float32), np.zeros((B, T, S), bool)
    db, da = np.zeros((B, D, 1, 5), np.float32), np.zeros((B, D, 1, APP_DIM), np.float32)
    dm = np.zeros((B, D, 1), bool)
    tl, dl = np.full((B, T), -1, np.int64), np.full((B, D), -1, np.int64)
    for b, (key, tracks, dets) in enumerate(items):
        d = seqdata[key]
        f = d["frame"][dets[0][0]] if dets else None
        if f is None:
            continue
        for k, (t, lb) in enumerate(tracks):
            for j, r in enumerate(reversed(t[-MAX_LEN:])):           # newest first
                tb[b, k, j], ta[b, k, j], tg[b, k, j], tm[b, k, j] = d["b5"][r], d["app"][r], f - d["frame"][r], True
            tl[b, k] = lb
        for k, (r, lb) in enumerate(dets):
            db[b, k, 0], da[b, k, 0], dm[b, k, 0] = d["b5"][r], d["app"][r], True
            dl[b, k] = lb if lb >= 0 else -(10 ** 6 + k)
    drop_tb, drop_ta = np.zeros_like(tm), np.zeros_like(tm)
    drop_db, drop_da = np.zeros_like(dm), np.zeros_like(dm)
    if aug:
        # DropoutFeatures (app 0.4, bbox 0.1), never both on one observation
        for dt, dbx, m in ((drop_ta, drop_tb, tm), (drop_da, drop_db, dm)):
            dt[:] = (rng.random(m.shape) < 0.4) & m
            dbx[:] = (rng.random(m.shape) < 0.1) & m & ~dt
        # BBoxShake 0.05 (tracklets): left/top += U(-a,a) * w/h, w/h *= 1 + U(-a,a)
        a = 0.05
        w, h = tb[..., 2].copy(), tb[..., 3].copy()
        tb[..., 0] += rng.uniform(-a, a, w.shape) * w
        tb[..., 1] += rng.uniform(-a, a, h.shape) * h
        tb[..., 2] *= 1 + rng.uniform(-a, a, w.shape)
        tb[..., 3] *= 1 + rng.uniform(-a, a, h.shape)
        # AppEmbNoise 0.4 (tracklets): + alpha * std(emb) * N(0,1)
        ta += 0.4 * ta.std(-1, keepdims=True) * rng.standard_normal(ta.shape).astype(np.float32)
        tb *= tm[..., None]
        ta *= tm[..., None]
    z = lambda x: torch.from_numpy(x).to(device)                    # noqa: E731
    trk = dict(bbox=z(tb), app=z(ta), age=z(tg), fmask=z(tm), drop_bbox=z(drop_tb), drop_app=z(drop_ta))
    det = dict(bbox=z(db), app=z(da), age=z(np.zeros((B, D, 1), np.float32)), fmask=z(dm), drop_bbox=z(drop_db),
               drop_app=z(drop_da))
    return trk, det, z(tl), z(dl)


# ------------------------------------------------------------------ evaluation of the association
@torch.no_grad()
@torch.no_grad()
def collect_pairs(model, samples, seqdata, by_id, device, max_samples=3000, seed=0):
    """-> similarities of positive / negative (tracklet, detection) pairs and per-sample matrices."""
    rng = np.random.default_rng(seed)
    model.eval()
    pos, neg, mats = [], [], []
    order = rng.permutation(len(samples))[:max_samples]
    for s0 in range(0, len(order), 16):
        items = []
        for i in order[s0:s0 + 16]:
            key, f, c = samples[i]
            items.append((key, *make_sample(seqdata[key], by_id[key], f, c, rng, aug=False)))
        trk, det, tl, dl = collate(items, seqdata, rng, False, device)
        sim = model(trk, det)[2].cpu().numpy()
        tl, dl = tl.cpu().numpy(), dl.cpu().numpy()
        for b, it in enumerate(items):
            nt, nd = len(it[1]), len(it[2])
            if nt == 0 or nd == 0:
                continue
            s = sim[b, :nt, :nd]
            same = np.equal.outer(tl[b, :nt], dl[b, :nd])
            pos.append(s[same])
            neg.append(s[~same])
            mats.append((s, same))
    return np.concatenate(pos), np.concatenate(neg), mats


def best_threshold(pos, neg):
    grid = np.linspace(-0.2, 1.0, 241)
    bal = [((pos >= t).mean() + ((neg < t).mean() if len(neg) else 1.0)) / 2 for t in grid]
    k = int(np.argmax(bal))
    return float(grid[k]), float(bal[k])


def assoc_accuracy(mats, thr):
    """share of the true (tracklet, detection) pairs recovered by Hungarian at thr."""
    ok = tot = 0
    for s, same in mats:
        m = hungarian(s, thr)
        tot += int(same.sum())
        ok += sum(1 for r, c in m if same[r, c])
    return ok / max(tot, 1)


@torch.no_grad()
def online_hota(model, files, seqs, thr, split="train"):
    """v1.1-b model selection: the online tracker (CAMELTracker) on held-out train sequences -> mean HOTA / AssA."""
    from common.mot.camel import CAMELTracker
    from common.mot.evaluate import KITTI_CLASSES, evaluate
    model.eval()
    gt_all, _ = load_kittimots(C.DATASETS_ROOT / "KITTI_MOTS_VOC", split)
    gt = {s: v for s, v in gt_all.items() if s in seqs}
    res = []
    for p in files:
        z = np.load(Path(p).expanduser(), allow_pickle=False)
        pred = {}
        for seq, frames in gt.items():
            rows, feats = z[seq], z[f"{seq}__feat"].astype(np.float32)
            trk = {c: CAMELTracker(frame_rate=10, model=model, sim_threshold=thr) for c in (0, 1)}
            out = []
            for f in range(len(frames)):
                part = []
                for c, t in trk.items():
                    sel = (rows[:, 0] == f) & (rows[:, 6] == c)
                    o = t.update(rows[sel][:, 1:7], feats=feats[sel])
                    if len(o):
                        o[:, 4] += c * 100000
                        o[:, 5] = c
                        part.append(o)
                out.append(np.concatenate(part) if part else np.zeros((0, 7)))
            pred[seq] = out
        res.append(evaluate(gt, pred, KITTI_CLASSES)["mean"])
    return float(np.mean([r["HOTA"] for r in res])), float(np.mean([r["AssA"] for r in res]))


def set_box_stats(model, seqdata):
    b = np.concatenate([d["b5"] for d in seqdata.values()])
    model.box_mu.copy_(torch.from_numpy(b.mean(0)))
    model.box_sd.copy_(torch.from_numpy(np.maximum(b.std(0), 1e-3)))
    log(f"box mean {np.round(b.mean(0), 3).tolist()}  std {np.round(b.std(0), 3).tolist()}")


# ------------------------------------------------------------------ main
def load_data(paths, split, holdout=(), only=None):
    gt, _ = load_kittimots(C.DATASETS_ROOT / "KITTI_MOTS_VOC", split)
    train, hold = {}, {}
    for p in paths:
        z = np.load(Path(p).expanduser(), allow_pickle=False)
        name = Path(p).stem
        for seq in gt:
            if only and seq != only:
                continue
            if f"{seq}__feat" not in z.files:
                sys.exit(f"STOP: {p} has no ReID features for {seq} (run tools/mot/dump_reid.py first)")
            d = label_sequence(z[seq], z[f"{seq}__feat"], gt[seq])
            (hold if seq in holdout else train)[(name, seq)] = d
    return train, hold, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", nargs="+", required=True, help="train-split *_reid.npz files")
    ap.add_argument("--split", default="train")
    ap.add_argument("--holdout", nargs="*", default=["0001", "0019"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate-steps", type=int, default=600)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(C.RUNS_ROOT / "camel"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if a.gate:
        tr_all, _, gt = load_data(a.dets[:1], a.split)
        seq = max(gt, key=lambda s: len({i for fr in gt[s] for i in fr["ids"]}))
        data = {k: v for k, v in tr_all.items() if k[1] == seq}
        samples, by_id = build_index(data)
        samples = samples[:64]
        log(f"gate: {a.dets[0]} seq {seq}, {len(samples)} samples (fixed), {a.gate_steps} steps, no augmentation")
        model = CAMEL().to(dev)
        set_box_stats(model, data)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        for step in range(a.gate_steps):
            model.train()
            idx = rng.choice(len(samples), min(a.batch, len(samples)), replace=False)
            items = [(samples[i][0], *make_sample(data[samples[i][0]], by_id[samples[i][0]], samples[i][1],
                                                  samples[i][2], rng, False)) for i in idx]
            trk, det, tl, dl = collate(items, data, rng, False, dev)
            t_emb, d_emb, _, t_mask, d_mask = model(trk, det)
            loss = batch_loss(t_emb, d_emb, t_mask, d_mask, tl, dl)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == a.gate_steps - 1:
                log(f"gate step {step}: loss {loss.item():.4f}")
        pos, neg, mats = collect_pairs(model, samples, data, by_id, dev, max_samples=len(samples))
        thr, bal = best_threshold(pos, neg)
        acc = assoc_accuracy(mats, thr)
        res = "PASS" if acc >= 0.95 else "FAIL"
        log(f"gate result: association accuracy {acc:.3f} at threshold {thr:.3f} (balanced {bal:.3f}) -> {res}")
        Path("~/mot_logs/camel_gate.txt").expanduser().write_text(f"{res} acc {acc:.4f} thr {thr:.3f}\n")
        sys.exit(0 if res == "PASS" else 1)

    train, hold, _ = load_data(a.dets, a.split, a.holdout)
    samples, by_id = build_index(train)
    h_samples, h_by_id = build_index(hold)
    n_obs = sum(len(d["frame"]) for d in train.values())
    log(f"train: {len(a.dets)} detection files, {len(train)} sequences, {n_obs} detections (score >= {MIN_CONF}), "
        f"{len(samples)} samples; holdout {sorted({k[1] for k in hold})}: {len(h_samples)} samples")
    model = CAMEL().to(dev)
    set_box_stats(model, train)
    log(f"CAMEL parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = len(samples) // a.batch
    total, warm = steps_per_epoch * a.epochs, max(1, steps_per_epoch * a.epochs // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    start = 0
    if a.resume and (out / "last.pth").exists():
        ck = torch.load(out / "last.pth", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["epoch"] + 1
        log(f"resumed after epoch {ck['epoch']}")
    for ep in range(start, a.epochs):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(len(samples))
        for s in range(steps_per_epoch):
            items = []
            for i in perm[s * a.batch:(s + 1) * a.batch]:
                key, f, c = samples[i]
                items.append((key, *make_sample(train[key], by_id[key], f, c, rng, True)))
            trk, det, tl, dl = collate(items, train, rng, True, dev)
            t_emb, d_emb, _, t_mask, d_mask = model(trk, det)
            loss = batch_loss(t_emb, d_emb, t_mask, d_mask, tl, dl)
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            sched.step()
        pos, neg, mats = collect_pairs(model, h_samples, hold, h_by_id, dev, max_samples=1000)
        thr, bal = best_threshold(pos, neg)
        hota, assa = online_hota(model, a.dets, a.holdout, thr)
        log(f"epoch {ep + 1}/{a.epochs}: loss {np.mean(losses):.4f}  holdout pairs balanced {bal:.3f} at {thr:.3f}, "
            f"recall {assoc_accuracy(mats, thr):.3f} | holdout ONLINE HOTA {hota:.3f} AssA {assa:.3f}  "
            f"({time.time() - t0:.0f} s)")
        best = json.loads((out / "best.json").read_text()) if (out / "best.json").exists() else {"hota": -1}
        if hota > best["hota"]:
            model.sim_threshold.fill_(thr)
            torch.save({"model_state_dict": model.state_dict()}, out / "best.pth")
            (out / "best.json").write_text(json.dumps({"hota": hota, "assa": assa, "epoch": ep + 1, "thr": thr}))
        torch.save({"model_state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep}, out / "last.pth")
    # v1.1-b model selection: the epoch with the best ONLINE HOTA on the held-out train sequences (never val)
    best = json.loads((out / "best.json").read_text())
    model.load_state_dict(torch.load(out / "best.pth", map_location=dev, weights_only=False)["model_state_dict"])
    info = {"dets": [str(p) for p in a.dets], "holdout": a.holdout, "epochs": a.epochs, "samples": len(samples),
            "selection": "best holdout online HOTA", "best_epoch": best["epoch"], "sim_threshold": best["thr"],
            "holdout_online_hota": best["hota"], "holdout_online_assa": best["assa"], "version": "v1.1-b"}
    torch.save({"model_state_dict": model.state_dict(), "info": info}, out / "camel_kittimots.pth")
    (out / "camel_kittimots.json").write_text(json.dumps(info, indent=1))
    log(f"done: best epoch {best['epoch']} (holdout online HOTA {best['hota']:.3f}, AssA {best['assa']:.3f}), "
        f"sim_threshold {best['thr']:.3f} -> {out / 'camel_kittimots.pth'}")


if __name__ == "__main__":
    main()
