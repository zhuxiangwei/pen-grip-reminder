#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度分析一段录像：指标分布 + 检出率时间线 + 关键帧存图。只读，不改任何配置。"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
import mediapipe as mp                                  # noqa: E402
import pg_utils as pg                                   # noqa: E402
import grip_metrics as gm                               # noqa: E402
import hand_probe as hp                                 # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None
OUT = os.path.join(ROOT, "bench", "frames")
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
hand = hp.make_hand(pg.find_model(hp.HAND_MODEL))
sels = hp.HandSelector(lock=hp.load_cfg().get("hand_lock"))

rows = []
saved = {"first": False}
tick = 0
n = 0
save_at = set()   # 每 5 秒存一帧（若有检到手）
for s in range(0, int(total / fps) + 1, 5):
    save_at.add(int(s * fps))

while True:
    ok, fr = cap.read()
    if not ok or fr is None:
        break
    n += 1
    if n % 2:
        continue
    h, w = fr.shape[:2]
    tick += 33
    r = hand.detect_for_video(mp.Image(
        image_format=mp.ImageFormat.SRGB, data=fr[:, :, ::-1].copy()), tick)
    hands = [(lm, "?") for lm in (r.hand_landmarks or [])]
    sel = sels.update(hands, w, h) if hands else None
    lm = hands[sel][0] if sel is not None else None
    m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
    q, _ = gm.view_quality(m) if m else ("bad", "")
    rows.append({
        "frame": n, "t": round(n / fps, 2), "n_hands": len(hands),
        "px_w": m and round(m["px_w"], 1),
        "palm_angle": m and round(m["palm_view_angle"], 1),
        "thumb_side": m and round(m["thumb_side"], 3),
        "tia": m and round(m["thumb_index_angle"], 1),
        "tid": m and round(m["thumb_index_dist"], 3),
        "gap": m and round(m["thumb_index_gap"], 3),
        "q": q,
    })
    # 存图：第一次检到手的帧 + 每 5 秒有手的帧
    if lm is not None and ((not saved["first"]) or (n in save_at and len([x for x in rows if x["frame"] in save_at and x["thumb_side"] is not None]) <= 8)):
        if (not saved["first"]) or n in save_at:
            vis = hp.draw(fr, lm, m, q, {k: 0 for k in hp.CLASSES},
                          f"t={n/fps:.1f}s", zoom=True, hands=hands, sel=sel)
            p = os.path.join(OUT, f"f{n:05d}_t{int(n/fps)}s.jpg")
            cv2.imwrite(p, vis)
            saved["first"] = True

cap.release()
hand.close()

# ---- 汇总 ----
det = [r for r in rows if r["thumb_side"] is not None]
print("=" * 72)
print(f"深度分析：{os.path.basename(VIDEO)}")
print(f"总帧 {total}  fps {fps:.1f}  时长 {total/fps:.0f}s   取样每2帧 -> {len(rows)} 样本")
print("=" * 72)
print(f"检出率（写字手有指标）: {len(det)}/{len(rows)} = {100*len(det)/max(1,len(rows)):.0f}%")
two = sum(1 for r in rows if r["n_hands"] > 1)
print(f"画面里两只手          : {two}/{len(rows)} = {100*two/max(1,len(rows)):.0f}%")
print(f"视角判定 ok 比例       : {100*sum(1 for r in det if r['q']=='ok')/max(1,len(det)):.0f}%")

def pct(name, vals, unit=""):
    if not vals:
        print(f"  {name:<18} 无数据")
        return
    a = np.array([v for v in vals if v is not None], dtype=float)
    print(f"  {name:<18} 中位 {np.median(a):8.2f}{unit}   p10 {np.percentile(a,10):8.2f}   p90 {np.percentile(a,90):8.2f}")

print("\n核心指标分布（检出帧）:")
pct("px_w 手宽", [r["px_w"] for r in det], "px")
pct("palm_angle 视角角", [r["palm_angle"] for r in det], "°")
pct("thumb_side 越过量", [r["thumb_side"] for r in det])
pct("thumb_index_angle 夹角", [r["tia"] for r in det], "°")
pct("thumb_index_dist 距离", [r["tid"] for r in det])
pct("thumb_index_gap 间距", [r["gap"] for r in det])

# 检出率时间线（每 5 秒）
print("\n检出率时间线（每 5 秒）:")
bucket = {}
for r in rows:
    b = int(r["t"] // 5) * 5
    c = bucket.setdefault(b, [0, 0])
    c[0] += 1
    if r["thumb_side"] is not None:
        c[1] += 1
for b in sorted(bucket):
    tot, got = bucket[b]
    bar = "#" * int(20 * got / max(1, tot))
    print(f"  {b:>3}s-{b+5:<3}s  {got:>3}/{tot:<3} {bar}")

# thumb_side 时间序列（每 2 秒取中位）—— 看有没有"越过去"的片段
print("\nthumb_side 时间序列（每 2 秒中位，正值=拇指越到食指外侧）:")
ts = {}
for r in det:
    b = int(r["t"] // 2) * 2
    ts.setdefault(b, []).append(r["thumb_side"])
for b in sorted(ts):
    v = float(np.median(ts[b]))
    flag = " ←偏正" if v > -0.1 else ""
    print(f"  {b:>3}s  {v:+.2f}{flag}")

with open(os.path.join(OUT, "deep_rows.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
print(f"\n逐帧数据已存 {os.path.join(OUT, 'deep_rows.json')}")
print(f"标注帧已存 {OUT}/")
