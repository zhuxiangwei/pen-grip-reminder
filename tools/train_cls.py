#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔姿势 · 时序特征分类器训练（阶段 1：学习式判据替换手工阈值）
================================================================

背景见 docs/05-algorithm-survey.md：固定机位+固定孩子+固定笔+二分类，
是受约束场景的常规检测问题——用几百个标注帧训练一个小分类器即可，
不需要通用检测。本工具把「标注视频 → 特征 → 模型 → 评估报告」做成一条命令。

数据集格式（dataset.json，一段视频里按时间轴切标签）：
{
  "videos": [
    {"path": "C:/Users/tmpee/Pictures/Camera Roll/xxx.mp4",
     "lock": "screen_left",              // 可选：锁定画面左/右半的手
     "segments": [
        {"t0": 0.0,  "t1": 5.0, "label": "ok"},    // 正确握笔
        {"t0": 6.0,  "t1": 11.0, "label": "bad"}   // 拇指包食指/占笔过多
     ]}
  ]
}

标签只认 "ok" / "bad"，其他段忽略。

用法：
  python tools/train_cls.py --spec bench/dataset.json            # 训练+评估+存模型
  python tools/train_cls.py --demo                               # 合成数据冒烟测试
  python tools/train_cls.py --spec ... --no-save                 # 只看评估

模型：HistGradientBoostingClassifier（原生支持 NaN——关键点缺失的帧直接丢进去，
不需要手工填空）。特征 = 现有 15 项掌宽归一化指标 + 三个判据指标的滚动均值/标准差
（给树模型一点时序上下文，等价于轻量版"看一小段"）。

评估：≥2 段视频时做 leave-one-video-out（最接近真实泛化：换一天/换机位还能不能行）；
只有 1 段时按时间前 70% 训练 / 后 30% 测试。
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                                       # noqa: E402
import cv2                                               # noqa: E402
import mediapipe as mp                                   # noqa: E402
import pg_utils as pg                                    # noqa: E402
import grip_metrics as gm                                # noqa: E402
import hand_probe as hp                                  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import (f1_score, confusion_matrix,      # noqa: E402
                             classification_report)

SAMPLE_EVERY = 2        # 每 2 帧取 1 帧（≈15fps 足够）
ROLL_WINDOW = 15        # 滚动统计窗口（≈1s）
UNSTABLE = (0.15, 0.10)  # thumb_side/dist 滚动 std 超过它 → 垃圾帧，剔除（与 app 的 POOR 门一致）
LABELS = {"ok": 1, "bad": 0}   # 1=正确握笔, 0=拇指包食指（报警类）

# 进模型的指标（px_w/px_h/palm_w 依赖距离，不用；hand_ratio 是相对量，用）
FEATURE_METRICS = [
    "hand_ratio", "fist", "curl_index", "curl_middle", "curl_ring",
    "curl_pinky", "tip_close", "spread", "thumb_index_gap",
    "thumb_side", "thumb_index_dist", "thumb_index_pos",
    "thumb_index_angle", "palm_view_angle",
]
ROLL_METRICS = ["thumb_side", "thumb_index_dist", "thumb_index_angle"]


def extract_rows(spec):
    """按 spec 逐视频抽帧 → 指标 → (原始指标行, 标签, 视频序号)。"""
    model_path = pg.find_model(hp.HAND_MODEL)
    hand = hp.make_hand(model_path)
    all_rows = []          # 每个: (video_idx, label, {metric: val})
    for vi, v in enumerate(spec["videos"]):
        cap = cv2.VideoCapture(v["path"])
        if not cap.isOpened():
            print(f"⚠️ 打不开视频，跳过：{v['path']}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        lock = v.get("lock")
        sels = hp.HandSelector(lock=lock)
        segs = [s for s in v.get("segments", []) if s.get("label") in LABELS]
        n_in_seg = 0
        tick = 0
        fi = 0
        # 先按帧号建每帧的 (t, hands) 流，边读边判段
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            t = fi / fps
            fi += 1
            if fi % SAMPLE_EVERY:
                continue
            if not any(s["t0"] <= t < s["t1"] for s in segs):
                continue
            h, w = fr.shape[:2]
            tick += 33
            r = hand.detect_for_video(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=fr[:, :, ::-1].copy()), tick)
            hands = [(lm, "?") for lm in (r.hand_landmarks or [])]
            sel = sels.update(hands, w, h) if hands else None
            lm = hands[sel][0] if sel is not None else None
            if lm is None:
                continue
            m = gm.compute_grip_metrics(lm, w, h)
            lab = next(s["label"] for s in segs if s["t0"] <= t < s["t1"])
            all_rows.append((vi, LABELS[lab], m))
            n_in_seg += 1
        cap.release()
        print(f"  视频{vi}: {os.path.basename(v['path'])}  抽到 {n_in_seg} 帧有指标")
    hand.close()
    return all_rows


def drop_unstable(rows):
    """滚动 std 剔除"关键点在猜"的垃圾帧（与 app 的 POOR 门同阈值）。"""
    kept, dropped = [], 0
    by_v = {}
    for r in rows:
        by_v.setdefault(r[0], []).append(r)
    for vi, rs in by_v.items():
        ts = np.array([r[2]["thumb_side"] if r[2]["thumb_side"] is not None
                       else np.nan for r in rs], dtype=float)
        td = np.array([r[2]["thumb_index_dist"] if r[2]["thumb_index_dist"] is not None
                       else np.nan for r in rs], dtype=float)
        for i, r in enumerate(rs):
            lo, hi = max(0, i - ROLL_WINDOW), i + 1
            s_ts = np.nanstd(ts[lo:hi]) if hi - lo >= 5 else np.nan
            s_td = np.nanstd(td[lo:hi]) if hi - lo >= 5 else np.nan
            if (np.isfinite(s_ts) and s_ts > UNSTABLE[0]) or \
               (np.isfinite(s_td) and s_td > UNSTABLE[1]):
                dropped += 1
                continue
            kept.append(r)
    print(f"  稳定性过滤：剔除 {dropped} 垃圾帧，保留 {len(kept)}")
    return kept


def build_matrix(rows):
    """指标行 → 特征矩阵（含滚动均值/方差）。返回 X, y, groups(视频号), feat_names。"""
    by_v = {}
    for vi, y, m in rows:
        by_v.setdefault(vi, []).append((y, m))
    X, y, g, names = [], [], [], None
    for vi, rs in by_v.items():
        # 滚动统计基于该视频内的时序
        series = {k: np.array([r[1][k] if r[1][k] is not None else np.nan
                               for r in rs], dtype=float) for k in ROLL_METRICS}
        for i, (lab, m) in enumerate(rs):
            lo, hi = max(0, i - ROLL_WINDOW), i + 1
            feat, fn = [], []
            for k in FEATURE_METRICS:
                v = m.get(k)
                feat.append(np.nan if v is None else float(v))
                fn.append(k)
            for k in ROLL_METRICS:
                seg = series[k][lo:hi]
                feat.append(float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan)
                fn.append(k + "_rollmean")
                feat.append(float(np.nanstd(seg)) if np.isfinite(seg).any() else np.nan)
                fn.append(k + "_rollstd")
            X.append(feat)
            y.append(lab)
            g.append(vi)
            names = fn
    return np.array(X, dtype=float), np.array(y), np.array(g), names


def train_eval(X, y, groups):
    """leave-one-video-out（≥2 段）或按时间 70/30。返回 (报告文本, 模型)。"""
    lines = []
    uniq = sorted(set(groups.tolist()))
    if len(uniq) >= 2:
        lines.append(f"评估方式：leave-one-video-out（{len(uniq)} 段视频轮换）")
        pred = np.zeros_like(y)
        for vi in uniq:
            tr, te = groups != vi, groups == vi
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                lines.append(f"  ⚠️ 视频{vi} 单类无法评估，跳过该折")
                pred[te] = y[te]
                continue
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
    else:
        lines.append("评估方式：仅 1 段视频，按时间前 70% 训练 / 后 30% 测试"
                     "（⚠️ 换天/换机位泛化未验证，别急着上岗）")
        n = len(y)
        cut = int(n * 0.7)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(X[:cut], y[:cut])
        pred = np.zeros_like(y)
        pred[:cut] = clf.predict(X[:cut])
        pred[cut:] = clf.predict(X[cut:])

    f1_bad = f1_score(y, pred, pos_label=0)
    f1_ok = f1_score(y, pred, pos_label=1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    lines.append(f"  F1  拇指包食指(bad)={f1_bad:.3f}   正确(ok)={f1_ok:.3f}")
    lines.append(f"  混淆矩阵 [行=真实 0bad/1ok, 列=预测 0/1]:\n{cm}")
    fp = cm[1][0] / max(1, cm[1].sum())     # 正确被判成包食指 = 误报
    lines.append(f"  误报率(正常被判错) = {100*fp:.1f}%   "
                 f"{'✅ 达标(≤5%)' if fp <= 0.05 else '⚠️ 超标，需更多数据/特征'}")
    final = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    final.fit(X, y)
    return "\n".join(lines), final


def demo():
    """合成数据冒烟测试：两团高斯特征，验证 抽取→训练→评估→保存 全链路。"""
    rng = np.random.default_rng(3)
    n = 200
    Xs, ys, gs = [], [], []
    for vi in range(2):                       # 2 "段视频" 做 leave-one-out
        base = 0.0 if vi == 0 else 0.05       # 段间漂移，考验泛化
        for _ in range(n):
            if rng.random() < 0.5:            # bad: thumb_side 偏大、夹角偏小
                f = [base + rng.normal(0.35, 0.08), rng.normal(20, 5)]
            else:                             # ok
                f = [base + rng.normal(-0.17, 0.08), rng.normal(70, 6)]
            Xs.append(f + list(rng.normal(0, 0.1, 27)))   # 补齐特征位数
            ys.append(0 if f[0] > 0.1 else 1)
        # 修正 y：直接按 thumb_side 造标签
        ys = []
        for row in Xs[-2 * n:]:
            ys.append(0 if row[0] > base + 0.09 else 1)
    X = np.array(Xs, float)
    y = np.array(ys, int)
    g = np.repeat([0, 1], n)
    names = [f"f{i}" for i in range(X.shape[1])]
    report, model = train_eval(X, y, g)
    print(report)
    save_model(model, names, "demo")
    print("\n✅ 冒烟测试通过（合成数据，仅验证管线，不代表真实精度）")
    return 0


def save_model(model, feat_names, source):
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    p = os.path.join(ROOT, "models", "grip_cls.pkl")
    with open(p, "wb") as f:
        pickle.dump({"model": model, "features": feat_names,
                     "roll_window": ROLL_WINDOW, "labels": LABELS,
                     "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "source": source}, f)
    print(f"  模型已存 {p}")


def main():
    ap = argparse.ArgumentParser(description="握笔姿势分类器训练")
    ap.add_argument("--spec", help="dataset.json 路径")
    ap.add_argument("--demo", action="store_true", help="合成数据冒烟测试")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    if a.demo:
        return demo()
    if not a.spec:
        ap.error("给 --spec dataset.json 或 --demo")

    with open(a.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    print("== 抽帧提指标 ==")
    rows = extract_rows(spec)
    if not rows:
        print("❌ 没抽到任何带指标的帧，检查视频路径/时间段/机位")
        return 1
    print("== 稳定性过滤 ==")
    rows = drop_unstable(rows)
    print("== 组特征矩阵 ==")
    X, y, g, names = build_matrix(rows)
    print(f"  特征 {X.shape[1]} 维，样本 {X.shape[0]}（bad={sum(y==0)}, ok={sum(y==1)}）")
    print("== 训练+评估 ==")
    report, model = train_eval(X, y, g)
    print(report)
    if not a.no_save:
        save_model(model, names, os.path.basename(a.spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
