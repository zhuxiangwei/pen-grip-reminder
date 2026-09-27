#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔姿势 · 阈值标定工具（连接"录样本"和"出判据"）
================================================

手里有了真实样本（`app.py` 里按 w 存下的 `hand_probe/samples_*.json`，或
`hand_probe.py --replay` 看的那个文件），这个脚本帮你把"拇指包食指"这道判据
从**人眼读对照表**升级成**可落地的阈值规则**：

  1. 对每個候选指标，按"越该报警值越大"统一方向，自动找最佳阈值
     （复用 pg.suggest_threshold，误报权重 1.5 —— 叫太勤孩子就不用了）；
  2. 报告每个指标的 margin（可分性）和单指标混淆（误报/漏报）；
  3. 给出一个 OR 组合规则（任一指标越线即报警），并报告组合后的混淆；
  4. 把规则写进 pen_grip_config.json 的 "classifier" 段，
     app.py 之后就能在界面上给出"⚠️ 拇指包食指 / ✅ 正常"的实时判定。

⚠️ 关于符号：机位从"侧视"换成"正前方俯视"后，图像里拇指在哪一侧可能翻转。
   本工具对每个指标都用**数据本身**判断方向（看哪一类均值更大），
   所以不需要你手工改符号 —— 直接拿真实样本来跑就行。

用法
----
   python tools/calibrate.py --samples hand_probe/samples_20260927-xxxx.json
   python tools/calibrate.py --samples <file> --out pen_grip_config.json --no-write
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                                          # noqa: E402
import pg_utils as pg                                      # noqa: E402
import grip_metrics as gm                                  # noqa: E402

# 候选指标 + 它们的"自然方向"：
#   "+" = 值越大越像"拇指包食指"（拇指横向越过量）
#   "-" = 值越小越像"拇指包食指"（夹角塌掉 / 真的搭到食指上）
CANDIDATES = [
    ("thumb_side", "+", "拇指横向越过量"),
    ("thumb_index_angle", "-", "拇指食指夹角"),
    ("thumb_index_dist", "-", "拇指尖到食指的距离"),
    ("thumb_index_gap", "-", "拇指食指间距"),
]
POS = "2"          # 类别键：拇指包食指（应该报警）
NEG = ("1", "3")   # 不该报警：正确握笔 + 其他错误


def _col(data, cls, metric):
    return [s[metric] for s in data.get(cls, [])
            if isinstance(s, dict) and s.get(metric) is not None
            and not (isinstance(s[metric], float) and s[metric] != s[metric])]


def _margin(wrap, ok):
    if len(wrap) < 2 or len(ok) < 2:
        return None
    mw, sw = float(np.mean(wrap)), float(np.std(wrap))
    mo, so = float(np.mean(ok)), float(np.std(ok))
    pooled = max(1e-9, ((sw ** 2 + so ** 2) / 2) ** 0.5)
    return abs(mw - mo) / pooled


def _trips(v, direction, th):
    if v is None:
        return False
    return (v > th) if direction == "+" else (v < th)


def calibrate(data):
    rows = []
    for metric, natural, cn in CANDIDATES:
        wrap = _col(data, POS, metric)
        ok = []
        for c in NEG:
            ok += _col(data, c, metric)
        if len(wrap) < 2 or len(ok) < 2:
            rows.append((metric, cn, natural, None, None, None, len(wrap), len(ok)))
            continue
        if natural == "+":
            should, shouldnt = wrap, ok
            direction = "+"
        else:
            should = [-v for v in wrap]
            shouldnt = [-v for v in ok]
            direction = "-"
        r = pg.suggest_threshold(should, shouldnt)
        if r is None:
            rows.append((metric, cn, natural, None, None, None, len(wrap), len(ok)))
            continue
        th = r["suggested"] if direction == "+" else -r["suggested"]
        margin = _margin(wrap, ok)
        rows.append((metric, cn, direction, th, margin,
                     (r["fp_at_suggested"], r["fn_at_suggested"]),
                     len(wrap), len(ok)))
    return rows


def eval_or_rule(data, chosen):
    """对一组 (metric, direction, th) 评估 OR 组合规则的混淆。"""
    def trips_sample(s):
        return any(_trips(s.get(m), d, t) for (m, d, t) in chosen)
    fp = fn = 0
    for c in NEG:
        for s in data.get(c, []):
            if isinstance(s, dict) and trips_sample(s):
                fp += 1
    for s in data.get(POS, []):
        if isinstance(s, dict) and not trips_sample(s):
            fn += 1
    return fp, fn


def main():
    ap = argparse.ArgumentParser(description="握笔姿势阈值标定")
    ap.add_argument("--samples", required=True, help="samples_*.json 路径")
    ap.add_argument("--out", default=os.path.join(ROOT, "pen_grip_config.json"))
    ap.add_argument("--no-write", action="store_true", help="只报告，不写配置")
    args = ap.parse_args()

    with open(args.samples, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["data"]

    rows = calibrate(data)
    valid = [r for r in rows if r[3] is not None]
    valid.sort(key=lambda r: (r[4] if r[4] is not None else -1), reverse=True)

    print("=" * 70)
    print("阈值标定报告（基于真实样本）")
    print("=" * 70)
    hdr = f"{'指标':<20}{'方向':>5}{'阈值':>10}{'margin':>9}{'误报/漏报':>12}  n(包/正常)"
    print(hdr)
    print("-" * 70)
    for metric, cn, direction, th, margin, err, nw, no in rows:
        if th is None:
            print(f"{cn:<20}{'-':>5}{'样本不足':>10}{'—':>9}{'—':>12}  {nw}/{no}")
            continue
        fp, fn = err
        print(f"{cn:<20}{direction:>5}{th:>10.3f}{margin:>9.2f}"
              f"{str(fp)+'/'+str(fn):>12}  {nw}/{no}")

    if not valid:
        print("\n⚠️ 没有足够样本做标定。每个类别至少录 2 条再跑。")
        return 1

    best = valid[0]
    best_metric, best_cn, best_dir, best_th, best_margin = best[:5]
    best_fp, best_fn = best[5]
    print("\n→ 单指标最佳：" + best_cn +
          f"（margin={best_margin:.2f}，阈值 {best_dir} {best_th:.3f}，"
          f"误报 {best_fp}/漏报 {best_fn}）")

    # OR 组合：仅在主指标仍有漏报时，叠加一个 margin 够大的次佳指标来补漏
    chosen = [(best_metric, best_dir, best_th)]
    if best_fn > 0 and len(valid) > 1 and valid[1][4] and valid[1][4] >= 1.0:
        m2, cn2, d2, t2, _ = valid[1][:5]
        chosen.append((m2, d2, t2))
        print(f"   叠加次佳 {cn2} 做 OR 组合，降低漏报")

    fp, fn = eval_or_rule(data, chosen)
    n_pos = len(data.get(POS, []))
    n_neg = sum(len(data.get(c, [])) for c in NEG)
    print(f"\nOR 组合规则混淆：误报 {fp}/{n_neg}，漏报 {fn}/{n_pos}")
    if fp + fn == 0:
        print("   ✅ 这套样本上完全分开。换角度/光照/手型再录一轮确认后再上岗。")
    else:
        print("   ⚠️ 还有分错的样本。可多录、或加第 3 个指标、或退回握笔器（docs/01-plan §6）。")

    if args.no_write:
        print("\n（--no-write，未写入配置）")
        return 0

    classifier = {
        "metrics": {m: {"dir": d, "th": round(float(t), 4)} for (m, d, t) in chosen},
        "rule": "or",
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": {POS: n_pos, **{c: len(data.get(c, [])) for c in NEG}},
    }
    # 读原配置、只更新 classifier 段、写回
    cfg = {}
    if os.path.isfile(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    cfg["classifier"] = classifier
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已写入 {args.out} 的 classifier 段。")
    print("   重启 app.py 后，界面右下会出现「实时判定：⚠️ 拇指包食指 / ✅ 正常」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
