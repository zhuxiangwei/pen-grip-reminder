#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔指标 · 离线自检
====================

用合成手验证 grip_metrics 算得对不对。**不需要摄像头、不需要模型、不需要孩子配合。**

    python repos\\p1-pen-grip\\tests\\test_grip_offline.py

⚠️ 这个文件**不能**证明"指标能识别握笔姿势对不对" —— 那要看真实数据（见 docs/04）。
   它能证明的是：
     · 指标算得对（方向对、不变性成立、退化输入不崩）
     · 宽高比那个坑被真正避开了
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grip_metrics as gm   # noqa: E402

PASS = FAIL = 0
FAILED = []


def case(name, got, want):
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}\n        期望 {want!r}\n        实际 {got!r}")


def close(name, got, want, tol=1e-6):
    global PASS, FAIL
    ok = got is not None and abs(got - want) <= tol
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}\n        期望 {want!r} ± {tol}\n        实际 {got!r}")


def ok_(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}  {extra}")


def metrics(**kw):
    lm, w, h = gm.synthetic_landmarks(**kw)
    return gm.compute_grip_metrics(lm, w, h)


_BASE_PX_W = None


def scale_for_px(target_px):
    """算出"让手在画面里宽约 target_px"所需的缩放。

    ⚠️ 别硬编码 scale —— 合成手的长度单位约等于掌宽（≈1.0），
       写 scale=3.0 得到的其实是 5px 宽的手。一开始就是这么错的，
       导致门控测试全线误判。
    """
    global _BASE_PX_W
    if _BASE_PX_W is None:
        _BASE_PX_W = metrics(scale=1.0)["px_w"]
    return target_px / _BASE_PX_W


# ---------------------------------------------------------------- A. 宽高比

def test_aspect_ratio():
    print("\n[A] 宽高比校正（坐姿项目踩过的同一个坑，这里做回归锁）")

    # 在**像素空间**里刚性旋转整只手。刚性旋转不改变任何内禀角度，
    # 所以角度类指标必须一模一样。
    base = metrics(rot_deg=0.0)
    for deg in (20.0, 35.0, -50.0, 90.0):
        r = metrics(rot_deg=deg)
        close(f"A1 旋转 {deg:>5.0f}° 后拇指食指夹角不变",
              r["thumb_index_angle"], base["thumb_index_angle"], 1e-6)

    # 距离比值类也要不变（它们都除以掌宽，刚性变换下比值恒等）
    r = metrics(rot_deg=35.0)
    for k in ("thumb_index_gap", "thumb_side", "fist", "tip_close", "spread"):
        close(f"A2 旋转 35° 后 {k} 不变", r[k], base[k], 1e-9)

    # ---- 让这个锁"有牙齿"：验证按错误方式算（不转像素）结果确实会不同 ----
    # 如果不先把归一化坐标转成像素，1280x720 的宽高比会把角度压扁。
    w, h = 1280, 720
    pts = gm.synthetic_hand(rot_deg=0.0)
    lm = gm.to_normalized(pts, w, h)

    def angle_in_normalized_space(landmarks):
        """故意错的算法：直接拿归一化坐标当平面坐标算角度。"""
        def p(i):
            return (landmarks[i].x, landmarks[i].y)

        def ang(a, v, b):
            v1 = (a[0] - v[0], a[1] - v[1])
            v2 = (b[0] - v[0], b[1] - v[1])
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            return math.degrees(math.acos(c))
        return ang(p(gm.THUMB_MCP), p(gm.INDEX_MCP), p(gm.INDEX_PIP))

    wrong0 = angle_in_normalized_space(lm)
    right0 = gm.compute_grip_metrics(lm, w, h)["thumb_index_angle"]
    ok_("A3 正确算法与错误算法在未旋转时就已不同（说明宽高比确实有影响）",
        abs(wrong0 - right0) > 1.0, f"错误={wrong0:.3f} 正确={right0:.3f}")

    # 旋转后：正确算法不变，错误算法会变 —— 这就是那个坑的真身
    pts_r = gm.synthetic_hand(rot_deg=35.0)
    lm_r = gm.to_normalized(pts_r, w, h)
    wrong1 = angle_in_normalized_space(lm_r)
    right1 = gm.compute_grip_metrics(lm_r, w, h)["thumb_index_angle"]
    ok_("A4 旋转后：错误算法漂移 >1°，正确算法纹丝不动",
        abs(wrong1 - wrong0) > 1.0 and abs(right1 - right0) < 1e-6,
        f"错误漂移={abs(wrong1-wrong0):.3f}° 正确漂移={abs(right1-right0):.2e}°")

    # 换个宽高比，正确算法依然不变（因为它先转回了像素空间）
    lm2 = gm.to_normalized(pts, 640, 360)
    close("A5 换分辨率（640x360）后指标一致",
          gm.compute_grip_metrics(lm2, 640, 360)["thumb_index_angle"],
          right0, 1e-6)


# ---------------------------------------------------------------- B. 不变性

def test_invariance():
    print("\n[B] 平移 / 缩放不变性（指标必须是手部内禀的）")

    base = metrics()
    moved = metrics(dx=400.0, dy=-250.0)
    scaled = metrics(scale=2.7)
    for k in ("fist", "tip_close", "spread", "thumb_index_gap", "thumb_side",
              "thumb_index_dist", "thumb_index_pos", "thumb_index_angle"):
        close(f"B1 平移后 {k} 不变", moved[k], base[k], 1e-9)
        close(f"B2 放大 2.7 倍后 {k} 不变", scaled[k], base[k], 1e-9)

    # 但像素尺寸类**应该**变（那是用来做视角门控的）
    ok_("B3 放大后 px_w 应该变大（像素尺寸是故意的非不变量）",
        scaled["px_w"] > base["px_w"] * 2.0,
        f"{base['px_w']:.0f} -> {scaled['px_w']:.0f}")


# ---------------------------------------------------------------- C. 方向正确性

def test_directions():
    print("\n[C] 每个指标的方向（「变大/变小」是否符合物理直觉）")

    # 食指越来越蜷 -> curl_index 单调下降
    vals = [metrics(curl=(c, 0.7, 0.75, 0.8))["curl_index"]
            for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ok_("C1 食指弯曲参数增大 -> curl_index 单调下降",
        all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
        f"{['%.3f' % v for v in vals]}")
    ok_("C2 完全伸直时 curl_index ≈ 1", abs(vals[0] - 1.0) < 0.02, f"{vals[0]:.4f}")
    ok_("C3 蜷到底时 curl_index 明显变小", vals[-1] < 0.55, f"{vals[-1]:.4f}")

    # 四指全蜷 -> fist 和 tip_close 都下降
    open_h = metrics(curl=(0.05, 0.05, 0.05, 0.05))
    fist_h = metrics(curl=(1.0, 1.0, 1.0, 1.0))
    ok_("C4 四指全蜷 -> fist 下降", fist_h["fist"] < open_h["fist"] * 0.75,
        f"{open_h['fist']:.3f} -> {fist_h['fist']:.3f}")
    ok_("C5 四指全蜷 -> tip_close 下降（指尖缩回掌心）",
        fist_h["tip_close"] < open_h["tip_close"],
        f"{open_h['tip_close']:.3f} -> {fist_h['tip_close']:.3f}")

    # 拇指往小指侧摆 -> thumb_side 增大（这是「拇指包食指」的假设判据）
    sides = [metrics(thumb_side=t)["thumb_side"] for t in (-0.5, -0.2, 0.0, 0.3, 0.6)]
    ok_("C6 拇指往小指侧摆 -> thumb_side 单调增大",
        all(sides[i] < sides[i + 1] for i in range(len(sides) - 1)),
        f"{['%.3f' % s for s in sides]}")

    # 拇指贴着食指 vs 远离
    near = metrics(thumb_spread=0.6)["thumb_index_gap"]
    far = metrics(thumb_spread=1.4)["thumb_index_gap"]
    ok_("C7 拇指远离食指 -> thumb_index_gap 变大", far > near,
        f"{near:.3f} -> {far:.3f}")

    # thumb_index_pos：**食指伸直时**，拇指沿手指方向伸出 -> 落点单调后移。
    # ⚠️ 两条几何事实必须写进断言：
    #   1) 食指一弯，折线先上再折回来，"最近点"会沿链条倒退，
    #      pos 不再代表"往指尖方向" —— 所以必须在 curl=0 下测。
    #      （这不是 bug，是"到折线最近点"这个定义的固有性质，
    #        因此 thumb_index_pos 只当参考信息，不当判据。）
    #   2) 拇指越过食指尖之后，最近点恒为食指尖，pos 钳在 1.0 —— 会饱和。
    #      所以是"单调不减 + 有实际上升"，不是严格递增。
    vals = [metrics(curl=(0.0, 0.0, 0.0, 0.0),
                    thumb_along_shift=s)["thumb_index_pos"]
            for s in (-0.2, 0.0, 0.2, 0.4)]
    ok_("C8 食指伸直时，拇指伸出手 -> thumb_index_pos 单调不减且落到饱和值 1.0",
        all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        and vals[-1] > vals[0] and abs(vals[-1] - 1.0) < 1e-9,
        f"{['%.3f' % v for v in vals]}")

    # ⚠️ 这条是对一个真实 bug 的回归锁：
    #    最初用「拇指尖沿食指轴的投影」衡量位置，食指一弯那条轴就转向侧面，
    #    投影会变成负数（实测 -0.166）。改成"到折线的最近点弧长"后
    #    天然落在 [0,1]，不可能为负。弯曲参数扫一遍确认。
    allpos = []
    for c in (0.0, 0.3, 0.6, 1.0):
        m = metrics(curl=(c, c, c, c))
        allpos.append(m["thumb_index_pos"])
    ok_("C9 thumb_index_pos 在食指任何弯曲度下都落在 [0,1]（回归锁）",
        all(v is not None and 0.0 <= v <= 1.0 for v in allpos),
        f"{['%.3f' % v for v in allpos]}")

    # 两个参数应当基本解耦：固定 thumb_side 只调 thumb_along_shift，
    # thumb_side 只许有很小的连带变化。
    # ⚠️ 不能要求**完全**不变：thumb_side 的投影轴是「食指根→小指根」，
    #    这个轴并不垂直于手指方向（合成手里它带 -0.14 的 y 分量，
    #    真实手掌也不是严格垂直）。所以沿 +y 移动拇指必然轻微改变投影 ——
    #    这是几何事实，不是 bug。要求的是"连带上限"。
    a = metrics(thumb_side=0.5, thumb_along_shift=0.0)
    b = metrics(thumb_side=0.5, thumb_along_shift=0.4)
    d_side = abs(a["thumb_side"] - b["thumb_side"])
    ok_("C10 调 thumb_along_shift 对 thumb_side 的连带影响很小（<0.15，且远小于直接效果 0.4）",
        d_side < 0.15,
        f"Δside={d_side:.4f}  side {a['thumb_side']:.4f}->{b['thumb_side']:.4f}")

    # 拇指抬离食指 -> 到食指的距离变大
    near = metrics(thumb_spread=1.0)["thumb_index_dist"]
    far = metrics(thumb_spread=1.8)["thumb_index_dist"]
    ok_("C11 拇指抬离食指 -> thumb_index_dist 变大", far > near,
        f"{near:.3f} -> {far:.3f}")


# ---------------------------------------------------------------- D. 退化输入

def test_degenerate():
    print("\n[D] 退化输入（不许崩、不许给 NaN）")

    # 21 个点全在同一个位置 -> 掌宽为 0
    class P:
        __slots__ = ("x", "y")

        def __init__(self, x=0.5, y=0.5):
            self.x, self.y = x, y

    m = gm.compute_grip_metrics([P() for _ in range(21)], 1280, 720)
    case("D1 所有点重合 -> 不崩，比值为 None", m["fist"], None)
    case("D2 掌宽为 0 时 palm_w 记 0", m["palm_w"], 0.0)
    case("D3 px_w/px_h 仍可算（都是 0）", (m["px_w"], m["px_h"]), (0.0, 0.0))

    # 含 NaN 的点
    bad = [P(float("nan"), 0.5) for _ in range(21)]
    m2 = gm.compute_grip_metrics(bad, 1280, 720)
    ok_("D4 含 NaN 不抛异常", isinstance(m2, dict))

    # 正常输入里不许出现 NaN
    m3 = metrics()
    nones = [k for k, v in m3.items() if v is None]
    ok_("D5 正常合成手应当所有指标都算得出来", not nones, f"None 项: {nones}")
    nans = [k for k, v in m3.items() if isinstance(v, float) and math.isnan(v)]
    ok_("D6 正常合成手里没有 NaN", not nans, f"NaN 项: {nans}")


# ---------------------------------------------------------------- E. 视角门控

def test_view_gate():
    print("\n[E] 视角质量门控（手太小就不许下结论）")

    small = metrics(scale=scale_for_px(60))
    mid = metrics(scale=scale_for_px(160))
    big = metrics(scale=scale_for_px(320))
    lv_s, _ = gm.view_quality(small)
    lv_m, _ = gm.view_quality(mid)
    lv_b, _ = gm.view_quality(big)
    print(f"        小手: px_w={small['px_w']:.0f}  -> {lv_s}")
    print(f"        中手: px_w={mid['px_w']:.0f}  -> {lv_m}")
    print(f"        大手: px_w={big['px_w']:.0f}  -> {lv_b}")

    case("E1 手太小时门控判 bad", lv_s, "bad")
    case("E2 中等偏小时判 marginal", lv_m, "marginal")
    case("E3 手够大时门控判 ok", lv_b, "ok")
    case("E4 完全没数据时判 bad", gm.view_quality(None)[0], "bad")
    case("E5 掌宽缺失时判 bad", gm.view_quality({"palm_w": None})[0], "bad")

    # 抖动太大 -> 即使是够大的手也降级
    lv_j, _ = gm.view_quality(big, stability_px=999.0)
    case("E6 抖动过大时把 ok 降级为 marginal", lv_j, "marginal")


# ---------------------------------------------------------------- 入口

def main():
    print("=" * 62)
    print("握笔指标 · 离线自检（合成手，无需摄像头/模型/真人）")
    print("=" * 62)
    test_aspect_ratio()
    test_invariance()
    test_directions()
    test_degenerate()
    test_view_gate()

    print("\n" + "=" * 62)
    if FAIL:
        print(f"结果：{PASS}/{PASS + FAIL} 通过，{FAIL} 个失败 ❌")
        for n in FAILED:
            print(f"  - {n}")
    else:
        print(f"结果：{PASS}/{PASS + FAIL} 全部通过 ✅")
    print("=" * 62)
    print("\n⚠️ 这份自检只证明「指标算得对」，**不能**证明「指标能识别握笔姿势对不对」。")
    print("   后者必须用真实数据实测 —— 用 hand_probe.py 做对照。")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
