#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
握笔姿势检测 · 窗口应用
=======================

**为什么是 tkinter 而不是 PySide6**
   PySide6 / PyQt 都没装（实测三选全 NO），而 **tkinter 是 Python 自带的**。
   这个项目要装的东西已经不少（opencv + mediapipe + matplotlib），
   窗口本身没必要再拉一个 100MB+ 的 Qt。
   **零新增依赖**对「双击就能用」这件事很重要。

**核心原则：这里不重新实现检测逻辑**
   所有指标、视角门控、手别选择、阈值对照**全部复用** `grip_metrics` /
   `hand_probe` 里已有的函数。窗口只是把它们套了个界面。
   两份实现会随时间漂移 —— 改一处忘另一处是最常见的坑。

**窗口长什么样**

    ┌─────────────────────────────┬───────────────────────┐
    │                             │  机位体检             │
    │        摄像头画面            │   手宽 / 视角角 / 稳定性│
    │      （含手部特写标注）        │                       │
    │                             │  样本计数             │
    │                             │   正确 / 包食指 / 其他 │
    │                             │                       │
    │                             │  [1] 正确握笔         │
    │                             │  [2] 拇指包食指        │
    │                             │  [3] 其他错误          │
    │                             │                       │
    ├─────────────────────────────┤  [出对照表]           │
    │ [ 开始 ] [ 冻结 ] [ ↵ 出表 ] │  [保存样本]           │
    └─────────────────────────────┴───────────────────────┘

**按键**（和命令行版一致，肌肉记忆可以复用）
   1 / 2 / 3   记录当前帧为对应类别
   z           开关拇指-食指放大镜
   [ / ]       锁定被测手 = 画面左半 / 右半（看不见"左右手"，只看画面位置）
   \\           取消锁定，回到自动（按运动量猜）
   space       冻结 / 继续画面
   p           出对照表（也等价于点按钮）
   w           保存样本到磁盘
   q / Esc     退出

用法
----
    python app.py
    python app.py --camera 1
"""
import argparse
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT)

import pg_utils as pg                                   # noqa: E402
import grip_metrics as gm                               # noqa: E402
import hand_probe as hp                                 # noqa: E402
import mediapipe as mp                                  # noqa: E402

# 画面配色（贴近深色主题，和命令行版一致）
BG = "#16181d"
FG = "#e8eaed"
DIM = "#8b93a1"
ACC = "#5aa9ff"
OKC = "#4ec96b"
WARNC = "#e0a13a"
BADC = "#e05c5c"


class GripApp:
    def __init__(self, root, cam_idx=None, backend=None):
        self.root = root
        self.root.title("握笔姿势检测")
        self.root.configure(bg=BG)
        self.root.minsize(1040, 620)

        self.cam_idx = cam_idx
        self.backend = backend

        # ---- 引擎（全部复用现有模块，不重写）----
        self.cap = None
        self.hand = None
        self.sel = hp.HandSelector()
        self.rec = hp.Recorder()
        self.cfg = hp.load_cfg()
        if self.cfg.get("hand_lock"):
            self.sel.lock = self.cfg["hand_lock"]

        self.zoom = True
        self.frozen = False
        self.running = False
        self.last = None                    # (m, quality, sel, note)
        self.err = None
        self.frame_size = (0, 0)
        self.tick = 0

        self._build_ui()
        self._bind_keys()
        self._start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ 界面
    def _build_ui(self):
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=10, pady=10)

        # 左边：画面
        left = tk.Frame(wrap, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#0c0e12", highlightthickness=1,
                                highlightbackground="#2a2f38")
        self.canvas.pack(fill="both", expand=True)
        self._canvas_item = self.canvas.create_image(0, 0, anchor="nw")

        self.status = tk.Label(left, text="正在打开摄像头…", bg=BG, fg=DIM,
                               anchor="w", font=("Consolas", 10))
        self.status.pack(fill="x", pady=(6, 0))

        # 右边：信息面板 + 按钮
        right = tk.Frame(wrap, bg=BG, width=330)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        self.quality_box = self._panel(right, "机位体检")
        self.count_box = self._panel(right, "样本计数")
        self.help_box = self._panel(right, "按键")

        tk.Label(right, text="记录样本", bg=BG, fg=FG, anchor="w",
                 font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", pady=(10, 4))

        btns = tk.Frame(right, bg=BG)
        btns.pack(fill="x")
        specs = [("1", "正确握笔", OKC), ("2", "拇指包食指", BADC),
                 ("3", "其他错误", WARNC)]
        for key, name, col in specs:
            b = tk.Button(btns, text=f"[{key}] {name}", bg="#232830", fg=col,
                          activebackground="#2e3540", activeforeground=col,
                          relief="flat", anchor="w", cursor="hand2",
                          font=("Microsoft YaHei UI", 10),
                          command=lambda k=key: self.record(k))
            b.pack(fill="x", pady=2)

        tk.Frame(right, bg="#2a2f38", height=1).pack(fill="x", pady=10)
        for text, fn in (("出对照表 (p)", self.compare),
                         ("保存样本 (w)", self.save_samples),
                         ("清空样本", self.clear_samples),
                         ("冻结画面 (空格)", self.toggle_freeze),
                         ("重新探测机位", self.reprobe)):
            b = tk.Button(right, text=text, bg="#232830", fg=FG,
                          activebackground="#2e3540", relief="flat",
                          anchor="w", cursor="hand2",
                          font=("Microsoft YaHei UI", 10), command=fn)
            b.pack(fill="x", pady=2)

        self.report = scrolledtext.ScrolledText(
            right, height=10, bg="#0c0e12", fg=FG, insertbackground=FG,
            relief="flat", font=("Consolas", 9), wrap="word")
        self.report.pack(fill="both", expand=True, pady=(10, 0))
        self.report.insert("1.0", "对照表会显示在这里。\n\n"
                                  "先按 1/2 各录一些样本，再点「出对照表」。")
        self.report.configure(state="disabled")

    def _panel(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=ACC, anchor="w",
                 font=("Microsoft YaHei UI", 10, "bold")).pack(fill="x", pady=(8, 3))
        lab = tk.Label(parent, text="—", bg=BG, fg=FG, anchor="w", justify="left",
                       font=("Consolas", 9))
        lab.pack(fill="x")
        return lab

    def _bind_keys(self):
        self.root.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())
        self.canvas.focus_set()

    # ------------------------------------------------------------ 摄像头
    def _start_camera(self):
        def work():
            try:
                model = pg.find_model(hp.HAND_MODEL)
                self.hand = hp.make_hand(model)
                # 分辨率来自探测配置（tools/probe_camera.py 写）
                self.cap = pg.open_camera_auto(idx=self.cam_idx,
                                               backend=self.backend)
                f = pg.grab_frame(self.cap)
                if f is not None:
                    self.frame_size = (f.shape[1], f.shape[0])
                self.running = True
                self.root.after(0, self._loop)
            except SystemExit as e:
                self.err = str(e)
                self.root.after(0, self._show_error)
            except Exception as e:                       # noqa: BLE001
                self.err = f"{type(e).__name__}: {e}"
                self.root.after(0, self._show_error)

        threading.Thread(target=work, daemon=True).start()

    def _show_error(self):
        self.status.configure(text=f"❌ {self.err}", fg=BADC)
        self.canvas.delete(self._canvas_item)
        self.canvas.create_text(
            max(200, self.canvas.winfo_width() // 2),
            max(100, self.canvas.winfo_height() // 2),
            text="摄像头打不开\n\n" + (self.err or "")[:600],
            fill=BADC, font=("Microsoft YaHei UI", 11), width=520, justify="center")

    # ------------------------------------------------------------ 主循环
    def _loop(self):
        import cv2
        import numpy as np
        if not self.running:
            return
        try:
            frame = None if self.frozen else pg.grab_frame(self.cap)
            if frame is None and not self.frozen:
                self.root.after(30, self._loop)
                return
            if frame is None:
                frame = self._last_frame
            else:
                self._last_frame = frame

            h, w = frame.shape[:2]
            rgb = frame[:, :, ::-1].copy()
            self.tick += 33

            r = self.hand.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                self.tick)

            hands = []
            for i, lm in enumerate(r.hand_landmarks or []):
                lab = "?"
                if r.handedness and i < len(r.handedness) and r.handedness[i]:
                    cat = r.handedness[i][0]
                    lab = (getattr(cat, "category_name", None)
                           or getattr(cat, "display_name", "?"))
                hands.append((lm, lab))

            sel = self.sel.update(hands, w, h)
            lm = hands[sel][0] if sel is not None else None
            m = gm.compute_grip_metrics(lm, w, h) if lm is not None else None
            self.rec.add_center(m, lm, w, h)
            self.rec.add_focus(m)
            quality, qmsg = gm.view_quality(m, self.rec.jitter)
            pv, pvmsg = gm.palm_view_verdict(m)

            vis = hp.draw(frame, lm, m, quality, self.rec.counts(),
                          self.rec.note, zoom=self.zoom, hands=hands, sel=sel,
                          sel_lines=self.sel.label(hands, sel, w, h),
                          lock=self.sel.lock, focus_std=self.rec.focus_std)

            if self.frozen:
                cv2.putText(vis, "FROZEN", (w // 2 - 70, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 200, 255), 3)

            # 缩放到画布尺寸显示
            cw = max(320, self.canvas.winfo_width())
            chh = max(240, self.canvas.winfo_height())
            sc = min(cw / w, chh / h)
            nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
            shown = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_AREA)
            self._photo = self._to_photo(shown)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, chh // 2, image=self._photo,
                                     anchor="center")

            self._update_panels(m, quality, qmsg, pv, pvmsg, sel)
            self.last = (m, quality, sel, self.rec.note)
        except Exception as e:                           # noqa: BLE001
            self.status.configure(text=f"⚠️ {type(e).__name__}: {e}", fg=BADC)
        self.root.after(15, self._loop)

    def _to_photo(self, bgr):
        """BGR ndarray -> tkinter PhotoImage（用 PPM 走内存，不落盘）。"""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        hdr = f"P6\n{w} {h}\n255\n".encode("ascii")
        return tk.PhotoImage(data=hdr + rgb.tobytes(), format="PPM")

    def _update_panels(self, m, quality, qmsg, pv, pvmsg, sel):
        if m is None:
            self.quality_box.configure(
                text=f"手宽        —\n视角判定    {'—'}\n"
                     f"手掌视角角  —\n指标稳定性  —\n\n"
                     f"⚠️ {'没检测到手' if not self.frozen else '（已冻结）'}",
                fg=WARNC)
        else:
            col = {"ok": OKC, "marginal": WARNC, "bad": BADC}[quality]
            s1, s2 = self.rec.focus_std
            if s1 is None:
                stab, scol = "—（样本不足）", DIM
            else:
                if s1 < 0.08 and s2 < 0.05:
                    stab, scol = f"GOOD  ({s1:.3f} / {s2:.3f})", OKC
                elif s1 < 0.15 and s2 < 0.10:
                    stab, scol = f"FAIR  ({s1:.3f} / {s2:.3f})", WARNC
                else:
                    stab, scol = f"POOR  ({s1:.3f} / {s2:.3f})", BADC
            pw = f"{m['px_w']:.0f}px"
            pwc = OKC if m["px_w"] >= gm.HAND_PX_OK else (
                WARNC if m["px_w"] >= gm.HAND_PX_MARGINAL else BADC)
            self.quality_box.configure(text="", fg=FG)
            self.quality_box.configure(
                text=(f"手宽        {pw}   (要 ≥{gm.HAND_PX_OK:.0f}px)\n"
                      f"视角判定    {quality.upper()}\n"
                      f"手掌视角角  {m['palm_view_angle']:.0f}°\n"
                      f"指标稳定性  {stab}\n\n"
                      f"手别        {self._sel_text(sel)}\n"
                      f"锁定        {self.sel.lock or '自动(按运动量)'}"))
            # 数字颜色单独用状态栏表达（tkinter Label 内不好混色）
            self.status.configure(
                text=f"{qmsg}    |    {pvmsg}", fg=col)

        # 稳定性的颜色用 count 面板的标题代替（简化）
        c = self.rec.counts()
        self.count_box.configure(
            text=(f"正确握笔     {c['1']:>4}\n"
                  f"拇指包食指   {c['2']:>4}\n"
                  f"其他错误     {c['3']:>4}\n"
                  f"合计         {sum(c.values()):>4}"),
            fg=FG)
        self.help_box.configure(
            text=("1/2/3 记录    z 放大镜\n"
                  "[ ] 锁定手    \\ 取消锁定\n"
                  "空格 冻结     p 出表\n"
                  "w 存样本      q 退出"),
            fg=DIM)

    def _sel_text(self, sel):
        if sel is None:
            return "⚠️ 分不出（先按 [ 或 ] 锁定）"
        return {0: "第1只（画面左）", 1: "第2只（画面右）"}.get(sel, f"第{sel+1}只")

    # ------------------------------------------------------------ 交互
    def on_key(self, e):
        k = e.keysym
        if k in ("1", "2", "3"):
            self.record(k)
        elif k in ("z", "Z"):
            self.zoom = not self.zoom
        elif k == "bracketleft":
            self._set_lock("screen_left")
        elif k == "bracketright":
            self._set_lock("screen_right")
        elif k == "backslash":
            self._set_lock(None)
        elif k == "space":
            self.toggle_freeze()
        elif k in ("p", "P"):
            self.compare()
        elif k in ("w", "W"):
            self.save_samples()
        elif k in ("q", "Q", "Escape"):
            self.on_close()
        return "break"

    def _set_lock(self, lock):
        self.sel.lock = lock
        self.cfg = hp.load_cfg()
        if lock:
            self.cfg["hand_lock"] = lock
            self.rec.note = f"已锁定 {lock}"
        else:
            self.cfg.pop("hand_lock", None)
            self.rec.note = "已取消锁定"
        hp.save_cfg(self.cfg)
        self.status.configure(text=self.rec.note, fg=ACC)

    def record(self, key):
        if self.last is None:
            self._note("还没有可用的一帧", BADC)
            return
        m, quality, sel, _ = self.last
        if self.frozen and m is None:
            self._note("冻结的这一帧没有手", BADC)
            return
        if sel is None:
            # 和命令行版同一条规则：录到错的手上比没数据更糟
            self._note("拒绝记录：分不出哪只是写字的手，先按 [ 或 ] 锁定", BADC)
            return
        if m is None:
            self._note("拒绝记录：这一帧没数据", BADC)
            return
        name = hp.CLASSES[key][0]
        if quality == "bad":
            self._note(f"已记录「{name}」但**视角不合格**（{m['px_w']:.0f}px）—— "
                       f"这类样本会污染结论", WARNC)
        else:
            self._note(f"已记录「{name}」  共 {self.rec.counts()[key]} 条", OKC)
        self.rec.record(key, m)

    def compare(self):
        if self.last and self.last[0] is None:
            self._note("先让画面里出现手", WARNC)
            return
        if self.frozen:
            self._note("已冻结，先按空格继续", WARNC)
            return
        text, sep = self.rec.compare()
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")
        self._note("对照表已生成（看右侧）", ACC)

    def save_samples(self):
        try:
            d = hp.save_samples(self.rec, os.path.join(ROOT, "hand_probe"))
            self._note(f"已保存到 {os.path.basename(d)}", OKC)
        except Exception as e:                           # noqa: BLE001
            self._note(f"保存失败：{e}", BADC)

    def clear_samples(self):
        self.rec = hp.Recorder()
        self._note("样本已清空", ACC)
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.configure(state="disabled")

    def toggle_freeze(self):
        self.frozen = not self.frozen
        self._note("画面已冻结（按空格继续）" if self.frozen else "画面继续",
                   ACC if not self.frozen else WARNC)

    def reprobe(self):
        """重新跑机位探测（会短暂占用摄像头）。"""
        import subprocess
        self._note("正在重新探测摄像头…", ACC)
        self.root.update_idletasks()

        def work():
            p = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tools", "probe_camera.py")],
                capture_output=True, text=True, errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            top = ""
            for line in out.splitlines():
                if "最高可用分辨率" in line:
                    top = line.strip()
            self.root.after(0, lambda: self._reprobe_done(top, out))

        threading.Thread(target=work, daemon=True).start()

    def _reprobe_done(self, top, out):
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", (top or "探测完成（未找到结论行）") + "\n\n" + out)
        self.report.configure(state="disabled")
        self._note("探测完成，重启应用生效", OKC)

    def _note(self, msg, color=DIM):
        self.status.configure(text=msg, fg=color)

    def on_close(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
            if self.hand is not None:
                self.hand.close()
        except Exception:                                # noqa: BLE001
            pass
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="握笔姿势检测 · 窗口应用")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--backend", default=None,
                    choices=["DSHOW", "MSMF", "ANY"])
    a = ap.parse_args()

    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:                                    # noqa: BLE001
        pass
    GripApp(root, cam_idx=a.camera, backend=a.backend)
    root.mainloop()


if __name__ == "__main__":
    main()
