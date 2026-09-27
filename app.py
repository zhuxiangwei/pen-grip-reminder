#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
握笔姿势检测 · 窗口应用（稳健版）
==================================

**为什么是 tkinter 而不是 PySide6**
   PySide6 / PyQt5 / PyQt6 **全没装**（实测三选全 NO），而 **tkinter 是 Python 自带的**。
   项目依赖已经不少（opencv + mediapipe + matplotlib），窗口本身没必要再拉 100MB+ 的 Qt。
   **零新增依赖**对「双击就能用」很重要。

**核心原则：不重写检测逻辑**
   指标、视角门控、手别选择、阈值对照**全部复用** `grip_metrics` / `hand_probe`。
   窗口只是套了个界面。两份实现会随时间漂移 —— 改一处忘另一处是最常见的坑。

**这一版相对初版最大的改动：能报错、能自救**
   ─────────────────────────────────────────────
   初版把「打开摄像头 + 加载模型」放在后台线程里，一旦失败只是线程静默死掉，
   主窗口卡在"正在打开摄像头…"，用户看着就像"启动不了"；而且任何异常只打到
   那个一闪而过的黑框里，**根本看不到原因**。

   这一版：
     · 全局 sys.excepthook + 主线程 try/except —— 任何异常都弹出窗口 + 写日志文件
     · 摄像头/模型初始化走后台线程，但**失败会明确回传并展示原因**（不是卡死）
     · 图片显示失败有兜底（PPM 不行就落临时文件）
     · `.bat` 出错不再关窗
   所以"不能启动"时，你会直接看到**为什么**，而不是一个打不开的窗口。

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
    │ 状态栏：实时提示              │  [保存样本]           │
    └─────────────────────────────┴───────────────────────┘

**按键**（和命令行版一致，肌肉记忆可复用）
   1 / 2 / 3   记录当前帧为对应类别
   z           开关拇指-食指放大镜
   [ / ]       锁定被测手 = 画面左半 / 右半（不看"左右手"，只看画面位置）
   \\           取消锁定，回到自动（按运动量猜）
   空格         冻结 / 继续画面
   p           出对照表（等价于点按钮）
   w           保存样本到磁盘
   q / Esc     退出

用法
----
    python app.py
    python app.py --camera 1
   双击「窗口应用.bat」也可（出错会保留窗口，并在 app_error.log 留痕）
"""
import argparse
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT)

import pg_utils as pg                                   # noqa: E402
import grip_metrics as gm                               # noqa: E402
import hand_probe as hp                                 # noqa: E402
import mediapipe as mp                                  # noqa: E402

# 画面配色（贴近深色主题）
BG = "#16181d"
FG = "#e8eaed"
DIM = "#8b93a1"
ACC = "#5aa9ff"
OKC = "#4ec96b"
WARNC = "#e0a13a"
BADC = "#e05c5c"
PANEL = "#1d2129"


def log_error(msg):
    """把错误同时打到 stderr 和 app_error.log（项目根），方便用户回传。"""
    try:
        p = os.path.join(ROOT, "app_error.log")
        with open(p, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg + "\n")
    except Exception:                                    # noqa: BLE001
        pass


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
        self._photo = None                  # 持引用，避免被 GC 后画面消失
        self._last_frame = None
        self._last_frame_t = None           # 最近一次成功收到帧的时间（看门狗用）

        # 初始化状态（后台线程回传）
        self.init_done = False
        self.init_error = None
        self._init_lock = threading.Lock()

        self._build_ui()
        self._bind_keys()
        # 先启动初始化（后台线程），主线程轮询结果 —— 窗口立刻可见
        self.root.after(50, self._poll_init)
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
        self.canvas.create_text(400, 300, text="正在初始化摄像头…",
                                fill=DIM, font=("Microsoft YaHei UI", 13),
                                tags="hint")

        self.status = tk.Label(left, text="正在初始化…", bg=BG, fg=DIM,
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
                         ("重新探测机位", self.reprobe),
                         ("重新打开摄像头", self.restart_camera)):
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
        lab = tk.Label(parent, text="—", bg=PANEL, fg=FG, anchor="w", justify="left",
                       font=("Consolas", 9), padx=6, pady=4)
        lab.pack(fill="x")
        return lab

    def _bind_keys(self):
        self.root.bind("<Key>", self.on_key)
        self.root.focus_set()

    # ------------------------------------------------------------ 初始化轮询
    def _poll_init(self):
        """主线程每 200ms 检查后台初始化结果，避免线程里 sys.exit 静默死掉。"""
        if self.init_error is not None:
            self._on_init_failed(self.init_error)
            return
        if self.init_done:
            if not self.running:
                self.running = True
                self._last_frame_t = time.time()        # 看门狗计时起点
                self.root.after(0, self._loop)
                self.root.after(1000, self._watchdog)   # 4s 内收不到帧就报错
            return
        self.root.after(200, self._poll_init)

    # ------------------------------------------------------------ 摄像头
    def _start_camera(self):
        def work():
            import traceback
            try:
                model = pg.find_model(hp.HAND_MODEL)
                hand = hp.make_hand(model)
                # 分辨率从探测配置读（tools/probe_camera.py 写）
                cap = pg.open_camera_auto(idx=self.cam_idx, backend=self.backend)
                f = pg.grab_frame(cap)
                if f is not None:
                    self.frame_size = (f.shape[1], f.shape[0])
                with self._init_lock:
                    self.hand = hand
                    self.cap = cap
                    self.init_done = True
            except SystemExit as e:
                tb = traceback.format_exc()
                log_error("init SystemExit: " + str(e) + "\n" + tb)
                with self._init_lock:
                    self.init_error = f"摄像头/模型初始化失败：{e}\n\n" + tb
            except Exception as e:                       # noqa: BLE001
                tb = traceback.format_exc()
                log_error("init Exception: " + repr(e) + "\n" + tb)
                with self._init_lock:
                    self.init_error = f"{type(e).__name__}: {e}\n\n" + tb

        threading.Thread(target=work, daemon=True).start()

    def _show_fatal(self, title, msg):
        """把致命错误显式展示到画面 + 报告 + 弹窗（不再静默卡死）。"""
        self.running = False
        self.canvas.delete("all")
        self.canvas.create_text(
            max(200, self.canvas.winfo_width() // 2),
            max(120, self.canvas.winfo_height() // 2),
            text=title + "\n\n" + msg[:500],
            fill=BADC, font=("Microsoft YaHei UI", 11), width=560, justify="center")
        self.status.configure(text="❌ " + title + "（详见 app_error.log）", fg=BADC)
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", title + "：\n\n" + msg)
        self.report.configure(state="disabled")
        try:
            messagebox.showerror("启动失败", msg[:1200])
        except Exception:                                # noqa: BLE001
            pass

    def _on_init_failed(self, msg):
        self._show_fatal("摄像头/模型初始化失败", msg)

    def _watchdog(self):
        """初始化成功后若长时间收不到帧，明确报错而不是空转卡死。"""
        if not self.running:
            return
        if self.frozen:
            # 冻结时本来就不取新帧，只看门狗不计时
            self.root.after(1000, self._watchdog)
            return
        if self._last_frame_t is not None and (time.time() - self._last_frame_t) > 4:
            self._show_fatal(
                "摄像头已打开但读不到画面",
                "摄像头设备打开了，但 4 秒内没有回传任何一帧。\n"
                "常见原因：被其他程序占用、权限不足、虚拟摄像头、或驱动异常。\n"
                "点右侧「重新打开摄像头」重试，或换 --camera 索引 / --backend MSMF。")
            return
        self.root.after(1000, self._watchdog)

    def restart_camera(self):
        """「重新打开摄像头」按钮：释放旧资源，重跑初始化。"""
        try:
            if self.cap is not None:
                self.cap.release()
            if self.hand is not None:
                self.hand.close()
        except Exception:                                  # noqa: BLE001
            pass
        self.cap = None
        self.hand = None
        self.init_done = False
        self.init_error = None
        self.running = False
        self._last_frame_t = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="正在重新初始化摄像头…",
                                fill=DIM, font=("Microsoft YaHei UI", 13))
        self.status.configure(text="正在重新初始化…", fg=DIM)
        self.root.after(50, self._poll_init)
        self._start_camera()

    # ------------------------------------------------------------ 主循环
    def _loop(self):
        import cv2
        if not self.running:
            return
        try:
            if self.frozen:
                frame = self._last_frame
            else:
                frame = pg.grab_frame(self.cap)

            if frame is None:
                if self.frozen and self._last_frame is not None:
                    frame = self._last_frame
                else:
                    self.root.after(30, self._loop)
                    return

            self._last_frame = frame
            self._last_frame_t = time.time()     # 喂看门狗
            h, w = frame.shape[:2]
            rgb = frame[:, :, ::-1].copy()
            self.tick += 33

            r = self.hand.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.tick)

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

            self._show_frame(vis, w, h)
            self._update_panels(m, quality, qmsg, pv, pvmsg, sel)
            self.last = (m, quality, sel, self.rec.note)
        except Exception as e:                           # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            log_error("loop Exception: " + repr(e) + "\n" + tb)
            self.status.configure(text=f"⚠️ 处理帧出错：{type(e).__name__}: {e}", fg=BADC)
            self.root.after(500, self._loop)              # 出错不要刷屏，停一下再试
            return
        self.root.after(15, self._loop)

    def _show_frame(self, bgr, w, h):
        """BGR ndarray -> Canvas 上的图片（PPM 内存流，失败落临时文件兜底）。"""
        import cv2
        import os
        import tempfile
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cw = max(320, self.canvas.winfo_width())
        chh = max(240, self.canvas.winfo_height())
        sc = min(cw / w, chh / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        shown = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        hdr = f"P6\n{nw} {nh}\n255\n".encode("ascii")
        data = hdr + shown.tobytes()
        try:
            self._photo = tk.PhotoImage(data=data, format="PPM")
        except Exception:                                # noqa: BLE001
            p = os.path.join(tempfile.gettempdir(), "grip_frame.ppm")
            with open(p, "wb") as f:
                f.write(data)
            self._photo = tk.PhotoImage(file=p)
        self.canvas.delete("hint")
        self.canvas.delete("frame")
        self.canvas.create_image(cw // 2, chh // 2, image=self._photo,
                                 anchor="center", tags="frame")

    def _update_panels(self, m, quality, qmsg, pv, pvmsg, sel):
        if m is None:
            self.quality_box.configure(
                text=("手宽        —\n视角判定    —\n"
                      f"手掌视角角  —\n指标稳定性  —\n\n"
                      f"{'⚠️ 没检测到手' if not self.frozen else '（已冻结）'}"))
        else:
            col = {"ok": OKC, "marginal": WARNC, "bad": BADC}[quality]
            s1, s2 = self.rec.focus_std
            if s1 is None:
                stab = "—（样本不足）"
            else:
                if s1 < 0.08 and s2 < 0.05:
                    stab = f"GOOD  ({s1:.3f} / {s2:.3f})"
                elif s1 < 0.15 and s2 < 0.10:
                    stab = f"FAIR  ({s1:.3f} / {s2:.3f})"
                else:
                    stab = f"POOR  ({s1:.3f} / {s2:.3f})"
            pw = f"{m['px_w']:.0f}px"
            self.quality_box.configure(
                text=(f"手宽        {pw}   (要 ≥{gm.HAND_PX_OK:.0f}px)\n"
                      f"视角判定    {quality.upper()}\n"
                      f"手掌视角角  {m['palm_view_angle']:.0f}°\n"
                      f"指标稳定性  {stab}\n\n"
                      f"手别        {self._sel_text(sel)}\n"
                      f"锁定        {self.sel.lock or '自动(按运动量)'}"))
            self.status.configure(text=f"{qmsg}    |    {pvmsg}", fg=col)

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
        if not self.init_done:
            self._note("还没初始化完", BADC)
            return
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
        if not self.init_done:
            self._note("还没初始化完", BADC)
            return
        self.frozen = not self.frozen
        self._note("画面已冻结（按空格继续）" if self.frozen else "画面继续",
                   ACC if not self.frozen else WARNC)

    def reprobe(self):
        """重新跑机位探测（会短暂占用摄像头）。"""
        import subprocess
        self._note("正在重新探测摄像头…", ACC)
        self.root.update_idletasks()

        def work():
            try:
                p = subprocess.run(
                    [sys.executable, os.path.join(ROOT, "tools", "probe_camera.py")],
                    capture_output=True, text=True, errors="replace")
                out = (p.stdout or "") + (p.stderr or "")
                top = ""
                for line in out.splitlines():
                    if "最高可用分辨率" in line:
                        top = line.strip()
                self.root.after(0, lambda: self._reprobe_done(top, out))
            except Exception as e:                       # noqa: BLE001
                self.root.after(0, lambda: self._reprobe_done("", f"探测失败：{e}"))

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

    # 全局异常兜底：任何未捕获异常都弹窗 + 写日志，而不是静默消失
    def excepthook(etype, value, tb):
        import traceback as _tb
        msg = "".join(_tb.format_exception(etype, value, tb))
        log_error("UNCAUGHT: " + msg)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("程序异常", msg[:1500])
        except Exception:                                # noqa: BLE001
            pass
    sys.excepthook = excepthook

    try:
        root = tk.Tk()
        try:
            root.call("tk", "scaling", 1.25)
        except Exception:                                # noqa: BLE001
            pass
        GripApp(root, cam_idx=a.camera, backend=a.backend)
        root.mainloop()
    except SystemExit as e:
        log_error("main SystemExit: " + str(e))
        messagebox.showerror("启动失败", str(e))
        sys.exit(1)
    except Exception as e:                               # noqa: BLE001
        import traceback as _tb
        msg = _tb.format_exc()
        log_error("main Exception: " + msg)
        try:
            messagebox.showerror("启动失败", str(e))
        except Exception:                                # noqa: BLE001
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
