"""Small toolbox: snip a Queens puzzle off the screen, then show the solution or play it."""
import ctypes
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parent
sys.path.insert(0, str(TOOL))

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # use real pixels on scaled displays

import tkinter as tk
import numpy as np
import pyautogui
import torch
from PIL import Image, ImageGrab, ImageTk
from queens.colour_zones import zones_by_colour
from queens.drawer import GridDrawer
from queens.solvers import backtrack
from queens.zone_model import ZoneModel

pyautogui.PAUSE = 0.05  # small pause after each action

OUT = ROOT / "output" / "captures"
zone_model = ZoneModel().load(ROOT / "weights" / "zone_model_any_size.pt")


def solve(path):
    # Size by the network (fit), zones by colour grouping, queens by backtracking
    x = zone_model.image_to_tensor(path).unsqueeze(0).to(zone_model.device)
    with torch.no_grad():
        m = zone_model.detect_size_by_fit(x)[0]
    zones = zones_by_colour(Image.open(path), m)
    if zones is None:
        return None, m, None
    sols, tries = backtrack(zones)
    if not sols:
        return None, m, None
    queens = np.zeros((m, m), dtype=int)
    queens[np.arange(m), sols[0]] = 1
    out = path.with_name(path.stem + "_solved.png")
    GridDrawer(SimpleNamespace(m=m, zones=zones, queens=queens)).draw(show_queens=True).save(out)
    return out, m, sols[0]


class Snipper:
    """Select a screen area, then show the solution (Solve) or click it in (Play)."""

    def __init__(self, root):
        self.root = root
        root.title("Queens")
        root.attributes("-toolwindow", True)  # stay above other windows
        root.resizable(False, False)
        tk.Button(root, text="Solve", width=16, command=lambda: self.start(False)).pack(padx=10, pady=(10, 4))
        tk.Button(root, text="Play", width=16, command=lambda: self.start(True)).pack(padx=10, pady=4)
        self.status = tk.Label(root, text="Ready")
        self.status.pack(padx=10, pady=(0, 10))

    def start(self, play):
        self.play_mode = play
        self.root.withdraw()  # hide the toolbox so it's not in the screenshot
        self.root.update()
        time.sleep(0.2)
        self.shot = ImageGrab.grab()  # freeze the screen
        self.top = tk.Toplevel(self.root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.top, cursor="cross", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.photo = ImageTk.PhotoImage(self.shot)  # show the frozen screenshot
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.top.bind("<Escape>", lambda e: self.finish())  # Esc cancels

    def press(self, e):
        self.x0, self.y0 = e.x, e.y
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

    def drag(self, e):
        self.canvas.coords(self.rect, self.x0, self.y0, e.x, e.y)

    def release(self, e):
        box = (min(self.x0, e.x), min(self.y0, e.y), max(self.x0, e.x), max(self.y0, e.y))
        self.finish()
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return  # too small: treat as a mis-click
        path = OUT / time.strftime("capture_%Y%m%d_%H%M%S.png")
        self.shot.crop(box).save(path)
        out, m, cols = solve(path)
        if out is None:
            self.status.config(text=f"Read as {m}x{m}, but no solution: try a tighter selection")
            return
        self.status.config(text=f"Solved ({m}x{m})")
        if self.play_mode:
            self.play(box, m, cols)
        else:
            os.startfile(out)  # open with the default image viewer

    def play(self, box, m, cols):
        # Click each queen cell twice: 1st click = X, 2nd = queen
        self.root.withdraw()  # keep the toolbox off the grid
        self.root.update()
        w, h = (box[2] - box[0]) / m, (box[3] - box[1]) / m
        user32 = ctypes.windll.user32
        if not user32.BlockInput(True):  # freeze user input (needs admin)
            self.status.config(text="Run as admin to lock the mouse")
        try:
            for r, c in enumerate(cols):
                pyautogui.click(box[0] + (c + .5) * w, box[1] + (r + .5) * h, clicks=2, interval=0.02)
        finally:
            user32.BlockInput(False)  # always give the mouse back
            self.root.deiconify()

    def finish(self):
        self.top.destroy()
        self.root.deiconify()  # show the toolbox again


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    Snipper(root)
    root.mainloop()