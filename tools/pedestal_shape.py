#!/usr/bin/env python3
"""Форма основания юнита: круг (ранг 1) или капля (ранг 2).

Слитый юнит рисуется не на круге, а на «капле» — по форме основания можно
понять ранг. Основание — сплошное цветное пятно (плитка вокруг — зелёная),
поэтому берём область, прилегающую к центру юнита, и меряем её форму.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np


def is_tile(img: np.ndarray) -> np.ndarray:
    """Зелёная плитка — это фон (по оттенку, а не по каналу G: основание
    бывает и зелёным)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (h >= 26) & (h <= 56) & (s > 130) & (v > 110)


def shape_of(img: np.ndarray, cx: int, cy: int, radius: int = 38) -> dict:
    """Форма пятна вокруг центра: вытянутость и заполненность."""
    tile = is_tile(img)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    near = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    mask = (near & ~tile).astype(np.uint8)
    # Оставляем область, связанную с центром.
    count, labels = cv2.connectedComponents(mask, 8)
    label = labels[max(0, min(labels.shape[0] - 1, cy)), max(0, min(labels.shape[1] - 1, cx))]
    if label == 0:
        # Центр попал на фон — берём ближайшую по центру область.
        best = None
        for i in range(1, count):
            ys_i, xs_i = np.nonzero(labels == i)
            d = np.min((xs_i - cx) ** 2 + (ys_i - cy) ** 2)
            if best is None or d < best[0]:
                best = (d, i)
        label = best[1] if best else 0
    blob = labels == label
    if blob.sum() < 50:
        return {}
    y = ys[blob].astype(float)
    x = xs[blob].astype(float)
    cov = np.cov(np.vstack([x, y]))
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    stretch = float(np.sqrt(vals[0] / max(vals[1], 1e-6)))
    d = np.hypot(x - cx, y - cy)
    rmax = max(d.max(), 1)
    fill = float(blob.sum() / (np.pi * rmax * rmax))
    return {"stretch": stretch, "fill": fill, "area": int(blob.sum()), "rmax": float(rmax)}


def describe(img: np.ndarray, cx: int, cy: int) -> str:
    s = shape_of(img, cx, cy)
    if not s:
        return "основание не найдено"
    return (f"вытянутость {s['stretch']:.2f}, заполненность {s['fill']:.2f}, "
            f"площадь {s['area']}, размах {s['rmax']:.0f}")


def main() -> int:
    before = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/exp_before2.png")
    after = cv2.imread(sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/exp_after2.png")
    x = int(sys.argv[3]) if len(sys.argv) > 3 else 549
    y = int(sys.argv[4]) if len(sys.argv) > 4 else 722
    print("ДО  (ранг 1):", describe(before, x, y))
    print("ПОСЛЕ (ранг 2):", describe(after, x, y))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
