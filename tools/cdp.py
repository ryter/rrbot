#!/usr/bin/env python3
"""Мини-драйвер Chrome через DevTools Protocol.

Позволяет открывать страницы, выполнять JS, кликать и снимать скриншоты
окна браузера. Используется для изучения веб-версии игры и подготовки
шаблонов.

Примеры:
    python tools/cdp.py open "https://example.com"
    python tools/cdp.py shot /tmp/opencode/page.png
    python tools/cdp.py js "document.title"
    python tools/cdp.py click "#play-button"
    python tools/cdp.py text
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

def _cdp_base() -> str:
    """Базовый адрес DevTools.

    Chrome слушает порт только на IPv6-локалхосте (``[::1]:port``), поэтому
    адрес проверяется в обоих вариантах.
    """
    env = os.environ.get("RRBOT_CDP")
    if env:
        return env
    for base in ("http://127.0.0.1:9222", "http://[::1]:9222"):
        try:
            with urllib.request.urlopen(f"{base}/json/version", timeout=3):
                return base
        except Exception:
            continue
    for port in (9223,):
        for host in ("127.0.0.1", "[::1]"):
            base = f"http://{host}:{port}"
            try:
                with urllib.request.urlopen(f"{base}/json/version", timeout=3):
                    return base
            except Exception:
                continue
    return "http://127.0.0.1:9222"


CDP = _cdp_base()

# Chrome не всегда слушает на обоих локальных адресах: если он поднялся
# только на IPv6, подключение к 127.0.0.1 отклоняется (и наоборот).
# Поэтому адрес выбирается при первом успешном запросе, а не задаётся жёстко.
_ALT = {"127.0.0.1": "[::1]", "[::1]": "127.0.0.1"}


def _bases() -> list[str]:
    base = CDP.rstrip("/")
    out = [base]
    for old, new in _ALT.items():
        if old in base:
            out.append(base.replace(old, new))
    return out


def _http(path: str) -> list | dict:
    last: Exception | None = None
    for base in _bases():
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 — пробуем следующий адрес
            last = exc
    raise last if last else RuntimeError("CDP недоступен")


def pages() -> list[dict]:
    return [t for t in _http("/json/list") if t["type"] == "page"]


def pick_page(url_hint: str | None = None) -> dict:
    """Выбрать вкладку: по совпадению URL, иначе первая доступная."""
    ps = pages()
    if not ps:
        raise SystemExit("Нет вкладок Chrome. Запустите Chrome с --remote-debugging-port=9222")
    if url_hint:
        for p in ps:
            if url_hint in p.get("url", ""):
                return p
    return ps[0]


class Session:
    """Сессия CDP поверх websockets-транспорта (используем curl-подобный fallback)."""

    def __init__(self, target: dict):
        self.ws = target["webSocketDebuggerUrl"]
        self.id = 0
        self._next = 0

    def send(self, method: str, params: dict | None = None) -> dict:
        self.id += 1
        msg = {"id": self.id, "method": method, "params": params or {}}
        payload = json.dumps(msg)
        # Минимальный websocket-клиент поверх стандартной библиотеки.
        import socket
        import os
        import struct
        from urllib.parse import urlparse

        u = urlparse(self.ws)
        sock = socket.create_connection((u.hostname, u.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        # Host берём из netloc: у IPv6-адреса он уже в скобках («[::1]:9223»),
        # а hostname без скобок Chrome отвергает — соединение сразу рвётся.
        req = (
            f"GET {u.path} HTTP/1.1\r\n"
            f"Host: {u.netloc}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode())
        # Читаем HTTP-ответ до конца заголовков.
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        data = payload.encode()
        header = bytearray([0x81])
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        sock.sendall(bytes(header) + masked)

        misses = 0
        while True:
            frame = self._recv(sock)
            if frame is None:
                misses += 1
                if misses > 3:
                    sock.close()
                    raise ConnectionError("соединение с Chrome прервано")
                continue
            misses = 0
            try:
                obj = json.loads(frame)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == self.id:
                sock.close()
                return obj

    @staticmethod
    def _recv(sock) -> str | None:
        import struct

        def read(n: int) -> bytes:
            out = b""
            while len(out) < n:
                chunk = sock.recv(n - len(out))
                if not chunk:
                    raise ConnectionError("closed")
                out += chunk
            return out

        try:
            b1, b2 = read(2)
        except (ConnectionError, OSError):
            return None
        opcode = b1 & 0x0F
        length = b2 & 0x7F
        try:
            if length == 126:
                length = struct.unpack(">H", read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", read(8))[0]
            payload = read(length)
        except (ConnectionError, OSError):
            return None
        if opcode == 0x8:
            return None
        return payload.decode("utf-8", errors="replace")


def evaluate(session: Session, expr: str) -> object:
    result = session.send(
        "Runtime.evaluate",
        {"expression": expr, "returnByValue": True, "awaitPromise": True},
    )
    return result.get("result", {}).get("result", {}).get("value")


def screenshot(session: Session, path: str) -> str:
    result = session.send("Page.captureScreenshot", {"format": "png"})
    data = result.get("result", {}).get("data")
    if not data:
        raise SystemExit("Не удалось получить скриншот")
    Path(path).write_bytes(base64.b64decode(data))
    return path


def click_at(session: Session, screen_x: float, screen_y: float, settle: float = 0.35) -> str:
    """Клик в экранных координатах через окно Chrome.

    Перевод: экран -> позиция окна в Hyprland -> рамка браузера (вкладки и
    адресная строка) -> координаты viewport, в которых работает CDP.

    Unity/WebGL требует последовательность: наведение, пауза (подсветка),
    нажатие с удержанием, отпускание. Отдельно пересылаем pointerType,
    иначе часть движков игнорирует событие.
    """
    import json
    import subprocess

    info = evaluate(
        session,
        """(() => ({
            oh: window.outerHeight, ih: window.innerHeight,
            ow: window.outerWidth,  iw: window.innerWidth,
            dpr: window.devicePixelRatio
        }))()""",
    )
    if not isinstance(info, dict):
        return "не удалось получить геометрию окна"

    dpr = float(info.get("dpr") or 1)
    border_top = float(info.get("oh", 0) - info.get("ih", 0))
    border_left = float(info.get("ow", 0) - info.get("iw", 0)) / 2

    # Положение окна на экране берём у композитора.
    win_x, win_y = 0.0, 0.0
    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, timeout=5).stdout
        for c in json.loads(out or b"[]"):
            if "chrome" in (c.get("class") or "").lower():
                at = c.get("at", [0, 0])
                win_x, win_y = float(at[0]), float(at[1])
                break
    except Exception:
        pass

    vx = (screen_x - win_x - border_left) / dpr
    vy = (screen_y - win_y - border_top) / dpr

    def send(kind: str, **extra) -> None:
        payload = {"type": kind, "x": vx, "y": vy, "pointerType": "mouse"}
        payload.update(extra)
        session.send("Input.dispatchMouseEvent", payload)

    # Наведение: без него кнопка не подсвечивается.
    send("mouseMoved")
    time.sleep(settle)

    # Нажатие и отпускание без промежуточного перемещения: иначе Unity
    # воспринимает это как перетаскивание, а не как клик.
    send("mousePressed", button="left", clickCount=1, buttons=1)
    time.sleep(0.1)
    send("mouseReleased", button="left", clickCount=1, buttons=0)
    return f"клик ({screen_x:.0f}, {screen_y:.0f}) -> viewport ({vx:.0f}, {vy:.0f})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Драйвер Chrome через CDP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_open = sub.add_parser("open", help="открыть URL в активной вкладке")
    p_open.add_argument("url")
    p_open.add_argument("--new", action="store_true", help="открыть в новой вкладке")

    sub.add_parser("shot", help="сделать скриншот вкладки").add_argument("path")

    p_js = sub.add_parser("js", help="выполнить JavaScript")
    p_js.add_argument("expr")

    p_click = sub.add_parser("click", help="кликнуть по селектору")
    p_click.add_argument("selector")

    p_at = sub.add_parser("click-at", help="кликнуть по экранным координатам X Y")
    p_at.add_argument("x", type=float)
    p_at.add_argument("y", type=float)

    p_move = sub.add_parser("move", help="навести мышь на экранные координаты X Y")
    p_move.add_argument("x", type=float)
    p_move.add_argument("y", type=float)

    sub.add_parser("text", help="текст страницы")
    sub.add_parser("list", help="список вкладок")
    sub.add_parser("logs", help="консоль браузера")

    args = parser.parse_args()

    if args.cmd == "list":
        for p in pages():
            print(f"{p['title'][:50]:50s} | {p['url'][:100]}")
        return 0

    if args.cmd == "open":
        if args.new:
            subprocess.run(
                ["google-chrome-stable", args.url],
                check=False,
                timeout=30,
            )
            time.sleep(3)
            return 0
        session = Session(pick_page())
        session.send("Page.enable")
        session.send("Page.navigate", {"url": args.url})
        time.sleep(4)
        print(args.url)
        return 0

    session = Session(pick_page())
    session.send("Page.enable")

    if args.cmd == "shot":
        print(screenshot(session, args.path))
    elif args.cmd == "js":
        print(evaluate(session, args.expr))
    elif args.cmd == "click":
        expr = (
            "(() => { const el = document.querySelector("
            f"{json.dumps(args.selector)}"
            "); if (!el) return 'not found'; el.click(); return 'clicked'; })()"
        )
        print(evaluate(session, expr))
    elif args.cmd == "click-at":
        print(click_at(session, args.x, args.y))
    elif args.cmd == "move":
        session.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": args.x, "y": args.y},
        )
        print(f"наведение на ({args.x:.0f}, {args.y:.0f})")
    elif args.cmd == "text":
        print(evaluate(session, "document.body.innerText") or "")
    elif args.cmd == "logs":
        print(evaluate(session, "window.__logs ? window.__logs.join('\\n') : 'нет логов'"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
