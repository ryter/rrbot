#!/usr/bin/env python3
"""Простой HTTP/HTTPS-прокси на хосте — обход MTU для гостя Waydroid.

Зачем: хост выходит в интернет через VPN с MTU 1376, а у гостя eth0
MTU 1500. Пакеты крупнее ~1350 байт теряются, и игра не может
скачать ресурсы («Не удалось обновить игру»). Гость не может поменять
MTU без root, поэтому трафик гоняем через прокси на хосте:

    python3 tools/host_proxy.py --port 8899 &
    adb reverse tcp:8899 tcp:8899
    adb shell settings put global http_proxy 127.0.0.1:8899

Прокси понимает и обычные запросы (http://), и CONNECT (https://) —
в последнем случае просто проксирует байты по TCP.

Вернуть как было:
    adb shell settings put global http_proxy :0
    adb reverse --remove tcp:8899
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading


def pump(src: socket.socket, dst: socket.socket) -> None:
    """Перекладывает байты из src в dst, пока один из сокетов жив."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def read_head(client: socket.socket) -> bytes | None:
    """Читает заголовки запроса (до пустой строки)."""
    data = b""
    while b"\r\n\r\n" not in data:
        try:
            chunk = client.recv(4096)
        except OSError:
            return None
        if not chunk:
            return None
        data += chunk
        if len(data) > 65536:
            return None
    return data


def handle(client: socket.socket) -> None:
    upstream: socket.socket | None = None
    try:
        client.settimeout(30)
        request = read_head(client)
        if request is None:
            return
        head, _, rest = request.partition(b"\r\n\r\n")
        parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(parts) < 3:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        method, target = parts[0].upper(), parts[1]

        if method == "CONNECT":
            host, _, port_s = target.partition(":")
            port = int(port_s or 443)
            upstream = socket.create_connection((host, port), timeout=30)
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if rest:
                upstream.sendall(rest)
        else:
            host, port, path = parse_absolute(target)
            upstream = socket.create_connection((host, port), timeout=30)
            lines = head.split(b"\r\n")
            lines[0] = f"{method} {path} HTTP/1.1".encode("latin-1")
            upstream.sendall(b"\r\n".join(lines) + b"\r\n\r\n" + rest)

        client.settimeout(None)
        upstream.settimeout(None)
        threads = [
            threading.Thread(target=pump, args=(client, upstream), daemon=True),
            threading.Thread(target=pump, args=(upstream, client), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except Exception as exc:  # noqa: BLE001 — прокси не должен падать
        print(f"ошибка соединения: {exc}", file=sys.stderr)
    finally:
        for s in (client, upstream):
            if s is None:
                continue
            try:
                s.close()
            except OSError:
                pass


def parse_absolute(target: str) -> tuple[str, int, str]:
    """Разбирает абсолютный URL запроса: хост, порт, путь."""
    if target.startswith("http://"):
        rest = target[len("http://") :]
    else:
        rest = target
    hostport, _, path = rest.partition("/")
    host, _, port_s = hostport.partition(":")
    return host, int(port_s or 80), "/" + path


def main() -> int:
    parser = argparse.ArgumentParser(description="Прокси для гостя Waydroid")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(64)
    print(f"прокси слушает {args.host}:{args.port}", flush=True)
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
