#!/bin/bash
# Бои по кругу: пока эмулятор жив.
#
#     setsid tools/battle_loop.sh > /tmp/opencode/battle_loop_run.log 2>&1 &
#
# Каждый проход — до 6 раундов (бой → выход в меню → бой). Логи боёв:
# /tmp/opencode/battle_loop.log, кадры боёв: /tmp/opencode/battle_loop/.
#
# Если контейнер Waydroid «замёрз» (screencap отдаёт пустой файл), его надо
# перезапустить. Скрипт делает это сам, если задан пароль sudo в переменной
# окружения WAYDROID_SUDO_PASSWORD (в репозиторий пароль не кладём):
#
#     WAYDROID_SUDO_PASSWORD=... setsid tools/battle_loop.sh ...
set -u
ROOT=/home/lennistyle/rushroyale-lab
LOG=/tmp/opencode
PASS="${WAYDROID_SUDO_PASSWORD:-}"

# Контейнер жив? (пустой кадр — значит, SurfaceFlinger встал)
alive() {
    adb exec-out screencap -p > "$LOG/alive.png" 2>/dev/null
    [ -s "$LOG/alive.png" ]
}

# Перезапуск сессии Waydroid + восстановление сети.
#
# Перезапуск одной сессии НЕ всегда помогает: если сломался графический
# буфер (см. notes/ОКРУЖЕНИЕ.md, раздел «Эмулятор вылетает»), то после
# `waydroid session start` screencap продолжает отдавать пустой файл. Тогда
# перезапускаем сам контейнер (systemctl restart waydroid-container) — сеть
# после этого чинится waydroid_net.sh.
restart_emulator() {
    if [ -z "$PASS" ]; then
        echo "==> контейнер замёрз: нужен перезапуск (нет WAYDROID_SUDO_PASSWORD)"
        return 1
    fi
    echo "==> контейнер замёрз: перезапускаю сессию"
    echo "$PASS" | sudo -S waydroid session stop >/dev/null 2>&1
    sleep 5
    setsid waydroid session start > "$LOG/session_start.log" 2>&1 &
    for _ in $(seq 1 40); do
        sleep 2
        grep -q ready "$LOG/session_start.log" 2>/dev/null && break
    done
    setsid waydroid show-full-ui > "$LOG/show_ui.log" 2>&1 &
    sleep 6
    echo "$PASS" | sudo -S bash "$ROOT/tools/waydroid_net.sh" "$PASS" >/dev/null 2>&1
    sleep 3
    waydroid app launch com.my.defense >/dev/null 2>&1
    sleep 10
    if alive; then
        echo "==> сессия помогла"
        return 0
    fi
    # Сессия не помогла — сломан графический буфер, перезапускаем контейнер.
    echo "==> сессия не помогла: перезапускаю контейнер (графический буфер)"
    echo "$PASS" | sudo -S waydroid session stop >/dev/null 2>&1
    sleep 3
    echo "$PASS" | sudo -S systemctl restart waydroid-container >/dev/null 2>&1
    sleep 5
    setsid waydroid session start > "$LOG/session_start.log" 2>&1 &
    for _ in $(seq 1 40); do
        sleep 2
        grep -q ready "$LOG/session_start.log" 2>/dev/null && break
    done
    setsid waydroid show-full-ui > "$LOG/show_ui.log" 2>&1 &
    sleep 6
    echo "$PASS" | sudo -S bash "$ROOT/tools/waydroid_net.sh" "$PASS" >/dev/null 2>&1
    sleep 3
    waydroid app launch com.my.defense >/dev/null 2>&1
    sleep 10
    alive
}

cd "$ROOT/cpp" || exit 1
while true; do
    echo "=== проход, $(date +%H:%M:%S) ==="
    if ! alive; then
        restart_emulator || { echo "=== пауза 60 с ==="; sleep 60; continue; }
    fi
    # Размер окна влияет на размер кадра: сменился — поехали все области
    # раскладки. Приводим окно к опорному размеру перед каждым проходом.
    if [ "$(adb shell wm size | awk '{print $3}')" != "941x1030" ]; then
        echo "=== размер окна не 941x1030, привожу ==="
        adb shell wm size 941x1030
        sleep 2
    fi
    # Режим боя: PvP (по умолчанию) или Co-Op (BATTLE_MODE=coop).
    MODE_ARGS="--pvp"
    [ "${BATTLE_MODE:-pvp}" = "coop" ] && MODE_ARGS=""
    timeout 600 ./build/rrbot --backend emulator $MODE_ARGS --steps 60 --rounds 6 \
        --record "$LOG/battle_loop" >> "$LOG/battle_loop.log" 2>&1
    echo "=== пауза 5 с ==="
    sleep 5
done
