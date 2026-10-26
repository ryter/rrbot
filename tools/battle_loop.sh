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

# Перевод ARM-библиотек: после перезапуска одной сессии оверлей
# libndk_translation.so иногда не монтируется, и игра падает
# («libmain.so is for EM_AARCH64 instead of EM_X86_64»).
ndk_ok() {
    adb shell ls /system/lib64/libndk_translation.so >/dev/null 2>&1
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
    # Запуск игры через adb: `waydroid app launch` иногда падает с
    # «Sending reply failed», и игра не поднимается.
    adb shell am start -n com.my.defense/games.my.heart.commonpreloaderlib.PreloadActivity \
        >/dev/null 2>&1
    sleep 10
    if alive && ndk_ok; then
        echo "==> сессия помогла"
        return 0
    fi
    if alive && ! ndk_ok; then
        echo "==> сессия поднялась без libndk_translation — нужен контейнер"
    else
        echo "==> сессия не помогла: перезапускаю контейнер (графический буфер)"
    fi
    # Сессия не помогла — сломан графический буфер или не смонтировался
    # оверлей ARM-перевода; перезапускаем контейнер.
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
    # Запуск игры через adb: `waydroid app launch` иногда падает с
    # «Sending reply failed», и игра не поднимается.
    adb shell am start -n com.my.defense/games.my.heart.commonpreloaderlib.PreloadActivity \
        >/dev/null 2>&1
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
    # Режим боя: Co-Op (по умолчанию) или PvP (BATTLE_MODE=pvp).
    # В кооп путь короткий: меню → Co-Op → «Играть» (обычный или золотой
    # билет) → «Случайный» → модификатор. Карточки колоды высаживаем сами.
    MODE_ARGS=""
    [ "${BATTLE_MODE:-coop}" = "pvp" ] && MODE_ARGS="--pvp"
    # Призыв + слияние. Колоду и умение героя не трогаем.
    timeout 600 ./build/rrbot --backend emulator $MODE_ARGS --steps 60 --rounds 6 \
        --no-ability ${RRBOT_ARGS:-} --record "$LOG/battle_loop" >> "$LOG/battle_loop.log" 2>&1
    echo "=== пауза 5 с ==="
    sleep 5
done
