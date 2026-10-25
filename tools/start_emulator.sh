#!/bin/bash
# Подъём эмулятора с игрой и запуск боёв по кругу.
#
#     bash tools/start_emulator.sh <sudo-пароль>
#
# Что делает:
#   1. запускает сессию Waydroid (в фоне — команда не возвращается);
#   2. показывает окно эмулятора (окно держит контейнер размороженным);
#   3. чинит сеть контейнера (нужен sudo: ufw, NAT, маршрут гостя);
#   4. запускает игру Rush Royale;
#   5. запускает бои по кругу (tools/battle_loop.sh).
#
# После перезагрузки компьютера контейнер надо поднимать именно так: без
# шага 3 в контейнере не будет сети, и игра зависнет на «Проверка обновлений».
set -u

ROOT=/home/lennistyle/rushroyale-lab
PASS="${1:-}"
LOG=/tmp/opencode
mkdir -p "$LOG"

if [ -z "$PASS" ]; then
    echo "нужен sudo-пароль: bash tools/start_emulator.sh <пароль>" >&2
    exit 1
fi

echo "==> сессия Waydroid"
setsid waydroid session start > "$LOG/session_start.log" 2>&1 &
for _ in $(seq 1 40); do
    sleep 2
    grep -q "ready" "$LOG/session_start.log" 2>/dev/null && break
done
tail -2 "$LOG/session_start.log"

echo "==> окно эмулятора"
setsid waydroid show-full-ui > "$LOG/show_ui.log" 2>&1 &
sleep 6

echo "==> сеть контейнера"
echo "$PASS" | sudo -S bash "$ROOT/tools/waydroid_net.sh" "$PASS" 2>&1 | tail -6

echo "==> игра"
waydroid app launch com.my.defense >/dev/null 2>&1 || \
    adb shell monkey -p com.my.defense -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
sleep 10

echo "==> бои по кругу"
pkill -9 -f "[b]attle_loop.sh" 2>/dev/null
pkill -9 -f "[b]uild/rrbot --backend" 2>/dev/null
sleep 1
: > "$LOG/battle_loop.log"
setsid "$ROOT/tools/battle_loop.sh" > "$LOG/battle_loop_run.log" 2>&1 &
sleep 5
echo "готово: бои идут, лог — $LOG/battle_loop.log, кадры — $LOG/battle_loop/"
