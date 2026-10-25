#!/usr/bin/env bash
# Доступ к /dev/uinput: для виртуальной мыши бота и для ydotool.
#
# Запуск:  sudo bash tools/setup_ydotool.sh
#
# Что делает:
#   1. Создаёт правило udev, открывающее /dev/uinput группе input.
#   2. Добавляет пользователя в группу input.
#   3. Перезагружает правила udev.
#
# После этого нужно перезайти в сессию (группа вступает в силу при
# новом логине).
#
# Своё устройство (rrbot/input.py, UinputMouseBackend) бот создаёт сам —
# ydotool при этом не нужен. Демон ydotool поднимается только если им
# пользуются сторонние программы:
#     systemctl --user enable --now ydotool

set -euo pipefail

RULE=/etc/udev/rules.d/99-uinput.rules
TARGET_USER="${1:-${SUDO_USER:-$USER}}"

echo "==> Правило udev для /dev/uinput"
cat > "$RULE" <<'EOF'
KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"
EOF
chmod 0644 "$RULE"
echo "    записано: $RULE"

echo "==> Группа input для пользователя $TARGET_USER"
usermod -aG input "$TARGET_USER"

echo "==> Перезагрузка правил udev"
udevadm control --reload-rules
udevadm trigger

echo "==> Проверка"
ls -l /dev/uinput

echo
echo "Готово. Осталось:"
echo "  1) перезайти в сессию (выйти из входа и войти снова), чтобы группа input применилась;"
echo "  2) проверить виртуальную мышь бота:"
echo "       .venv/bin/python tools/mouse_click.py --move 400 400"
echo "  3) (по желанию) включить демон ydotool:  systemctl --user enable --now ydotool"
