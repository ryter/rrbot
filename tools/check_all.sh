#!/usr/bin/env bash
# Проверить всё, что можно проверить без игры: сборка, тесты, регрессионные
# наборы по кадрам.
#
#     bash tools/check_all.sh
#
# Живая игра проверяется отдельно (нужен работающий сервер и эмулятор):
#     cd cpp && ./build/rrbot --backend adb --steps 45 --rounds 3
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
FAILED=0

step() {
    echo
    echo "=== $* ==="
}

run() {
    local label="$1"
    shift
    if "$@"; then
        echo "OK: $label"
    else
        echo "СБОЙ: $label"
        FAILED=1
    fi
}

step "сборка C++-бота"
run "make" make -C cpp

step "тесты"
run "pytest" .venv/bin/python -m pytest tests -q

step "чтение энергии по вырезам"
run "check_digits" .venv/bin/python tools/check_digits.py

step "навигация по меню по записанным кадрам"
run "check_nav" .venv/bin/python tools/check_nav.py

step "поиск арены по настоящим кадрам"
run "check_arena" .venv/bin/python -m pytest tests/test_cpp_arena.py -q

echo
if [ "$FAILED" -eq 0 ]; then
    echo "всё в порядке"
else
    echo "есть сбои (см. выше)"
fi
exit "$FAILED"
