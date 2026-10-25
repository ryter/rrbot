#!/bin/bash
# Залить проект на GitHub: создать приватный репозиторий и запушить.
#
#     bash tools/push_github.sh [имя-репозитория]
#
# Нужен один раз авторизованный gh:
#     gh auth login --hostname github.com --git-protocol https --web
#
# Репозиторий создаётся приватным, в ветку main. Если репозиторий уже есть,
# скрипт просто запушит в него.
set -eu
ROOT=/home/lennistyle/rushroyale-lab
NAME="${1:-rushroyale-lab}"
cd "$ROOT"

if ! gh auth status >/dev/null 2>&1; then
    echo "==> gh не авторизован. Выполни:"
    echo "    gh auth login --hostname github.com --git-protocol https --web"
    exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "==> это не git-репозиторий"
    exit 1
fi

if gh repo view "$NAME" >/dev/null 2>&1; then
    echo "==> репозиторий $NAME уже есть — пушим в него"
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com:$(gh api user --jq .login)/$NAME.git"
    git push -u origin main
else
    echo "==> создаю приватный репозиторий $NAME"
    gh repo create "$NAME" --private --source=. --remote=origin --push \
        --description "Бот для Rush Royale: зрение, слияния, доска, навигация по меню"
fi
echo "==> готово: https://github.com/$(gh api user --jq .login)/$NAME"
