#!/usr/bin/env bash
# Восстановить сеть и ADB у эмулятора Waydroid.
#
# После заморозки контейнера (Android уходит в suspend) теряются маршруты
# гостя, а на хосте остаётся невалидная ARP-запись. После перезагрузки
# хоста или включения ufw теряется ещё и доступ к DNS.
#
# Здесь всё возвращается:
#   1) поднимаем окно (это размораживает контейнер);
#   2) разрешаем в ufw трафик waydroid0 (DNS, adb) и форвардинг;
#   3) настраиваем NAT для подсети контейнера;
#   4) возвращаем гостю маршрут по умолчанию и MTU;
#   5) чистим ARP на хосте и подключаем adb.
#
# Запуск:
#   bash tools/waydroid_net.sh <sudo-пароль>
set -u

SUDO_PASS="${1:-}"
NET=192.168.240.0/24
GW=192.168.240.1

run_sudo() {
    if [ -n "$SUDO_PASS" ]; then
        echo "$SUDO_PASS" | sudo -S -p '' "$@"
    else
        sudo -p '' "$@"
    fi
}

quiet_sudo() {
    if [ -n "$SUDO_PASS" ]; then
        echo "$SUDO_PASS" | sudo -S -p '' "$@" >/dev/null 2>&1
    else
        sudo -p '' "$@" >/dev/null 2>&1
    fi
}

echo "==> поднимаю окно эмулятора (размораживает контейнер)"
waydroid show-full-ui >/dev/null 2>&1
sleep 5

# --- 1. ufw: трафик от контейнера -----------------------------------------
# DNS (53) и adb (5555) идут на хост, форвардинг — наружу.
echo "==> ufw: DNS и форвардинг для waydroid0"
quiet_sudo ufw allow in on waydroid0 to any port 53 proto udp
quiet_sudo ufw allow in on waydroid0 to any port 53 proto tcp
quiet_sudo ufw allow in on waydroid0 to any port 5555 proto tcp
quiet_sudo ufw route allow in on waydroid0

# --- 2. NAT ----------------------------------------------------------------
echo "==> NAT для $NET"
if ! quiet_sudo iptables -t nat -C POSTROUTING -s "$NET" ! -o waydroid0 -j MASQUERADE; then
    quiet_sudo iptables -t nat -A POSTROUTING -s "$NET" ! -o waydroid0 -j MASQUERADE
fi
quiet_sudo sysctl -w net.ipv4.ip_forward=1 >/dev/null

# --- 3. гость: маршрут по умолчанию и MTU ----------------------------------
IP=$(waydroid status 2>/dev/null | awk '/IP address/ {print $3}')
if [ -z "$IP" ] || [ "$IP" = "UNKNOWN" ]; then
    IP=$(awk '/waydroid0.leases/ {print $3}' /var/lib/misc/dnsmasq.waydroid0.leases 2>/dev/null | tail -1)
fi
echo "==> адрес контейнера: ${IP:-неизвестен}"

if [ -n "$IP" ] && [ "$IP" != "UNKNOWN" ]; then
    echo "==> возвращаю гостю маршрут по умолчанию и MTU 1376"
    quiet_sudo waydroid shell -- ip link set eth0 mtu 1376
    quiet_sudo waydroid shell -- ip route replace default via "$GW" dev eth0 onlink
    echo "==> чищу ARP на хосте"
    quiet_sudo ip neigh flush dev waydroid0
    echo "==> подключаю adb"
    adb connect "$IP:5555" 2>&1 | tail -1
fi

sleep 2
adb devices
echo "==> проверка связи из контейнера"
adb shell ping -c 1 -W 3 my.games 2>&1 | tail -2
