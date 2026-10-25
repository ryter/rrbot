#include "adb.hpp"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

namespace rr {
namespace {

// Выполнить команду и вернуть её вывод.
std::string run(const std::string& command) {
    std::array<char, 4096> buffer{};
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) return {};
    std::string out;
    while (::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        out += buffer.data();
    }
    ::pclose(pipe);
    return out;
}

// Выполнить команду, вернуть код возврата.
//
// Перенаправления вывода задаёт вызывающий: дописывать свои здесь нельзя
// — иначе они перекроют те, что уже есть в команде (например, вывод
// screencap в файл ушёл бы в /dev/null).
int run_status(const std::string& command) {
    return std::system(command.c_str());
}

}  // namespace

Adb::Adb(std::string serial) : serial_(std::move(serial)) {
    // Ищем устройство: либо заданный серийник, либо единственное в списке.
    const std::string devices = run("adb devices | tail -n +2");
    std::string found;
    std::size_t pos = 0;
    while (pos < devices.size()) {
        const std::size_t end = devices.find('\n', pos);
        const std::string line = devices.substr(pos, end - pos);
        pos = (end == std::string::npos) ? devices.size() : end + 1;
        if (line.empty()) continue;
        const std::string serial = line.substr(0, line.find('\t'));
        if (serial.find("offline") != std::string::npos) continue;
        if (serial_.empty() || serial == serial_) {
            found = serial;
            break;
        }
    }
    if (found.empty()) {
        error_ = "устройство adb не найдено (запусти эмулятор: waydroid session start)";
        return;
    }
    serial_ = found;
    available_ = true;
}

bool Adb::screencap(cv::Mat* out) {
    if (!available_) return false;
    // Имя файла с PID: два бота не должны мешать друг другу — иначе один
    // читает файл, который пишет другой, и получает битый PNG.
    const std::string path = "/tmp/rrbot-adb-" + std::to_string(::getpid()) + ".png";
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (run_status("adb -s " + serial_ + " exec-out screencap -p > " + path +
                       " 2>/dev/null") != 0) {
            continue;
        }
        // Проверяем, что файл дописан: у PNG есть подпись, а в конце — IEND.
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (file == nullptr) continue;
        unsigned char head[8] = {0};
        const size_t got = std::fread(head, 1, sizeof(head), file);
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        std::fclose(file);
        static const unsigned char kPng[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
        if (got != sizeof(head) || size < 1024 || std::memcmp(head, kPng, sizeof(kPng)) != 0) {
            continue;
        }
        cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        ::remove(path.c_str());
        *out = std::move(frame);
        return true;
    }
    ::remove(path.c_str());
    return false;
}

bool Adb::screen_size(int* width, int* height) {
    if (!available_) return false;
    const std::string out = run("adb -s " + serial_ + " shell wm size 2>/dev/null");
    // «Physical size: 941x1030» — число перед 'x' и после.
    const std::size_t x = out.find("x");
    if (x == std::string::npos) return false;
    // Перед «x» число целиком: брать последнюю цифру нельзя (из «941»
    // получилось бы 1).
    std::size_t start = x;
    while (start > 0 && std::isdigit(static_cast<unsigned char>(out[start - 1]))) --start;
    if (start == x) return false;
    *width = std::atoi(out.c_str() + start);
    *height = std::atoi(out.c_str() + x + 1);
    return *width > 0 && *height > 0;
}

bool Adb::tap(int x, int y) {
    if (!available_) return false;
    return run_status("adb -s " + serial_ + " shell input tap " + std::to_string(x) + " " +
                      std::to_string(y) + " >/dev/null 2>&1") == 0;
}

bool Adb::key_event(int code) {
    if (!available_) return false;
    return run_status("adb -s " + serial_ + " shell input keyevent " + std::to_string(code) +
                      " >/dev/null 2>&1") == 0;
}

bool Adb::swipe(int x1, int y1, int x2, int y2, int ms) {
    if (!available_) return false;
    return run_status("adb -s " + serial_ + " shell input swipe " + std::to_string(x1) + " " +
                      std::to_string(y1) + " " + std::to_string(x2) + " " + std::to_string(y2) + " " +
                      std::to_string(ms) + " >/dev/null 2>&1") == 0;
}

}  // namespace rr
