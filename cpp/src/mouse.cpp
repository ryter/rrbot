#include "mouse.hpp"

#include <fcntl.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>

namespace rr {
namespace {

// Создать устройство uinput с абсолютными осями.
int create_device(int width, int height) {
    int fd = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd < 0) return -1;

    if (::ioctl(fd, UI_SET_EVBIT, EV_SYN) < 0 ||
        ::ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0 ||
        ::ioctl(fd, UI_SET_EVBIT, EV_ABS) < 0) {
        ::close(fd);
        return -1;
    }
    ::ioctl(fd, UI_SET_KEYBIT, BTN_LEFT);
    ::ioctl(fd, UI_SET_KEYBIT, BTN_RIGHT);
    ::ioctl(fd, UI_SET_KEYBIT, BTN_MIDDLE);
    ::ioctl(fd, UI_SET_ABSBIT, ABS_X);
    ::ioctl(fd, UI_SET_ABSBIT, ABS_Y);

    uinput_setup setup{};
    setup.id.bustype = BUS_VIRTUAL;
    setup.id.vendor = 0x2333;
    setup.id.product = 0x6666;
    setup.id.version = 1;
    std::strncpy(setup.name, "rrbot-virtual-mouse", UINPUT_MAX_NAME_SIZE - 1);
    if (::ioctl(fd, UI_DEV_SETUP, &setup) < 0) {
        ::close(fd);
        return -1;
    }
    // Диапазоны абсолютных осей: экран в пикселях.
    uinput_abs_setup abs_x{};
    abs_x.code = ABS_X;
    abs_x.absinfo.minimum = 0;
    abs_x.absinfo.maximum = width - 1;
    abs_x.absinfo.fuzz = 0;
    abs_x.absinfo.flat = 0;
    abs_x.absinfo.resolution = 0;
    uinput_abs_setup abs_y{};
    abs_y.code = ABS_Y;
    abs_y.absinfo.minimum = 0;
    abs_y.absinfo.maximum = height - 1;
    abs_y.absinfo.fuzz = 0;
    abs_y.absinfo.flat = 0;
    abs_y.absinfo.resolution = 0;
    if (::ioctl(fd, UI_ABS_SETUP, &abs_x) < 0 || ::ioctl(fd, UI_ABS_SETUP, &abs_y) < 0) {
        ::close(fd);
        return -1;
    }
    if (::ioctl(fd, UI_DEV_CREATE) < 0) {
        ::close(fd);
        return -1;
    }
    ::usleep(200000);  // ядру нужно время создать устройство
    return fd;
}

void emit(int fd, int type, int code, int value) {
    input_event event{};
    event.type = static_cast<uint16_t>(type);
    event.code = static_cast<uint16_t>(code);
    event.value = value;
    ::write(fd, &event, sizeof(event));
}

}  // namespace

void desktop_size(int* width, int* height) {
    *width = 1920;
    *height = 1080;
    FILE* pipe = ::popen("hyprctl monitors 2>/dev/null", "r");
    if (pipe == nullptr) return;
    std::string out;
    char buffer[512];
    while (::fgets(buffer, sizeof(buffer), pipe) != nullptr) out += buffer;
    ::pclose(pipe);
    std::regex re(R"(\d+x(\d+)@\d+)");
    int max_w = 0;
    int max_h = 0;
    auto begin = std::sregex_iterator(out.begin(), out.end(), re);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        int h = std::stoi((*it)[1].str());
        max_h = std::max(max_h, h);
    }
    std::regex re2(R"((\d+)x\d+@\d+)");
    begin = std::sregex_iterator(out.begin(), out.end(), re2);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        int w = std::stoi((*it)[1].str());
        max_w = std::max(max_w, w);
    }
    if (max_w > 0) *width = max_w;
    if (max_h > 0) *height = max_h;
}

Mouse::Mouse(int width, int height) : width_(width), height_(height) {
    fd_ = create_device(width_, height_);
    if (fd_ < 0) {
        error_ = "нет доступа к /dev/uinput (нужна группа input и правило udev)";
        throw std::runtime_error(error_);
    }
}

Mouse::~Mouse() {
    if (fd_ >= 0) {
        ::ioctl(fd_, UI_DEV_DESTROY);
        ::close(fd_);
    }
}

void Mouse::writeAbs(int code, int value) {
    emit(fd_, EV_ABS, code, value);
}

void Mouse::move(int x, int y) {
    x = std::clamp(x, 0, width_ - 1);
    y = std::clamp(y, 0, height_ - 1);
    writeAbs(ABS_X, x);
    writeAbs(ABS_Y, y);
    emit(fd_, EV_SYN, SYN_REPORT, 0);
}

void Mouse::click(int x, int y) {
    move(x, y);
    ::usleep(25000);
    emit(fd_, EV_KEY, BTN_LEFT, 1);
    emit(fd_, EV_SYN, SYN_REPORT, 0);
    ::usleep(35000);
    emit(fd_, EV_KEY, BTN_LEFT, 0);
    emit(fd_, EV_SYN, SYN_REPORT, 0);
}

void Mouse::press(int x, int y) {
    move(x, y);
    // Держим кнопку: игра подхватывает карточку только после короткого
    // нажатия, быстрое движение она считает свайпом.
    ::usleep(200000);
    emit(fd_, EV_KEY, BTN_LEFT, 1);
    emit(fd_, EV_SYN, SYN_REPORT, 0);
    ::usleep(150000);
}

void Mouse::release() {
    emit(fd_, EV_KEY, BTN_LEFT, 0);
    emit(fd_, EV_SYN, SYN_REPORT, 0);
}

void Mouse::drag(int x1, int y1, int x2, int y2, int steps) {
    if (steps < 1) steps = 1;
    press(x1, y1);
    // Ведём курсор плавно: игра ждёт, пока палец «проведёт» по карточке,
    // иначе перетаскивание не засчитывается.
    for (int step = 1; step <= steps; ++step) {
        const double t = static_cast<double>(step) / steps;
        move(static_cast<int>(std::lround(x1 + (x2 - x1) * t)),
             static_cast<int>(std::lround(y1 + (y2 - y1) * t)));
        ::usleep(40000);
    }
    ::usleep(200000);
    release();
}

}  // namespace rr
