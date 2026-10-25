// Захват экрана и ввод через Android-эмулятор (ADB).
//
// В эмуляторе окно игры — это весь экран устройства: кадр берётся из
// буфера (adb exec-out screencap), а не с рабочего стола, поэтому
// перекрытие окна другими приложениями не мешает. Ввод — adb shell input.
#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace rr {

class Adb {
public:
    explicit Adb(std::string serial = "");

    bool available() const { return available_; }
    const std::string& error() const { return error_; }

    // Кадр устройства (как есть, в пикселях устройства).
    bool screencap(cv::Mat* out);
    // Размер экрана устройства.
    bool screen_size(int* width, int* height);
    // Нажать в координатах устройства.
    bool tap(int x, int y);
    // Свайп (прокрутка).
    bool swipe(int x1, int y1, int x2, int y2, int ms);
    // Нажать клавишу (4 — «Назад»).
    bool key_event(int code);

private:
    std::string serial_;
    bool available_ = false;
    std::string error_;
};

}  // namespace rr
