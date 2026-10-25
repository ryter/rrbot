// Экранная обёртка над живой игрой: окно, холст, перевод координат.
//
// Игра может идти в двух режимах, и это меняет только «откуда берём кадр
// и куда шлём клик»:
//
//   * desktop — веб-клиент в браузере: кадр с экрана (grim), клик
//     виртуальной мышью (uinput);
//   * adb — клиент игры в Android-эмуляторе: кадр из буфера устройства
//     (adb exec-out screencap), клик через adb shell input tap.
//
// Всё остальное — разбор интерфейса, цифры, кнопки — общее.
#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>

#include "common.hpp"

namespace rr {

struct GameWindow {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    bool valid = false;
};

// Найти окно браузера с игрой по списку окон композитора.
GameWindow find_game_window();
// Найти окно эмулятора Waydroid.
GameWindow find_waydroid_window();

// Откуда берём кадр и куда шлём клик.
class Backend {
public:
    virtual ~Backend() = default;
    // Кадр целиком (окно браузера или экран устройства).
    virtual bool grab(cv::Mat* frame) = 0;
    // Положение холста в кадре. Если вернуть false, холст — весь кадр.
    virtual bool canvas_in_frame(const cv::Mat& frame, cv::Rect* rect) { return false; }
    // Перевод точки холста в координаты устройства.
    virtual cv::Point canvas_to_device(cv::Point canvas_point, const cv::Rect& canvas_rect) = 0;
    // Клик в координатах устройства.
    virtual bool tap(int x, int y) = 0;
    // Клик настоящим касанием (adb shell input tap).
    //
    // Нужен там, где игра реагирует только на касание, а не на клик
    // мышью: так ведёт себя окно выбора модификатора в «Подземельях» —
    // клик мышью по значку не даёт эффекта, а касание запускает бой.
    virtual bool tap_touch(int x, int y) { return false; }
    // Перетаскивание в координатах устройства: зажать, провести, отпустить.
    // Нужно, чтобы поставить карточку юнита на поле (слияние пешек).
    virtual bool drag(int x1, int y1, int x2, int y2) { return false; }
    // Перетаскивание по частям: зажать, вести, отпустить.
    //
    // Нужно, чтобы вести палец за целью: в бою камера едет, и цель под
    // курсором смещается — если вести по старым координатам, слияние
    // попадает мимо.
    virtual bool drag_start(int x, int y) { return false; }
    virtual bool drag_move(int x, int y) { return false; }
    virtual bool drag_end(int x, int y) { return false; }
    // Раскладка интерфейса для этого источника кадра.
    virtual Layout layout() const = 0;
    // Настоящий размер кадра: раскладка приводится к нему, чтобы кадр не
    // приходилось растягивать (от растяжения размываются цифры энергии).
    virtual cv::Size frame_size() const { return layout().canvas_size; }
    // Холст — это весь кадр (экран устройства): растягивать его нельзя,
    // раскладка приводится к размеру кадра сама.
    virtual bool canvas_is_frame() const { return false; }
    // Клавиша «Назад»: снимает всплывающие окна.
    virtual bool press_back() { return false; }
};

// Веб-клиент в браузере: grim + виртуальная мышь.
std::unique_ptr<Backend> make_desktop_backend(bool dry_run = false);
// Клиент в Android-эмуляторе: adb.
std::unique_ptr<Backend> make_adb_backend(bool host_mouse = false);

class Game {
public:
    explicit Game(Backend* backend) : backend_(backend) {}

    // Снять холст игры (пустой Mat, если не получилось).
    cv::Mat grab_canvas();
    // Кликнуть по точке холста.
    bool click_canvas(cv::Point canvas_point);
    // Коснуться точки холста (настоящим касанием, а не мышью).
    bool touch_canvas(cv::Point canvas_point);
    // Перетащить в координатах холста: поставить карточку юнита на поле.
    bool drag_canvas(cv::Point from, cv::Point to);
    // Перетаскивание по частям (см. Backend::drag_start).
    bool drag_start_canvas(cv::Point at);
    bool drag_move_canvas(cv::Point at);
    bool drag_end_canvas(cv::Point at);
    // Экранные координаты точки холста.
    cv::Point to_screen(cv::Point canvas_point) const;
    // Нажать «Назад» (снимает поп-апы).
    bool press_back() { return backend_ && backend_->press_back(); }

private:
    Backend* backend_ = nullptr;
    // Прямоугольник холста внутри кадра: помним найденное, чтобы пережить
    // экраны, где рамка другого цвета (например, «Подземелья»).
    cv::Rect canvas_rect_{225, 139, 492, 804};
    bool canvas_known_ = false;
};

}  // namespace rr
