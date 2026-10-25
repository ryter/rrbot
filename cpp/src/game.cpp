#include "game.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "adb.hpp"
#include "capture.hpp"
#include "common.hpp"
#include "mouse.hpp"

namespace rr {
namespace {

// Окно браузера с игрой.
class DesktopBackend : public Backend {
public:
    explicit DesktopBackend(bool dry_run) {
        window_ = find_game_window();
        if (dry_run) return;
        int screen_w = 1920;
        int screen_h = 1080;
        desktop_size(&screen_w, &screen_h);
        try {
            mouse_ = std::make_unique<Mouse>(screen_w, screen_h);
        } catch (const std::exception&) {
            mouse_.reset();
        }
    }

    bool grab(cv::Mat* frame) override {
        window_ = find_game_window();
        if (!window_.valid) return false;
        return grab_screen(window_.x, window_.y, window_.width, window_.height, frame);
    }

    bool canvas_in_frame(const cv::Mat& frame, cv::Rect* rect) override {
        return find_canvas_rect(frame, rect);
    }

    cv::Point canvas_to_device(cv::Point canvas_point, const cv::Rect& canvas_rect) override {
        return {window_.x + canvas_rect.x + canvas_point.x,
                window_.y + canvas_rect.y + canvas_point.y};
    }

    bool tap(int x, int y) override {
        if (!mouse_) return false;
        mouse_->click(x, y);
        return true;
    }

    bool drag(int x1, int y1, int x2, int y2) override {
        if (!mouse_) return false;
        mouse_->drag(x1, y1, x2, y2);
        return true;
    }

    bool drag_start(int x, int y) override {
        if (!mouse_) return false;
        mouse_->press(x, y);
        return true;
    }

    bool drag_move(int x, int y) override {
        if (!mouse_) return false;
        mouse_->move(x, y);
        return true;
    }

    bool drag_end(int x, int y) override {
        if (!mouse_) return false;
        mouse_->move(x, y);
        ::usleep(200000);
        mouse_->release();
        return true;
    }

    Layout layout() const override { return web_layout(); }

private:
    GameWindow window_;
    std::unique_ptr<Mouse> mouse_;
};

// Клиент игры в Android-эмуляторе.
class AdbBackend : public Backend {
public:
    // host_mouse: кликаем настоящей мышью по окну эмулятора — так клик
    // видно, в отличие от adb input tap.
    explicit AdbBackend(bool host_mouse) : adb_() {
        if (!adb_.available()) return;
        adb_.screen_size(&device_width_, &device_height_);
        if (!host_mouse) return;
        window_ = find_waydroid_window();
        int screen_w = 1920;
        int screen_h = 1080;
        desktop_size(&screen_w, &screen_h);
        try {
            mouse_ = std::make_unique<Mouse>(screen_w, screen_h);
        } catch (const std::exception&) {
            mouse_.reset();
        }
    }

    bool grab(cv::Mat* frame) override {
        if (!adb_.screencap(frame)) return false;
        frame_width_ = frame->cols;
        frame_height_ = frame->rows;
        return true;
    }

    // Окно эмулятора: его могут перетащить, и тогда клики мышью уходят
    // мимо. Положение перепроверяем раз в несколько секунд.
    GameWindow current_window() {
        const auto now = std::chrono::steady_clock::now();
        if (!window_.valid || now - window_checked_ > std::chrono::seconds(3)) {
            window_ = find_waydroid_window();
            window_checked_ = now;
        }
        return window_;
    }

    // В эмуляторе игра занимает весь экран устройства.
    bool canvas_in_frame(const cv::Mat& frame, cv::Rect* rect) override {
        *rect = cv::Rect(0, 0, frame.cols, frame.rows);
        return true;
    }

    // Холст — весь кадр: растягивать нельзя (растяжение смазывает цифры
    // энергии), раскладка приводится к размеру кадра.
    bool canvas_is_frame() const override { return true; }

    cv::Point canvas_to_device(cv::Point canvas_point, const cv::Rect& canvas_rect) override {
        // Координаты холста переводим в пиксели экрана устройства.
        const double scale_x = static_cast<double>(frame_width_) / std::max(1, canvas_rect.width);
        const double scale_y = static_cast<double>(frame_height_) / std::max(1, canvas_rect.height);
        return {static_cast<int>(canvas_point.x * scale_x), static_cast<int>(canvas_point.y * scale_y)};
    }

    bool tap(int x, int y) override {
        if (mouse_) {
            const GameWindow window = current_window();
            if (window.valid) {
                // Окно эмулятора показывает экран устройства один в один.
                mouse_->click(window.x + x, window.y + y);
                return true;
            }
        }
        return adb_.tap(x, y);
    }

    // Настоящее касание: adb shell input tap.
    //
    // Клик мышью окно выбора модификатора в «Подземельях» не принимает
    // (значок модификатора под курсором, а бой не начинается), а касание
    // работает. Поэтому для таких экранов нужен отдельный путь.
    bool tap_touch(int x, int y) override { return adb_.tap(x, y); }

    bool drag(int x1, int y1, int x2, int y2) override {
        if (mouse_ && window_.valid) {
            mouse_->drag(window_.x + x1, window_.y + y1, window_.x + x2, window_.y + y2);
            return true;
        }
        // Без окна (или без мыши) — свайп через adb: он тоже умеет тянуть.
        return adb_.swipe(x1, y1, x2, y2, 600);
    }

    // Перетаскивание по частям — ведём палец за целью (камера едет).
    bool drag_start(int x, int y) override {
        if (!mouse_) return false;
        const GameWindow window = current_window();
        if (!window.valid) return false;
        window_ = window;
        mouse_->press(window.x + x, window.y + y);
        return true;
    }

    bool drag_move(int x, int y) override {
        if (!mouse_ || !window_.valid) return false;
        mouse_->move(window_.x + x, window_.y + y);
        return true;
    }

    bool drag_end(int x, int y) override {
        if (!mouse_ || !window_.valid) return false;
        mouse_->move(window_.x + x, window_.y + y);
        ::usleep(200000);
        mouse_->release();
        return true;
    }

    // Раскладка приведена к настоящему размеру кадра устройства: кадр
    // растягивать нельзя — от растяжения на процент размываются цифры
    // энергии и число перестаёт читаться.
    Layout layout() const override { return emulator_layout().scaled_to(frame_size()); }

    // Настоящий размер кадра устройства: раскладка приводится к нему.
    //
    // Берём размер последнего кадра, а не того, что был при старте: игра
    // может повернуться (портрет -> ландшафт), и раскладка, посчитанная по
    // стартовому размеру, уехала бы на пиксели.
    cv::Size frame_size() const override {
        if (frame_width_ > 0 && frame_height_ > 0) {
            return cv::Size(frame_width_, frame_height_);
        }
        return cv::Size(device_width_, device_height_);
    }

    bool press_back() override {
        if (!adb_.available()) return false;
        return adb_.key_event(4);  // KEYCODE_BACK
    }

    const std::string& error() const { return adb_.error(); }

private:
    Adb adb_;
    std::unique_ptr<Mouse> mouse_;
    GameWindow window_;
    // Когда окно эмулятора проверяли в последний раз (для кэша на 3 с).
    std::chrono::steady_clock::time_point window_checked_{};
    int device_width_ = 1080;
    int device_height_ = 1920;
    int frame_width_ = 1080;
    int frame_height_ = 1920;
};

}  // namespace

// Окно по классу: "google-chrome" или "Waydroid".
static GameWindow find_window_by_class(const std::string& class_name,
                                      const std::string& title_hint) {
    GameWindow result;
    FILE* pipe = ::popen("hyprctl clients 2>/dev/null", "r");
    if (pipe == nullptr) return result;
    std::string out;
    char buffer[512];
    while (::fgets(buffer, sizeof(buffer), pipe) != nullptr) out += buffer;
    ::pclose(pipe);

    // Блоки окон разделены заголовками «Window N (id):».
    std::vector<std::string> blocks;
    std::istringstream stream(out);
    std::string line;
    std::string current;
    while (std::getline(stream, line)) {
        if (line.rfind("Window ", 0) == 0) {
            if (!current.empty()) blocks.push_back(current);
            current.clear();
            continue;
        }
        current += line + "\n";
    }
    if (!current.empty()) blocks.push_back(current);

    for (const std::string& block : blocks) {
        if (block.find("class: " + class_name) == std::string::npos) continue;
        const bool game = block.find(title_hint) != std::string::npos;
        std::smatch match;
        std::regex at_re(R"(at:\s*(-?\d+),(-?\d+))");
        std::regex size_re(R"(size:\s*(\d+),(\d+))");
        if (!std::regex_search(block, match, at_re)) continue;
        const int x = std::stoi(match[1].str());
        const int y = std::stoi(match[2].str());
        if (!std::regex_search(block, match, size_re)) continue;
        const int w = std::stoi(match[1].str());
        const int h = std::stoi(match[2].str());
        if (game) return {x, y, w, h, true};
        if (!result.valid) result = {x, y, w, h, true};
    }
    return result;
}

GameWindow find_game_window() { return find_window_by_class("google-chrome", "Rush Royale"); }

// Окно с игрой внутри эмулятора: его класс начинается с "waydroid".
GameWindow find_waydroid_window() { return find_window_by_class("waydroid", "waydroid"); }

std::unique_ptr<Backend> make_desktop_backend(bool dry_run) {
    return std::make_unique<DesktopBackend>(dry_run);
}

std::unique_ptr<Backend> make_adb_backend(bool host_mouse) {
    return std::make_unique<AdbBackend>(host_mouse);
}

cv::Mat Game::grab_canvas() {
    cv::Mat frame;
    if (!backend_ || !backend_->grab(&frame)) return {};

    cv::Rect found;
    if (backend_->canvas_in_frame(frame, &found)) {
        canvas_rect_ = found;
        canvas_known_ = true;
    } else if (!canvas_known_) {
        return {};
    } else {
        // Рамка другого цвета (например, на экране «Подземелья»): берём
        // найденное ранее положение. Если окно изменило размер, холст
        // центрируется по горизонтали, а вертикальное смещение сохраняем.
        const int centered = (frame.cols - canvas_rect_.width) / 2;
        const int x = canvas_rect_.x - 225 + centered;
        const int y = canvas_rect_.y;
        if (x < 0 || y < 0 || x + canvas_rect_.width > frame.cols ||
            y + canvas_rect_.height > frame.rows) {
            return {};
        }
        canvas_rect_ = cv::Rect(x, y, canvas_rect_.width, canvas_rect_.height);
    }

    cv::Mat canvas = frame(canvas_rect_).clone();
    if (!backend_->canvas_is_frame()) {
        const cv::Size target = backend_->layout().canvas_size;
        rr::resize_to_canvas(&canvas, target);
    }
    return canvas;
}

cv::Point Game::to_screen(cv::Point canvas_point) const {
    if (!backend_) return canvas_point;
    return backend_->canvas_to_device(canvas_point, canvas_rect_);
}

bool Game::click_canvas(cv::Point canvas_point) {
    if (!backend_) return false;
    const cv::Point device = to_screen(canvas_point);
    return backend_->tap(device.x, device.y);
}

bool Game::touch_canvas(cv::Point canvas_point) {
    if (!backend_) return false;
    const cv::Point device = to_screen(canvas_point);
    if (backend_->tap_touch(device.x, device.y)) return true;
    return backend_->tap(device.x, device.y);
}

bool Game::drag_canvas(cv::Point from, cv::Point to) {
    if (!backend_) return false;
    const cv::Point a = to_screen(from);
    const cv::Point b = to_screen(to);
    return backend_->drag(a.x, a.y, b.x, b.y);
}

bool Game::drag_start_canvas(cv::Point at) {
    if (!backend_) return false;
    const cv::Point p = to_screen(at);
    return backend_->drag_start(p.x, p.y);
}

bool Game::drag_move_canvas(cv::Point at) {
    if (!backend_) return false;
    const cv::Point p = to_screen(at);
    return backend_->drag_move(p.x, p.y);
}

bool Game::drag_end_canvas(cv::Point at) {
    if (!backend_) return false;
    const cv::Point p = to_screen(at);
    return backend_->drag_end(p.x, p.y);
}

}  // namespace rr
