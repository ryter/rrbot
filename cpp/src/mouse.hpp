// Виртуальная мышь: собственное устройство uinput с абсолютными осями.
//
// Почему своё, а не ydotool: тот эмулирует абсолютные координаты
// относительными смещениями, и компоновщик применяет к ним свой масштаб —
// курсор промахивался вдвое (mousemove --absolute 500 500 уводил курсор
// в 1000, 1000). Своё устройство с ABS_X/ABS_Y попадает пиксель в пиксель.
#pragma once

#include <string>

namespace rr {

class Mouse {
public:
    // Создаёт устройство. Бросает исключение, если нет доступа к uinput.
    Mouse(int width, int height);
    ~Mouse();

    Mouse(const Mouse&) = delete;
    Mouse& operator=(const Mouse&) = delete;

    // Навести курсор в экранных координатах.
    void move(int x, int y);
    // Клик в экранных координатах.
    void click(int x, int y);
    // Зажать кнопку в точке (и сразу там остаться).
    //
    // Нужно для «умного» перетаскивания: зажали в начале, потом вели
    // курсор за целью (камера едет, цель смещается) и отпустили.
    void press(int x, int y);
    // Отпустить кнопку (курсор остаётся на месте).
    void release();
    // Перетащить: зажать в одной точке, провести в другую, отпустить.
    //
    // Нужно, чтобы поставить карточку юнита на поле: в мобильном клиенте
    // юнит ставится перетаскиванием карточки из колоды, а клик по карточке
    // зовёт юнита в стандартное место.
    void drag(int x1, int y1, int x2, int y2, int steps = 12);

    const std::string& error() const { return error_; }

private:
    void writeAbs(int code, int value);

    int fd_ = -1;
    int width_ = 1920;
    int height_ = 1080;
    std::string error_;
};

// Размер рабочей области (для диапазонов абсолютных осей).
void desktop_size(int* width, int* height);

}  // namespace rr
