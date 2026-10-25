// Общие константы и структуры живого режима.
#pragma once

#include <opencv2/core.hpp>

#include <cmath>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

namespace rr {

// Раскладка интерфейса: где в кадре что лежит.
//
// У веб-клиента и у мобильного клиента HUD устроен по-разному, поэтому
// области задаются для каждого варианта отдельно. Размеры указаны в
// пикселях канонического холста.
struct Layout {
    // Число энергии (цифры).
    cv::Rect energy;
    // Голубой флакон рядом с числом — есть только в бою.
    cv::Rect energy_flask;
    // Шкала энергии: запасной источник значения, когда число не читается
    // (на потолке энергии цифры гаснут).
    cv::Rect mana_bar;
    // Виджет призыва: карточка юнита и цена.
    cv::Rect summon;
    // Кнопка умения героя: круглая, справа внизу над колодой.
    //
    // Прямоугольник — это внешний обод кольца отката (кнопка диаметром
    // около 100 px у мобильного клиента). Пустой прямоугольник — умением
    // пользоваться нельзя (веб-клиент).
    cv::Rect ability;
    // Ряд карточек юнитов внизу экрана.
    cv::Rect card_row;
    // Сколько карточек в колоде (нумеруются с 1).
    int card_count = 5;
    // Канонический размер холста.
    cv::Size canvas_size;
    // Каталог эталонов цифр (шрифт у клиентов разный).
    std::string digits_dir = "assets/digits";
    // Насколько далеко вправо от края экрана ищется крестик закрытия
    // модального окна (доля от ширины холста). У веб-клиента окно узкое,
    // у мобильного — шире, и крестик стоит левее.
    double close_x_min = 0.78;
    // Масштаб относительно веб-клиента (492 px по ширине): размеры
    // кнопок и пороги заданы в его пикселях.
    double scale = 1.0;
    // Название для логов.
    std::string name;

    // Копия раскладки, приведённая к другому размеру кадра (см. scaled_to).
    Layout scaled_to(cv::Size target) const;
};

// Раскладка веб-клиента в браузере: холст 492x804.
Layout web_layout();
// Раскладка мобильного клиента в эмуляторе: кадр устройства 950x1034.
Layout emulator_layout();

// Привести раскладку к другому размеру кадра.
//
// Устройство эмулятора даёт кадр 941x1030, а раскладка нарисована по
// 950x1034. Если приводить кадр к размеру раскладки, он растягивается на
// процент — и цифры энергии после этого не читаются: область смещается, а
// глифы размываются. Поэтому вместо этого масштабируется сама раскладка, а
// кадр остаётся как есть.
inline Layout Layout::scaled_to(cv::Size target) const {
    Layout out = *this;
    if (canvas_size.width <= 0 || canvas_size.height <= 0) return out;
    const double sx = static_cast<double>(target.width) / canvas_size.width;
    const double sy = static_cast<double>(target.height) / canvas_size.height;
    const auto scale_rect = [sx, sy](const cv::Rect& rect) {
        if (rect.width <= 0 || rect.height <= 0) return rect;
        // Округляем края, а не отбрасываем дробную часть: иначе область
        // энергии съезжает на пиксель и правый край цифр обрезается —
        // число перестаёт читаться.
        const int x1 = static_cast<int>(std::lround(rect.x * sx));
        const int y1 = static_cast<int>(std::lround(rect.y * sy));
        const int x2 = static_cast<int>(std::lround((rect.x + rect.width) * sx));
        const int y2 = static_cast<int>(std::lround((rect.y + rect.height) * sy));
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    };
    out.energy = scale_rect(energy);
    out.energy_flask = scale_rect(energy_flask);
    out.mana_bar = scale_rect(mana_bar);
    out.summon = scale_rect(summon);
    out.ability = scale_rect(ability);
    out.card_row = scale_rect(card_row);
    out.canvas_size = target;
    return out;
}

// Максимальная энергия в игре.
inline const double kManaMax = 500.0;

// Канонический размер холста веб-клиента.
inline const int kCanvasW = 492;
inline const int kCanvasH = 804;

// Элемент интерфейса в координатах холста.
struct UiElement {
    std::string kind;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    cv::Point center() const { return {x + w / 2, y + h / 2}; }
};

// Правило поиска цветной кнопки.
struct ButtonRule {
    const char* name;
    std::vector<std::pair<int, int>> hue_ranges;
    int sat_min, sat_max;
    int val_min, val_max;
    int min_area, min_w, min_h, max_w, max_h;
    // Минимальная доля заполнения прямоугольника (0 — не проверять).
    // Кнопка — плотная заливка, а мягкое свечение или картинка — нет.
    double min_fill = 0.0;
};

// Правила поиска кнопок. H задаётся списком пар: красный в шкале OpenCV
// (0..180) разрывается — крестик закрытия имеет оттенок ~171, а бейдж
// со скидкой ~5.
//
// Определение ровно одно — в layout.cpp. Раньше правила были описаны в двух
// местах (hud.cpp и layout.cpp), а объявление в заголовке помечалось inline:
// линкер тогда молча брал одну из копий, и правки второй не применялись.
const std::vector<ButtonRule>& button_rules();

// Привести кадр к каноническому размеру холста.
//
// Интерполяция выбирается по направлению: INTER_AREA хорош при уменьшении
// (сглаживает), но при увеличении он работает почти как nearest — и цифры
// энергии после этого перестают узнаваться. Поэтому при увеличении берём
// INTER_CUBIC.
//
// Замерено на регрессионном наборе энергии (assets/test_energy): на своём
// разрешении читается 70/70; если кадр уменьшить вдвое и увеличить обратно,
// то INTER_AREA даёт 18/70, а INTER_CUBIC — 40/70.
inline void resize_to_canvas(cv::Mat* canvas, cv::Size target) {
    if (canvas == nullptr || canvas->empty()) return;
    if (canvas->cols == target.width && canvas->rows == target.height) return;
    const bool shrinking = target.width <= canvas->cols && target.height <= canvas->rows;
    cv::resize(*canvas, *canvas, target, 0, 0, shrinking ? cv::INTER_AREA : cv::INTER_CUBIC);
}

}  // namespace rr
