// Разбор интерфейса: цифры HUD, кнопки, правила навигации.
#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <string>
#include <vector>

#include "common.hpp"

namespace rr {

// Чтение числа по эталонам цифр.
class DigitReader {
public:
    // Загрузить эталоны из каталога (assets/digits).
    bool load(const std::string& directory);
    // Сколько цифр известно (для проверки при отладке).
    size_t known() const { return templates_.size(); }

    struct Reading {
        std::string text;
        double value = 0.0;
        double confidence = 0.0;
        bool ok = false;
        // Что опознано по каждой цифре: имя и уверенность.
        std::vector<std::pair<std::string, double>> glyphs;
    };

    Reading read(const cv::Mat& region) const;
    // Есть ли в области хоть какие-то цифры (нужно для проверки «в бою»).
    bool has_glyphs(const cv::Mat& region) const;
    // Глифы области — сырые, до сопоставления с эталонами.
    std::vector<cv::Mat> glyphs_of(const cv::Mat& region) const;
    // Сохранить образец цифры в каталог эталонов. Пустой каталог — тот,
    // из которого эталоны загружены. Возвращает путь файла.
    std::string save_sample(const std::string& digit, const cv::Mat& glyph,
                            const std::string& directory = "") const;

private:
    struct Glyph {
        cv::Mat image;  // нормализованный глиф 8x14
    };

    std::vector<cv::Mat> glyphs(const cv::Mat& region) const;
    // Глифы с заданным порогом яркости: сначала ищем яркое число, а если
    // его нет — потускневшее (потолок энергии).
    std::vector<cv::Mat> glyphs_at(const cv::Mat& region, int value_threshold,
                                   int max_saturation) const;
    std::pair<std::string, double> match(const cv::Mat& glyph) const;
    static cv::Mat normalize(const cv::Mat& crop);
    static double correlate(const cv::Mat& a, const cv::Mat& b);

    std::map<std::string, std::vector<cv::Mat>> templates_;
    // Каталог, из которого загружены эталоны: туда же сохраняем новые.
    std::string directory_;
};

// Всё, что бот знает об игре на экране.
class Hud {
public:
    explicit Hud(const std::string& assets_dir, Layout layout);
    // Заменить раскладку (игра повернулась — размер кадра изменился).
    void set_layout(Layout layout) { layout_ = std::move(layout); }

    // Энергия и уверенность чтения.
    double read_mana(const cv::Mat& canvas, double* confidence = nullptr) const;
    // Прочитанные цифры энергии: что именно распознано.
    DigitReader::Reading read_mana_digits(const cv::Mat& canvas) const;
    // Глифы числа энергии — сырые, для сбора эталонов.
    std::vector<cv::Mat> mana_glyphs(const cv::Mat& canvas) const;
    // Есть ли флакон энергии (признак боя).
    bool has_energy_flask(const cv::Mat& canvas) const;
    // Хватает ли энергии на призыв: виджет оранжевый, когда хватает.
    bool summon_ready(const cv::Mat& canvas) const;
    // Точка кнопки призыва в координатах холста.
    cv::Point summon_button(const cv::Mat& canvas) const;
    // Прямоугольник карточки в колоде (номер с 1) — карточка, которую
    // тянут на поле.
    cv::Rect deck_card_rect(int position) const;
    // Кнопка умения героя: точка в координатах холста.
    //
    // Пустая точка (-1,-1) — умением пользоваться нельзя (веб-клиент).
    cv::Point ability_button(const cv::Mat& canvas) const;
    // Готово ли умение (откат кончился).
    //
    // Пока идёт откат, кольцо вокруг кнопки заполняется постепенно; когда
    // умение готово, кольцо замыкается и вокруг него появляется жёлтое
    // свечение — это и признак готовности.
    bool ability_ready(const cv::Mat& canvas) const;
    // Идёт ли бой.
    bool in_battle(const cv::Mat& canvas) const;

    // Найденные цветные кнопки.
    std::vector<UiElement> find_buttons(const cv::Mat& canvas) const;
    // Кнопки для продвижения по меню, в порядке приоритета.
    std::vector<UiElement> navigation_candidates(const cv::Mat& canvas) const;

    // Кнопка покупки — по ней бот не кликает никогда.
    bool is_purchase(const cv::Mat& canvas, const UiElement& element) const;
    // Найти элемент по надписи (например, «PvP»). Надёжнее цвета: надпись
    // однозначно указывает на кнопку, даже если рядом такой же фон.
    //
    // threshold — порог совпадения. Для надписей входа в Co-Op он выше
    // обычного: на экране награды «Играть» совпадает с шаблоном на 0.59, а
    // на своём экране — на 0.99, поэтому порог 0.85 отделяет одно от другого.
    bool find_text(const cv::Mat& canvas, const std::string& name, cv::Rect* out,
                   double threshold = 0.55) const;

    // Экраны входа в Co-Op (Подземелья). Бот входит в кооп сам, без PvP:
    // на этом сервере соперник находится через раз, а кооп начинается сразу.
    //
    // Возвращает true, если на кадре узнан один из этих экранов, и кладёт в
    // out кнопку, которую надо нажать.
    enum class CoopScreen {
        kNone,       // Это не экран коопа.
        kFloors,    // Список этажей: жмём «Играть» на первом этаже.
        kAlly,      // Выбор союзника: жмём «Случайный».
        kModifier,  // Выбор модификатора: жмём первую карточку.
    };
    CoopScreen coop_screen(const cv::Mat& canvas, cv::Rect* out) const;

    const DigitReader& digits() const { return digits_; }
    // Заменить каталог эталонов цифр: нужно, чтобы сравнить наборы.
    bool use_digits(const std::string& directory) { return digits_.load(directory); }

private:
    bool is_close_badge(const cv::Mat& canvas, const UiElement& element) const;
    // То же, но в произвольной раскладке: разбор идёт в каноническом размере
    // холста (см. navigation_candidates).
    std::vector<UiElement> find_buttons(const cv::Mat& canvas, const Layout& layout) const;
    bool in_battle(const cv::Mat& canvas, const Layout& layout) const;
    bool has_energy_flask(const cv::Mat& canvas, const Layout& layout) const;
    std::vector<UiElement> coop_buttons(const std::vector<UiElement>& buttons,
                                        const std::vector<UiElement>& pvps,
                                        const Layout& layout,
                                        const UiElement* exclude = nullptr) const;
    // Кандидаты навигации в произвольной раскладке.
    std::vector<UiElement> navigation_candidates_at(const cv::Mat& canvas,
                                                     const Layout& layout) const;
    // Раскладка, в которой настроены правила поиска кнопок и шаблоны
    // надписей: размер холста здесь канонический, а не настоящий.
    Layout canonical_layout() const;

    DigitReader digits_;
    Layout layout_;
    // Эталоны надписей: имя -> картинка.
    std::map<std::string, cv::Mat> text_templates_;
    std::vector<cv::Mat> currency_templates_;
};

// Канонический размер холста, в котором настроены правила поиска кнопок,
// пороги и шаблоны надписей: 492x804 у веб-клиента, 941x1030 — у мобильного.
//
// Раскладку приводят к настоящему размеру кадра (Layout::scaled_to), и по
// ней уже не видно, какая это была раскладка: определяем по масштабу
// (1.0 — веб-клиент, 950/492 — мобильный; при приведении к размеру кадра
// масштаб не меняется).
cv::Size canonical_canvas_size(const Layout& layout);

}  // namespace rr
