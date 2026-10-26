#include "hud.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

namespace rr {
namespace {
// Маска красных пятен внутри прямоугольника (координаты — как у картинки).
//
// Значок модификатора — красный квадрат; такие же пятна есть у кнопок
// («Отмена», «Закрыть»), поэтому ищем их только внутри карточки.
cv::Mat red_blobs(const cv::Mat& image, const cv::Rect& region) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    if (region.width <= 0 || region.height <= 0) return mask;
    const cv::Rect clipped = region & cv::Rect(0, 0, image.cols, image.rows);
    if (clipped.width <= 0 || clipped.height <= 0) return mask;
    cv::Mat hsv;
    cv::cvtColor(image(clipped), hsv, cv::COLOR_BGR2HSV);
    cv::Mat red;
    cv::inRange(hsv, cv::Scalar(0, 110, 90), cv::Scalar(12, 255, 255), red);
    cv::Mat red2;
    cv::inRange(hsv, cv::Scalar(168, 110, 90), cv::Scalar(180, 255, 255), red2);
    red |= red2;
    red.copyTo(mask(clipped));
    return mask;
}

// Колонки с разрывом между цифрами.
std::vector<int> find_gaps(const cv::Mat& piece) {
    std::vector<int> cuts;
    const int width = piece.cols;
    if (width == 0) return cuts;
    std::vector<double> profile(width, 0.0);
    for (int x = 0; x < width; ++x) {
        double sum = 0.0;
        for (int y = 0; y < piece.rows; ++y) {
            sum += piece.at<uchar>(y, x) ? 1.0 : 0.0;
        }
        profile[x] = sum;
    }
    double max_value = 0.0;
    for (double v : profile) max_value = std::max(max_value, v);
    const double level = std::max(0.5, max_value * 0.18);
    int index = 0;
    while (index < width) {
        if (profile[index] > level) {
            ++index;
            continue;
        }
        const int start = index;
        while (index < width && profile[index] <= level) ++index;
        cuts.push_back((start + index) / 2);
    }
    return cuts;
}


// Минимальная ширина цифры: группы уже — часть цифры.
constexpr int kMinDigitWidth = 7;
// Минимальная ширина части при разрезании слипшихся цифр.
constexpr int kMinPartWidth = 8;
// Максимальная ширина одной цифры: шире — значит, цифры слиплись.
constexpr int kMaxDigitWidth = 14;
// Нормализованный размер эталона цифры.
constexpr int kTemplateW = 8;
constexpr int kTemplateH = 14;
// Минимальная уверенность, при которой глиф считается цифрой.
constexpr double kMinDigitConfidence = 0.75;
// Порог яркости глифа цифры.
//
// Порог 150: при нём глифы совпадают с эталонами (набор собран с этим
// порогом). Более высокий порог отделяет цифры друг от друга, но делает
// глифы тоньше эталонных, и половина цифр перестаёт узнаваться.
constexpr int kBrightValue = 150;
// Порог для потускневшего числа: на потолке энергии цифры гаснут и
// сереют — начертание то же, но яркости меньше.
constexpr int kDimValue = 95;
// Насыщенность глифа: выше — это уже фон (зелёная шкала, значки).
constexpr int kMaxGlyphSaturation = 120;
constexpr int kMaxDimGlyphSaturation = 150;

cv::Mat crop(const cv::Mat& image, cv::Rect rect) {
    rect &= cv::Rect(0, 0, image.cols, image.rows);
    if (rect.width <= 0 || rect.height <= 0) return {};
    return image(rect).clone();
}

// Трофейная дорога («Место в рейтинге»): снизу широкая синяя полоса на
// всю ширину. Стрелка «назад» и кнопка рейтинга сливаются в одну
// компоненту, поэтому find_buttons их не видит (слишком широко).
bool trophy_road(const cv::Mat& work, UiElement* back_out) {
    if (work.empty()) return false;
    const int width = work.cols;
    const int height = work.rows;
    const int top = static_cast<int>(height * 0.82);
    if (top >= height - 4) return false;
    cv::Mat hsv;
    cv::cvtColor(work(cv::Rect(0, top, width, height - top)), hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(95, 80, 80), cv::Scalar(135, 255, 255), mask);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat cents;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8);
    int best = -1;
    int best_area = 0;
    for (int i = 1; i < n; ++i) {
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (w < width * 0.55) continue;
        if (h < height * 0.05 || h > height * 0.22) continue;
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }
    if (best < 0) return false;
    // Экран итогов тоже с синей полосой внизу, но там по центру
    // «Продолжить» — не трофейная дорога, «назад» там ничего не делает.
    {
        cv::Mat bar_hsv = hsv;
        cv::Mat blue;
        cv::inRange(bar_hsv, cv::Scalar(95, 80, 80), cv::Scalar(135, 255, 255), blue);
        cv::Mat labels2;
        cv::Mat stats2;
        cv::Mat cents2;
        const int n2 = cv::connectedComponentsWithStats(blue, labels2, stats2, cents2, 8);
        for (int i = 1; i < n2; ++i) {
            const int w = stats2.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats2.at<int>(i, cv::CC_STAT_HEIGHT);
            const int cx = stats2.at<int>(i, cv::CC_STAT_LEFT) + w / 2;
            if (w < width * 0.12 || w > width * 0.28) continue;
            if (h < height * 0.03 || h > height * 0.10) continue;
            if (std::abs(cx - width / 2) > width * 0.12) continue;
            // «Продолжить» на итогах — не рейтинг на трофейной дороге.
            return false;
        }
    }
    if (back_out != nullptr) {
        const int x = stats.at<int>(best, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(best, cv::CC_STAT_TOP) + top;
        const int bh = stats.at<int>(best, cv::CC_STAT_HEIGHT);
        const int side = std::max(24, std::min(bh - 8, width / 12));
        *back_out = {"back", x + 8, y + std::max(0, (bh - side) / 2), side, side};
    }
    return true;
}

// Карты сокровищ внизу экрана выбора союзника: две кнопки (~140×70).
// Выбранная подсвечивается жёлтым. Берём любую доступную (сначала
// невыбранную синюю — чтобы не щёлкать уже выбранную по кругу).
bool treasure_map(const cv::Mat& work, const std::vector<UiElement>& buttons,
                  cv::Rect* out, bool* gold) {
    if (work.empty() || out == nullptr) return false;
    const int width = work.cols;
    const int height = work.rows;
    std::vector<UiElement> maps;
    for (const UiElement& b : buttons) {
        if (b.kind != "continue" && b.kind != "fight" && b.kind != "orange") continue;
        if (b.y < height * 0.88) continue;
        if (b.x < width * 0.25 || b.x + b.w > width * 0.75) continue;
        if (b.w < width * 0.10 || b.w > width * 0.22) continue;
        if (b.h < height * 0.04 || b.h > height * 0.10) continue;
        maps.push_back(b);
    }
    if (maps.empty()) return false;
    std::sort(maps.begin(), maps.end(),
              [](const UiElement& a, const UiElement& b) { return a.x < b.x; });

    auto yellow_fill = [&](const UiElement& btn) {
        const cv::Rect clipped =
            cv::Rect(btn.x, btn.y, btn.w, btn.h) & cv::Rect(0, 0, width, height);
        if (clipped.width <= 0 || clipped.height <= 0) return 0.0;
        cv::Mat hsv;
        cv::cvtColor(work(clipped), hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(14, 100, 120), cv::Scalar(40, 255, 255), mask);
        return cv::countNonZero(mask) * 1.0 / mask.total();
    };
    auto green_icon = [&](const UiElement& btn) {
        const cv::Rect clipped =
            cv::Rect(btn.x, btn.y, btn.w, btn.h) & cv::Rect(0, 0, width, height);
        if (clipped.width <= 0 || clipped.height <= 0) return 0;
        cv::Mat hsv;
        cv::cvtColor(work(clipped), hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(35, 70, 70), cv::Scalar(95, 255, 255), mask);
        return cv::countNonZero(mask);
    };

    // Уже выбранная (жёлтая) — не трогаем, берём другую или эту, если одна.
    const UiElement* pick = nullptr;
    double pick_yellow = 2.0;
    for (const UiElement& b : maps) {
        const double y = yellow_fill(b);
        if (y < pick_yellow) {
            pick_yellow = y;
            pick = &b;
        }
    }
    if (pick == nullptr) pick = &maps.front();
    *out = cv::Rect(pick->x, pick->y, pick->w, pick->h);
    // Золотая обычно справа; зелёная иконка — обычная карта.
    if (gold != nullptr) {
        *gold = green_icon(*pick) < 150 && pick->x > width * 0.45;
    }
    return true;
}

// Карта уже выбрана: одна из нижних кнопок жёлтая, текст не «без карты».
bool treasure_map_selected(const cv::Mat& work, const std::vector<UiElement>& buttons) {
    if (work.empty()) return false;
    const int width = work.cols;
    const int height = work.rows;
    for (const UiElement& b : buttons) {
        if (b.kind != "continue" && b.kind != "fight" && b.kind != "orange") continue;
        if (b.y < height * 0.88) continue;
        if (b.x < width * 0.25 || b.x + b.w > width * 0.75) continue;
        if (b.w < width * 0.10 || b.w > width * 0.22) continue;
        const cv::Rect clipped =
            cv::Rect(b.x, b.y, b.w, b.h) & cv::Rect(0, 0, width, height);
        if (clipped.width <= 0) continue;
        cv::Mat hsv;
        cv::cvtColor(work(clipped), hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(14, 100, 120), cv::Scalar(40, 255, 255), mask);
        if (cv::countNonZero(mask) * 1.0 / mask.total() > 0.25) return true;
    }
    return false;
}

}  // namespace

// Правила поиска кнопок живут в layout.cpp: раньше они были описаны ещё и
// здесь, а объявление в заголовке помечалось inline — линкер тогда брал
// одну из копий, и правки второй не применялись.

// ---------------------------------------------------------------- цифры

bool DigitReader::load(const std::string& directory) {
    templates_.clear();
    directory_ = directory;
    if (!std::filesystem::exists(directory)) return false;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() != ".png") continue;
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        // Имя файла может быть «3.png» или «3_1.png» (несколько образцов
        // одной цифры) — цифра всегда в начале имени.
        const std::string stem = entry.path().stem().string();
        const std::string digit = stem.substr(0, stem.find('_'));
        templates_[digit].push_back(image);
    }
    return !templates_.empty();
}

cv::Mat DigitReader::normalize(const cv::Mat& input) {
    const int h = input.rows;
    const int w = input.cols;
    const double scale = std::min(kTemplateW / static_cast<double>(std::max(w, 1)),
                                  kTemplateH / static_cast<double>(std::max(h, 1)));
    const int new_w = std::max(1, static_cast<int>(std::lround(w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(h * scale)));
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    cv::Mat canvas(kTemplateH, kTemplateW, CV_8UC1, cv::Scalar(0));
    const int ox = (kTemplateW - new_w) / 2;
    const int oy = (kTemplateH - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(ox, oy, new_w, new_h)));
    return canvas;
}

double DigitReader::correlate(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    cv::Mat af;
    cv::Mat bf;
    a.convertTo(af, CV_32F);
    b.convertTo(bf, CV_32F);
    af = af.reshape(1, 1);
    bf = bf.reshape(1, 1);
    // Второй аргумент cv::mean — это маска, а не выход: среднее берём
    // возвратом значения.
    const cv::Scalar ma = cv::mean(af);
    const cv::Scalar mb = cv::mean(bf);
    af -= ma;
    bf -= mb;
    const double dot = cv::sum(af.mul(bf))[0];
    const double norm = std::sqrt(cv::sum(af.mul(af))[0] * cv::sum(bf.mul(bf))[0]);
    if (norm <= 1e-9) return 0.0;
    return dot / norm;
}

std::vector<cv::Mat> DigitReader::glyphs(const cv::Mat& region) const {
    // Сначала ищем яркое число.
    std::vector<cv::Mat> bright = glyphs_at(region, kBrightValue, kMaxGlyphSaturation);
    if (!bright.empty()) return bright;
    // Число на потолке энергии гаснет и сереет: контуры те же, но яркости
    // меньше — читаем с пониженным порогом.
    return glyphs_at(region, kDimValue, kMaxDimGlyphSaturation);
}

std::vector<cv::Mat> DigitReader::glyphs_at(const cv::Mat& region, int value_threshold,
                                             int max_saturation) const {
    std::vector<cv::Mat> out;
    if (region.empty()) return out;
    cv::Mat gray;
    if (region.channels() == 3) {
        cv::cvtColor(region, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = region;
    }
    cv::Mat saturated;
    if (region.channels() == 3) {
        cv::cvtColor(region, saturated, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        // Бинарная маска цифр: яркие и не насыщенные пиксели. Серые
        // значения глифа портили бы корреляцию с эталонами.
        cv::inRange(saturated, cv::Scalar(0, 0, value_threshold),
                    cv::Scalar(180, max_saturation, 255), mask);
        cv::Mat binary;
        cv::threshold(gray, binary, static_cast<double>(value_threshold), 255.0, cv::THRESH_BINARY);
        cv::bitwise_and(binary, mask, gray);
    } else {
        cv::threshold(gray, gray, value_threshold, 255, cv::THRESH_BINARY);
    }
    if (cv::countNonZero(gray) == 0) return out;

    const int height = gray.rows;
    const int width = gray.cols;
    std::vector<std::vector<int>> groups;
    for (int x = 0; x < width; ++x) {
        bool present = false;
        for (int y = 0; y < height; ++y) {
            if (gray.at<uchar>(y, x)) {
                present = true;
                break;
            }
        }
        if (!present) continue;
        if (!groups.empty() && x == groups.back().back() + 1) {
            groups.back().push_back(x);
        } else {
            groups.push_back({x});
        }
    }
    // Узкая группа — часть цифры, присоединяем к соседней.
    std::vector<std::vector<int>> merged;
    for (const auto& group : groups) {
        if (!merged.empty() &&
            (static_cast<int>(merged.back().size()) < kMinDigitWidth ||
             static_cast<int>(group.size()) < kMinDigitWidth)) {
            merged.back().insert(merged.back().end(), group.begin(), group.end());
        } else {
            merged.push_back(group);
        }
    }
    for (const auto& group : merged) {
        if (group.back() - group.front() < 2) continue;
        cv::Mat sub = crop(gray, cv::Rect(group.front(), 0, group.back() - group.front() + 1, height));
        std::vector<int> ys;
        for (int y = 0; y < sub.rows; ++y) {
            if (cv::countNonZero(sub.row(y)) > 0) ys.push_back(y);
        }
        if (ys.empty()) continue;
        cv::Mat piece = sub(cv::Rect(0, ys.front(), sub.cols, ys.back() - ys.front() + 1));
        if (piece.cols <= kMaxDigitWidth) {
            out.push_back(normalize(piece));
        } else {
            // Слипшиеся цифры: режем по разрывам между ними. В шрифте
            // игры между цифрами бывают пустые или почти пустые колонки
            // («690» — три цифры одним куском). Разрез подтверждаем
            // шириной частей: настоящая цифра шире 8 px, а дырка внутри
            // цифры дала бы слишком узкий кусок.
            const std::vector<int> cuts = find_gaps(piece);
            if (cuts.empty()) {
                out.push_back(normalize(piece));
                continue;
            }
            int start = 0;
            for (int cut : cuts) {
                if (cut - start >= kMinPartWidth && piece.cols - cut >= kMinPartWidth) {
                    out.push_back(normalize(piece(cv::Rect(start, 0, cut - start, piece.rows))));
                    start = cut;
                }
            }
            if (piece.cols - start >= kMinPartWidth) {
                out.push_back(normalize(piece(cv::Rect(start, 0, piece.cols - start, piece.rows))));
            }
        }
    }
    return out;
}

std::pair<std::string, double> DigitReader::match(const cv::Mat& glyph) const {
    if (templates_.empty()) return {"?", 0.0};
    std::string best_name = "?";
    double best_score = -1.0;
    for (const auto& [name, samples] : templates_) {
        for (const cv::Mat& sample : samples) {
            const double score = correlate(glyph, sample);
            if (score > best_score) {
                best_score = score;
                best_name = name;
            }
        }
    }
    const double confidence = std::max(0.0, best_score);
    // Если совпадение слабое, цифра не опознана: честнее сказать «?».
    if (confidence < kMinDigitConfidence) return {"?", confidence};
    return {best_name, confidence};
}

DigitReader::Reading DigitReader::read(const cv::Mat& region) const {
    Reading reading;
    const std::vector<cv::Mat> found = glyphs(region);
    if (found.empty()) return reading;
    std::string text;
    double total = 0.0;
    for (const cv::Mat& glyph : found) {
        const auto [name, score] = match(glyph);
        text += name;
        total += score;
        reading.glyphs.emplace_back(name, score);
    }
    reading.text = text;
    reading.confidence = total / static_cast<double>(found.size());
    reading.ok = !text.empty() && text.find('?') == std::string::npos;
    if (reading.ok) {
        try {
            reading.value = std::stod(text);
        } catch (...) {
            reading.value = 0.0;
        }
    }
    return reading;
}

std::vector<cv::Mat> DigitReader::glyphs_of(const cv::Mat& region) const {
    return glyphs(region);
}

std::string DigitReader::save_sample(const std::string& digit, const cv::Mat& glyph,
                                     const std::string& directory) const {
    const std::string dir = directory.empty() ? directory_ : directory;
    if (dir.empty() || digit.empty()) return {};
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // Имя не должно совпасть с уже существующим: образцов может быть
    // много, поэтому номер считаем по файлам на диске.
    int index = 0;
    if (std::filesystem::exists((std::filesystem::path(dir) / (digit + ".png")).string())) {
        index = 1;
        while (std::filesystem::exists(
            (std::filesystem::path(dir) / (digit + "_" + std::to_string(index) + ".png")).string())) {
            ++index;
        }
    }
    const std::string name = index == 0 ? digit + ".png" : digit + "_" + std::to_string(index) + ".png";
    const std::string path = (std::filesystem::path(dir) / name).string();
    if (!cv::imwrite(path, glyph)) return {};
    return path;
}

bool DigitReader::has_glyphs(const cv::Mat& region) const {
    return !glyphs(region).empty();
}

// ---------------------------------------------------------------- HUD

Hud::Hud(const std::string& assets_dir, Layout layout) : layout_(std::move(layout)) {
    digits_.load(assets_dir + "/" + layout_.digits_dir);
    for (const auto& entry : std::filesystem::directory_iterator(assets_dir + "/ui")) {
        if (entry.path().extension() != ".png") continue;
        const std::string stem = entry.path().stem().string();
        cv::Mat image;
        if (stem.rfind("currency_", 0) == 0) {
            image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (!image.empty()) currency_templates_.push_back(image);
            continue;
        }
        if (stem.rfind("text_", 0) == 0) {
            image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (!image.empty()) text_templates_[stem.substr(5)] = image;
        }
        if (stem == "result_victory") {
            image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (!image.empty()) victory_ribbon_ = image;
        }
    }
}

DigitReader::Reading Hud::read_mana_digits(const cv::Mat& canvas) const {
    return digits_.read(crop(canvas, layout_.energy));
}

std::vector<cv::Mat> Hud::mana_glyphs(const cv::Mat& canvas) const {
    return digits_.glyphs_of(crop(canvas, layout_.energy));
}

double Hud::read_mana(const cv::Mat& canvas, double* confidence) const {
    const DigitReader::Reading reading = read_mana_digits(canvas);
    if (reading.ok && reading.value > 0) {
        if (confidence != nullptr) *confidence = reading.confidence;
        return reading.value;
    }
    // Запасной вариант: доля заполнения зелёной шкалы. Область берём из
    // раскладки: у веб-клиента и у мобильного шкала стоит по-разному.
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    cv::Mat patch = crop(hsv, layout_.mana_bar);
    if (patch.empty()) {
        if (confidence != nullptr) *confidence = 0.0;
        return 0.0;
    }
    cv::Mat mask;
    cv::inRange(patch, cv::Scalar(35, 90, 60), cv::Scalar(85, 255, 255), mask);
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        if (confidence != nullptr) *confidence = 0.0;
        return 0.0;
    }
    int first = -1;
    int last = -1;
    for (int x = 0; x < mask.cols; ++x) {
        if (cv::countNonZero(mask.col(x)) > 0) {
            if (first < 0) first = x;
            last = x;
        }
    }
    if (first < 0) {
        if (confidence != nullptr) *confidence = 0.0;
        return 0.0;
    }
    const double fraction = std::min(1.0, (last - first + 1) / static_cast<double>(patch.cols - 2));
    if (confidence != nullptr) *confidence = fraction;
    return fraction * kManaMax;
}

bool Hud::has_energy_flask(const cv::Mat& canvas) const {
    return has_energy_flask(canvas, layout_);
}

bool Hud::has_energy_flask(const cv::Mat& canvas, const Layout& layout) const {
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    cv::Mat region = crop(hsv, layout.energy_flask);
    if (region.empty()) return false;
    cv::Mat mask;
    cv::inRange(region, cv::Scalar(85, 90, 130), cv::Scalar(125, 255, 255), mask);
    // В бою флакон занимает почти половину области. Меньше — это не флакон,
    // а посторонняя голубая мелочь (например, на экране итогов): иначе бот
    // решит, что идёт бой, и будет кликать не туда.
    return cv::countNonZero(mask) * 1.0 / mask.total() > 0.15;
}

bool Hud::summon_ready(const cv::Mat& canvas) const {
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    cv::Mat region = crop(hsv, layout_.summon);
    if (region.empty()) return false;
    cv::Mat mask;
    cv::inRange(region, cv::Scalar(8, 120, 120), cv::Scalar(26, 255, 255), mask);
    return cv::countNonZero(mask) * 1.0 / mask.total() > 0.03;
}

cv::Point Hud::summon_button(const cv::Mat& canvas) const {    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    cv::Mat region = crop(hsv, layout_.summon);
    if (!region.empty()) {
        cv::Mat mask;
        cv::inRange(region, cv::Scalar(8, 120, 120), cv::Scalar(26, 255, 255), mask);
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        int best = -1;
        double best_area = 0.0;
        for (int i = 1; i < count; ++i) {
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            const double area = stats.at<double>(i, cv::CC_STAT_AREA);
            if (w < 40 || h < 20) continue;
            if (area > best_area) {
                best_area = area;
                best = i;
            }
        }
        if (best > 0) {
            const int x = stats.at<int>(best, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(best, cv::CC_STAT_TOP);
            const int w = stats.at<int>(best, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(best, cv::CC_STAT_HEIGHT);
            // Жмём по карточке — она в левой части виджета.
            return {layout_.summon.x + x + w / 3, layout_.summon.y + y + h / 2};
        }
    }
    return {layout_.summon.x + layout_.summon.width / 3, layout_.summon.y + layout_.summon.height / 2};
}

// Кнопка умения героя: круглая, справа внизу над колодой.
//
// Пока идёт откат, кольцо вокруг кнопки заполняется по дуге (желтого цвета
// становится больше). Когда откат кончился, кольцо замыкается и вокруг
// него появляется жёлтое свечение — по нему и определяем готовность.
cv::Point Hud::ability_button(const cv::Mat& canvas) const {
    (void)canvas;
    const cv::Rect& rect = layout_.ability;
    if (rect.width <= 0 || rect.height <= 0) return {-1, -1};
    return {rect.x + rect.width / 2, rect.y + rect.height / 2};
}

bool Hud::ability_ready(const cv::Mat& canvas) const {
    const cv::Rect& rect = layout_.ability;
    if (rect.width <= 0 || rect.height <= 0) return false;
    const cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
    const double radius = rect.width / 2.0;
    // Свечение — тонкое кольцо сразу за ободом отката.
    const double inner = radius * 1.02;
    const double outer = radius * 1.22;
    const cv::Rect band(center.x - static_cast<int>(outer), center.y - static_cast<int>(outer),
                        static_cast<int>(2 * outer), static_cast<int>(2 * outer));
    cv::Mat hsv;
    cv::cvtColor(crop(canvas, band), hsv, cv::COLOR_BGR2HSV);
    if (hsv.empty()) return false;
    int yellow = 0;
    int total = 0;
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const double dx = band.x + x - center.x;
            const double dy = band.y + y - center.y;
            const double d = std::hypot(dx, dy);
            if (d < inner || d >= outer) continue;
            ++total;
            const cv::Vec3b& pixel = hsv.at<cv::Vec3b>(y, x);
            const int h = pixel[0];
            const int s = pixel[1];
            const int v = pixel[2];
            if (h >= 15 && h <= 40 && s > 140 && v > 170) ++yellow;
        }
    }
    if (total < 200) return false;
    // Готовое умение даёт заметное свечение (больше 10 % кольца), при
    // откате его нет вовсе.
    return yellow * 10 > total;
}

int Hud::battle_result(const cv::Mat& canvas) const {
    const cv::Rect& band = layout_.result_banner;
    if (band.width <= 0 || band.height <= 0) return 0;
    // Победа: лента «ПОБЕДА» — точный шаблон. По одному цвету ленту не
    // отличить от синего фона главного меню, поэтому сверяем картинку.
    if (!victory_ribbon_.empty()) {
        const double scale = layout_.canvas_size.height > 0
                                 ? static_cast<double>(canvas.rows) / layout_.canvas_size.height
                                 : 1.0;
        cv::Mat ribbon = victory_ribbon_;
        if (std::abs(scale - 1.0) > 0.02) {
            cv::resize(ribbon, ribbon, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        const cv::Rect search(band.x - 10, band.y - 20, band.width + 20, band.height + 40);
        const cv::Rect clipped = search & cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (clipped.width > ribbon.cols && clipped.height > ribbon.rows) {
            const cv::Mat region = canvas(clipped);
            cv::Mat response;
            cv::matchTemplate(region, ribbon, response, cv::TM_CCOEFF_NORMED);
            double best = 0.0;
            cv::minMaxLoc(response, nullptr, &best, nullptr, nullptr);
            if (best > 0.55) return 1;
        }
    }
    // Поражение: лента красная. Отличаем от прочих экранов тем, что лента
    // — широкая полоса одного оттенка, а в ней белый текст.
    cv::Mat hsv;
    cv::cvtColor(crop(canvas, band), hsv, cv::COLOR_BGR2HSV);
    cv::Mat sat;
    cv::inRange(hsv, cv::Scalar(0, 110, 110), cv::Scalar(179, 255, 255), sat);
    std::vector<int> hues;
    int white = 0;
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const cv::Vec3b& pixel = hsv.at<cv::Vec3b>(y, x);
            if (pixel[1] < 60 && pixel[2] > 200) ++white;
            if (sat.at<uchar>(y, x) == 0) continue;
            hues.push_back(pixel[0]);
        }
    }
    if (hues.size() < 1000 || white < 300) return 0;
    std::sort(hues.begin(), hues.end());
    const int hue = hues[hues.size() / 2];
    // Красный (или оранжевый) оттенок — поражение.
    if (hue <= 22 || hue >= 165) return -1;
    return 0;
}

cv::Rect Hud::deck_card_rect(int position) const {
    // Карточки делят ряд поровну, по порядку колоды.
    if (position < 1 || layout_.card_count < 1) return {};
    const cv::Rect& row = layout_.card_row;
    const int width = row.width / layout_.card_count;
    const int x = row.x + (position - 1) * width;
    if (x < row.x || x + width > row.x + row.width) return {};
    return cv::Rect(x, row.y, width, row.height);
}

double Hud::deck_slot_saturation(const cv::Mat& canvas, int position) const {
    const cv::Rect slot = deck_card_rect(position);
    const cv::Rect inner(slot.x + slot.width / 6, slot.y + slot.height / 6,
                         slot.width * 2 / 3, slot.height * 2 / 3);
    const cv::Rect bounded = inner & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (bounded.area() <= 0) return 0.0;
    cv::Mat hsv;
    cv::cvtColor(canvas(bounded), hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    return cv::mean(channels[1])[0];
}

bool Hud::deck_slot_ready(const cv::Mat& canvas, int position) const {
    // Порог с живого замера: цветная карточка ~80..140, серая (откат) < 20.
    return deck_slot_saturation(canvas, position) >= 58.0;
}

bool Hud::in_battle(const cv::Mat& canvas) const {
    return in_battle(canvas, layout_);
}

bool Hud::in_battle(const cv::Mat& canvas, const Layout& layout) const {
    // Число энергии в HUD, флакон рядом и ряд карточек юнитов.
    if (!digits_.has_glyphs(crop(canvas, layout.energy))) return false;
    if (!has_energy_flask(canvas, layout)) return false;
    cv::Mat gray;
    cv::cvtColor(crop(canvas, layout.card_row), gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 140);
    cv::Mat column_sum;
    cv::reduce(edges, column_sum, 0, cv::REDUCE_SUM, CV_32S);
    const int height = column_sum.rows;
    std::vector<int> peaks;
    for (int x = 0; x < column_sum.cols; ++x) {
        if (column_sum.at<int>(x) > height * 0.25) peaks.push_back(x);
    }
    if (peaks.empty()) return false;
    std::vector<std::vector<int>> groups;
    for (int p : peaks) {
        if (!groups.empty() && p - groups.back().back() <= 8) {
            groups.back().push_back(p);
        } else {
            groups.push_back({p});
        }
    }
    int cards = 0;
    for (const auto& group : groups) {
        if (group.back() - group.front() >= 25) ++cards;
    }
    return cards >= 4 && cards <= 7;
}

std::vector<UiElement> Hud::find_buttons(const cv::Mat& canvas) const {
    return find_buttons(canvas, layout_);
}

std::vector<UiElement> Hud::find_buttons(const cv::Mat& canvas,
                                          const Layout& layout) const {
    std::vector<UiElement> found;
    if (canvas.empty()) return found;
    const double scale = layout.scale;
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    for (const ButtonRule& rule : button_rules()) {
        cv::Mat mask = cv::Mat::zeros(canvas.size(), CV_8UC1);
        for (const auto& [lo, hi] : rule.hue_ranges) {
            cv::Mat part;
            cv::inRange(hsv, cv::Scalar(lo, rule.sat_min, rule.val_min),
                        cv::Scalar(hi, rule.sat_max, rule.val_max), part);
            cv::bitwise_or(mask, part, mask);
        }
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        for (int i = 1; i < count; ++i) {
            const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(i, cv::CC_STAT_TOP);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < rule.min_area * scale * scale) continue;
            if (w < rule.min_w * scale || h < rule.min_h * scale) continue;
            if (w > rule.max_w * scale || h > rule.max_h * scale) continue;
            if (w * h > layout.canvas_size.width * layout.canvas_size.height * 0.6) continue;
            // Кнопка — плотная заливка. Мягкое свечение и картинка дают
            // пятнистую область, и такие места бот нажимать не должен.
            if (rule.min_fill > 0.0) {
                if (area < w * h * rule.min_fill) continue;
                // Кнопка шире, чем высока: «PvP» 253x86, «Заработать» 152x62.
                // Картинка на заставке (например, оранжевая рубашка героя)
                // такому признаку не отвечает.
                if (w * 10 < h * 12) continue;
            }
            found.push_back({rule.name, x, y, w, h});
        }
    }
    return found;
}

bool Hud::is_close_badge(const cv::Mat& canvas, const UiElement& element) const {
    // Крестик закрытия окна, а не бейдж скидки «-90%».
    //
    // Признаки два. Первый: у крестика красного много (0.55-0.60), у
    // бейджа почти нет — там белые цифры. Второй, надёжнее: белый глиф
    // внутри крестика квадратный (знак «X»), а у бейджа это широкая
    // строка текста (h/w ≈ 0.6 против 1.0 у крестика).
    cv::Mat region = crop(canvas, cv::Rect(element.x, element.y, element.w, element.h));
    if (region.empty()) return false;
    cv::Mat hsv;
    cv::cvtColor(region, hsv, cv::COLOR_BGR2HSV);
    cv::Mat low;
    cv::Mat high;
    cv::inRange(hsv, cv::Scalar(0, 120, 110), cv::Scalar(10, 255, 255), low);
    cv::inRange(hsv, cv::Scalar(165, 120, 110), cv::Scalar(180, 255, 255), high);
    cv::Mat red;
    cv::bitwise_or(low, high, red);
    if (cv::countNonZero(red) * 1.0 / red.total() < 0.35) return false;

    cv::Mat white_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 170), cv::Scalar(180, 90, 255), white_mask);
    // Берём крупнейшую белую фигуру внутри красного значка: у крестика
    // это сам «X», у бейджа «-90%» — цифры, у значка в верхней полосе HUD —
    // целая картинка. Раньше брались все светлые пиксели сразу, и любой
    // светлый мусор рядом делал значок «квадратным».
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(white_mask, labels, stats, centroids, 8);
    if (count < 2) return false;
    int best = 1;
    for (int i = 2; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) best = i;
    }
    const int box_w = stats.at<int>(best, cv::CC_STAT_WIDTH);
    const int box_h = stats.at<int>(best, cv::CC_STAT_HEIGHT);
    const int area = stats.at<int>(best, cv::CC_STAT_AREA);
    if (box_w <= 0 || box_h <= 0) return false;
    // Знак «X» заполняет свой прямоугольник почти целиком, а картинка
    // внутри значка — нет.
    if (area * 10 < box_w * box_h * 6) return false;
    // И сам знак занимает заметную часть кружка: у красного значка в
    // верхней полосе HUD белый глиф мельче и не квадратный.
    const int button_w = std::max(1, element.w);
    const int button_h = std::max(1, element.h);
    if (box_w * 10 < button_w * 4 || box_h * 10 < button_h * 4) return false;
    const double aspect = static_cast<double>(box_h) / box_w;
    return aspect >= 0.75 && aspect <= 1.4;
}

bool Hud::is_purchase(const cv::Mat& canvas, const UiElement& element) const {
    cv::Mat region = crop(canvas, cv::Rect(element.x, element.y, element.w, element.h));
    if (region.empty()) return false;
    cv::Mat hsv;
    cv::cvtColor(region, hsv, cv::COLOR_BGR2HSV);
    cv::Mat gold;
    cv::inRange(hsv, cv::Scalar(18, 170, 150), cv::Scalar(40, 255, 255), gold);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(gold, labels, stats, centroids, 8);
    bool is_gold = false;
    for (int i = 1; i < count; ++i) {
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (w >= 60 && h >= 25 && area >= 1200) {
            is_gold = true;
            break;
        }
    }
    if (!is_gold) return false;

    // Золотая кнопка — это покупка, и бот её не нажимает никогда.
    //
    // Раньше здесь искали признак цены (знак валюты или цифры), но на
    // живых экранах это не работало: знак валюты совпадал с шаблоном всего
    // на 0.42 — ровно как на любой другой кнопке, а цифры в маленьком
    // шрифте («0,90 €») после уменьшения кадра не узнавались вовсе.
    // Отличить «Играть» в подземельях (без цены) от покупки по цвету не
    // выходит — зато можно просто не жать золотые кнопки: боту они не
    // нужны, он играет в PvP и Co-Op, где кнопки находятся по надписи.
    return true;
}

bool Hud::find_text(const cv::Mat& canvas, const std::string& name, cv::Rect* out,
                     double threshold) const {
    const auto it = text_templates_.find(name);
    if (it == text_templates_.end() || canvas.empty()) return false;
    const cv::Mat& pattern = it->second;
    if (pattern.rows > canvas.rows || pattern.cols > canvas.cols) return false;
    cv::Mat result;
    cv::matchTemplate(canvas, pattern, result, cv::TM_CCOEFF_NORMED);
    double max_value = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_value, nullptr, &max_loc);
    if (max_value < threshold) return false;
    *out = cv::Rect(max_loc.x, max_loc.y, pattern.cols, pattern.rows);
    return true;
}

Hud::CoopScreen Hud::coop_screen(const cv::Mat& canvas, cv::Rect* out,
                                 bool* gold_play) const {
    if (canvas.empty() || out == nullptr) return CoopScreen::kNone;
    if (gold_play != nullptr) *gold_play = false;
    const Layout layout = canonical_layout();
    cv::Mat work;
    if (canvas.size() == layout.canvas_size) {
        work = canvas;
    } else {
        const bool shrinking = layout.canvas_size.width <= canvas.cols;
        cv::resize(canvas, work, layout.canvas_size, 0, 0,
                   shrinking ? cv::INTER_AREA : cv::INTER_CUBIC);
    }

    // Выбор модификатора.
    //
    // Два дизайна экрана:
    //
    // * новый (после обновления игры): тёмная панель «Выберите
    //   модификатор:» с двумя серо-синими карточками; жмём значок первой
    //   карточки;
    // * старый: одна широкая светлая карточка с описанием и значком
    //   модификатора слева.
    //
    // Экран появляется уже в бою (над полем), поэтому проверять его надо
    // раньше, чем «идёт ли бой»: иначе бот начнёт играть, не выбрав
    // модификатор, и потеряет первые волны.
    {
        cv::Mat hsv;
        cv::cvtColor(work, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(100, 60, 120), cv::Scalar(125, 140, 190), mask);
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        // Карточки модификатора — крупные серо-синие прямоугольники. Размеры
        // заданы в пикселях веб-клиента, как и в find_buttons: на мобильном
        // клиенте они примерно в 1.93 раза больше.
        std::vector<cv::Rect> cards;
        for (int i = 1; i < count; ++i) {
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (w < 110 * layout.scale || w > 150 * layout.scale) continue;
            if (h < 100 * layout.scale || h > 140 * layout.scale) continue;
            // Порог площади: значок модификатора вырезает из карточки
            // кусок, поэтому площадь заметно меньше прямоугольника.
            if (area < 9000 * layout.scale * layout.scale) continue;
            cards.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT),
                               stats.at<int>(i, cv::CC_STAT_TOP), w, h);
        }
        if (!cards.empty()) {
            // Самая левая карточка; жмём её значок — он в верхней части
            // карточки, по центру.
            const cv::Rect card = *std::min_element(
                cards.begin(), cards.end(),
                [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
            *out = cv::Rect(card.x + card.width / 4, card.y + card.height / 6,
                            card.width / 2, card.height / 3);
            return CoopScreen::kModifier;
        }
    }

    // Старый дизайн: светлая карточка с описанием в нижней части экрана,
    // слева в ней — значок модификатора.
    {
        cv::Mat hsv;
        cv::cvtColor(work, hsv, cv::COLOR_BGR2HSV);
        // Светлая панель: светлота высокая, насыщенность низкая.
        cv::Mat light;
        cv::inRange(hsv, cv::Scalar(0, 0, 170), cv::Scalar(180, 60, 255), light);
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int count = cv::connectedComponentsWithStats(light, labels, stats, centroids, 8);
        const int width = work.cols;
        const int height = work.rows;
        int best = -1;
        int best_area = 0;
        for (int i = 1; i < count; ++i) {
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            // Карточка модификатора — широкая светлая панель внизу экрана.
            if (w < width * 0.30 || w > width * 0.95) continue;
            if (h < height * 0.07 || h > height * 0.25) continue;
            if (stats.at<int>(i, cv::CC_STAT_TOP) < height * 0.55) continue;
            if (area > best_area) {
                best_area = area;
                best = i;
            }
        }
        if (best > 0) {
            const int x = stats.at<int>(best, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(best, cv::CC_STAT_TOP);
            const int w = stats.at<int>(best, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(best, cv::CC_STAT_HEIGHT);
            // Значок модификатора — красный квадрат внутри карточки. Жмём его:
            // по тексту описания нажатие не срабатывает.
            const cv::Rect card(x, y, w, h);
            const cv::Mat icon_mask = red_blobs(work, card);
            int icon = -1;
            int icon_area = 0;
            cv::Mat icon_stats;
            {
                cv::Mat labels2;
                cv::Mat centroids2;
                const int n2 = cv::connectedComponentsWithStats(icon_mask, labels2, icon_stats,
                                                              centroids2, 8);
                for (int i = 1; i < n2; ++i) {
                    const int a = icon_stats.at<int>(i, cv::CC_STAT_AREA);
                    if (a < h * h * 0.05) continue;
                    if (a > icon_area) {
                        icon_area = a;
                        icon = i;
                    }
                }
            }
            if (icon > 0) {
                *out = cv::Rect(icon_stats.at<int>(icon, cv::CC_STAT_LEFT),
                                icon_stats.at<int>(icon, cv::CC_STAT_TOP),
                                icon_stats.at<int>(icon, cv::CC_STAT_WIDTH),
                                icon_stats.at<int>(icon, cv::CC_STAT_HEIGHT));
            } else {
                // Значка не нашли — жмём левый верхний угол карточки.
                *out = cv::Rect(x, y, std::max(4, w / 10), std::max(4, h / 2));
            }
            return CoopScreen::kModifier;
        }
    }

    if (trophy_road(work, nullptr)) {
        return CoopScreen::kNone;
    }
    const Layout canon = canonical_layout();
    const std::vector<UiElement> buttons = find_buttons(work, canon);

    // Выбор союзника: сначала любая карта сокровищ внизу, потом «Случайный»
    // (он стартует бой). Без выбора карты кооп идёт «без карты».
    cv::Rect random_text;
    if (find_text(work, "random", &random_text, 0.85)) {
        if (treasure_map_selected(work, buttons)) {
            *out = random_text;
            return CoopScreen::kAlly;
        }
        cv::Rect map;
        bool map_gold = false;
        if (treasure_map(work, buttons, &map, &map_gold)) {
            *out = map;
            if (gold_play != nullptr) *gold_play = map_gold;
            return CoopScreen::kAlly;
        }
        *out = random_text;
        return CoopScreen::kAlly;
    }

    // Главное меню тоже полное золотых пятен (PvP, сундуки, акции) —
    // без этой проверки бот «жмёт Играть золотым билетом» прямо в меню.
    cv::Rect menu_text;
    if (find_text(work, "pvp", &menu_text) || find_text(work, "coop", &menu_text)) {
        return CoopScreen::kNone;
    }

    // Экран итогов: внизу «Продолжить». Шаблон «Играть» там ложно
    // совпадает (~0.85), и бот кликал мимо кнопки выхода.
    for (const UiElement& b : buttons) {
        if (b.kind != "continue") continue;
        if (b.y < work.rows * 0.88) continue;
        if (std::abs(b.center().x - work.cols / 2) > work.cols * 0.14) continue;
        if (b.w < work.cols * 0.12 || b.w > work.cols * 0.30) continue;
        return CoopScreen::kNone;
    }

    // Список этажей: жмём «Играть». Два билета — обычный (синяя кнопка,
    // шаблон совпадает на ~0.99) и золотой (кнопка жёлтая, шаблон ~0.29,
    // поэтому ищем её цветом). Обычный берём первым: золото тратить не
    // нужно, пока есть бесплатный вход. Если синей нет — золотая.
    //
    // Надпись ищем справа: на этажах кнопка стоит в ряду справа, а ложные
    // совпадения на трофейной дороге попадали в левый столбец кубков.
    cv::Rect play_text;
    // «Играть» на этажах — в середине экрана справа. На лаунчере Android
    // шаблон ложно совпадает внизу (~y 0.95) — отсекаем по полосе.
    if (find_text(work, "play", &play_text, 0.85) && play_text.x > work.cols * 0.35 &&
        play_text.y > work.rows * 0.20 && play_text.y < work.rows * 0.72) {
        *out = play_text;
        return CoopScreen::kFloors;
    }
    std::vector<UiElement> gold_plays;
    for (const UiElement& b : buttons) {
        if (b.kind != "fight") continue;
        if (b.y < work.rows * 0.28 || b.y > work.rows * 0.72) continue;
        if (b.x < work.cols * 0.40) continue;
        if (b.w < work.cols * 0.16 || b.w > work.cols * 0.45) continue;
        if (b.h < work.rows * 0.03 || b.h > work.rows * 0.10) continue;
        gold_plays.push_back(b);
    }
    if (!gold_plays.empty()) {
        const UiElement& top = *std::min_element(
            gold_plays.begin(), gold_plays.end(),
            [](const UiElement& a, const UiElement& b) { return a.y < b.y; });
        *out = cv::Rect(top.x, top.y, top.w, top.h);
        if (gold_play != nullptr) *gold_play = true;
        return CoopScreen::kFloors;
    }
    if (find_text(work, "play", &play_text, 0.28) && play_text.x > work.cols * 0.35 &&
        play_text.y > work.rows * 0.20 && play_text.y < work.rows * 0.72) {
        *out = play_text;
        if (gold_play != nullptr) *gold_play = true;
        return CoopScreen::kFloors;
    }

    return CoopScreen::kNone;
}

std::vector<UiElement> Hud::coop_buttons(const std::vector<UiElement>& buttons,
                                          const std::vector<UiElement>& pvps,
                                          const Layout& layout,
                                          const UiElement* exclude) const {
    // Кнопка Co-Op — синяя, справа от оранжевой PvP. Обе стоят в один ряд и
    // в маске синего цвета склеиваются в одну компоненту, поэтому кнопку
    // Co-Op вычисляем по геометрии: это часть синей полосы правее PvP.
    std::vector<UiElement> found;
    const UiElement* pvp = pvps.empty() ? nullptr : &*std::min_element(
        pvps.begin(), pvps.end(), [](const UiElement& a, const UiElement& b) { return a.x < b.x; });
    for (const UiElement& b : buttons) {
        if (b.kind != "continue" || !(560 < b.y && b.y < 670)) continue;
        // Кнопка внутри модального окна — не «Co-Op» из главного меню.
        if (exclude != nullptr && b.x == exclude->x && b.y == exclude->y) continue;
        if (b.center().x <= layout.canvas_size.width * 0.58) continue;
        if (pvp != nullptr && b.x < pvp->x + pvp->w) {
            const int left = pvp->x + pvp->w + 5;
            if (left < b.x + b.w) {
                found.push_back({"coop", left, b.y, b.x + b.w - left, b.h});
            }
            continue;
        }
        found.push_back(b);
    }
    return found;
}

// Раскладка, в которой настроены правила поиска кнопок и шаблоны надписей.
//
// Раскладку приводят к настоящему размеру кадра (Layout::scaled_to): если
// устройство другого размера, кадр приходит растянутым, а правила к нему
// не подходят (пороги заданы в пикселях канонического холста, и шаблоны
// надписей на растянутом кадре не находятся). Поэтому разбор всегда идёт в
// каноническом размере, а найденные элементы пересчитываются обратно в
// координаты кадра.
Layout Hud::canonical_layout() const {
    const cv::Size size = canonical_canvas_size(layout_);
    if (size == layout_.canvas_size) return layout_;
    return layout_.scaled_to(size);
}

std::vector<UiElement> Hud::navigation_candidates(const cv::Mat& canvas) const {
    const Layout layout = canonical_layout();
    cv::Mat work;
    if (canvas.size() == layout.canvas_size) {
        work = canvas;
    } else {
        // При уменьшении — INTER_AREA (сглаживает), при увеличении —
        // INTER_CUBIC: INTER_AREA при увеличении работает почти как nearest.
        const bool shrinking = layout.canvas_size.width <= canvas.cols;
        cv::resize(canvas, work, layout.canvas_size, 0, 0,
                   shrinking ? cv::INTER_AREA : cv::INTER_CUBIC);
    }
    std::vector<UiElement> out = navigation_candidates_at(work, layout);
    if (canvas.size() == layout.canvas_size) return out;
    // Найденные элементы — обратно в координаты кадра.
    const double sx = static_cast<double>(canvas.cols) / layout.canvas_size.width;
    const double sy = static_cast<double>(canvas.rows) / layout.canvas_size.height;
    for (UiElement& element : out) {
        element.x = static_cast<int>(std::lround(element.x * sx));
        element.y = static_cast<int>(std::lround(element.y * sy));
        element.w = static_cast<int>(std::lround(element.w * sx));
        element.h = static_cast<int>(std::lround(element.h * sy));
    }
    return out;
}

std::vector<UiElement> Hud::navigation_candidates_at(const cv::Mat& canvas,
                                                       const Layout& layout) const {
    const std::vector<UiElement> buttons = find_buttons(canvas, layout);
    std::vector<UiElement> out;

    // В бою меню не разбираем: кнопки призыва и виджет — не элементы меню,
    // а жёлтые пятна на картинке арены иной раз сойдут за кнопку.
    if (in_battle(canvas, layout)) {
        out.push_back({"tap", layout.canvas_size.width / 2 - 10,
                       layout.canvas_size.height / 2 - 10, 20, 20});
        return out;
    }

    UiElement trophy_back;
    if (trophy_road(canvas, &trophy_back)) {
        // Нижняя синяя полоса есть и в главном меню (вкладки). Там же
        // стоят «PvP» и «Co-Op» — это не трофейная дорога, «назад» отсюда
        // спрашивает «выйти из игры?».
        cv::Rect menu_text;
        const bool main_menu =
            (find_text(canvas, "pvp", &menu_text) &&
             menu_text.y >= layout.canvas_size.height * 0.55) ||
            (find_text(canvas, "coop", &menu_text) &&
             menu_text.y >= layout.canvas_size.height * 0.55);
        if (!main_menu) {
            out.push_back(trophy_back);
            return out;
        }
    }

    // Кнопку «PvP» ищем по надписи: она надёжнее цвета — рядом с ней
    // много оранжевого (значки, рамки), и цветные пятна сливаются в один
    // большой блок.
    cv::Rect pvp_text;
    bool have_pvp_text = find_text(canvas, "pvp", &pvp_text);
    // Надпись «PvP» есть и на баннере подбора соперника — она там не
    // кнопка. Настоящая кнопка стоит внизу, в ряду с «Co-Op».
    if (have_pvp_text && pvp_text.y < layout.canvas_size.height * 0.55) have_pvp_text = false;
    // Кнопка «Co-Op» — тоже по надписи: серые пятна рядом с ней сливаются.
    cv::Rect coop_text;
    bool have_coop_text = find_text(canvas, "coop", &coop_text);
    if (have_coop_text && coop_text.y < layout.canvas_size.height * 0.55) have_coop_text = false;
    if (have_coop_text) {
        out.push_back({"coop", coop_text.x, coop_text.y, coop_text.width, coop_text.height});
    }

    // «Отмена» на экране подбора соперника: красно-оранжевая кнопка внизу
    // по центру. Идёт первым: это однозначная кнопка, в отличие от красных
    // значков, которые похожи на крестик.
    std::vector<UiElement> cancels;
    for (const UiElement& b : buttons) {
        if (b.kind == "cancel" && b.y > layout.canvas_size.height * 0.85 &&
            std::abs(b.center().x - layout.canvas_size.width / 2) < layout.canvas_size.width * 0.22) {
            cancels.push_back(b);
        }
    }
    if (!cancels.empty()) {
        out.push_back(*std::min_element(cancels.begin(), cancels.end(),
                                        [](const UiElement& a, const UiElement& b) {
                                            return a.y < b.y;
                                        }));
    }

    // Крестик закрытия модального окна: у правого края диалога.
    std::vector<UiElement> closes;
    for (const UiElement& b : buttons) {
        if (b.kind != "close") continue;
        if (b.y >= layout.canvas_size.height * 0.55) continue;
        if (b.x + b.w <= layout.canvas_size.width * layout.close_x_min) continue;
        if (b.w > 46) continue;
        if (!is_close_badge(canvas, b)) continue;
        closes.push_back(b);
    }

    // Синяя кнопка внутри диалога: «Понятно», «Продолжить», «Забрать».
    //
    // Отдельно ищем кнопку, которая лежит рядом с крестиком: это кнопка
    // самого окна (например, «Продолжить» в окне «Внимание!» на экране
    // итогов). Её надо нажать раньше крестика: крестик просто закрывает
    // окно, и бот снова оказывается на экране итогов — по кругу.
    UiElement dialog_ok;
    bool have_dialog_ok = false;
    if (!closes.empty()) {
        const UiElement& cross = *std::min_element(closes.begin(), closes.end(),
                                                    [](const UiElement& a, const UiElement& b) {
                                                        return a.y < b.y;
                                                    });
        for (const UiElement& b : buttons) {
            if (b.kind != "continue") continue;
            // Та же форма, что и у кнопок диалога: широкая плашка с
            // надписью. Квадратная кнопка (значок сундука) — не она.
            if (b.w < 50 * layout.scale || b.w > 260 * layout.scale) continue;
            if (b.h < 22 * layout.scale || b.h > 70 * layout.scale) continue;
            if (b.w * 10 < b.h * 13) continue;
            if (b.w > b.h * 4) continue;
            // Кнопка окна стоит под крестиком и недалеко от него.
            if (std::abs(b.center().x - cross.center().x) > layout.canvas_size.width * 0.32) continue;
            if (b.y < cross.y - 30) continue;
            if (b.y > cross.y + layout.canvas_size.height * 0.2) continue;
            if (!have_dialog_ok || b.y < dialog_ok.y) {
                dialog_ok = b;
                have_dialog_ok = true;
            }
        }
    }

    // PvP в главном меню — оранжевая кнопка. Считаем её заранее: по ней
    // отделяем склеенную с ней синюю кнопку Co-Op.
    std::vector<UiElement> pvps;
    for (const UiElement& b : buttons) {
        if (b.kind == "orange" && b.y > layout.canvas_size.height * 0.55) pvps.push_back(b);
    }

    const std::vector<UiElement> coops = coop_buttons(buttons, pvps, layout, have_dialog_ok
                                                                       ? &dialog_ok
                                                                       : nullptr);
    std::vector<UiElement> oks;
    for (const UiElement& b : buttons) {
        if (b.kind != "continue") continue;
        if (!(layout.canvas_size.height * 0.25 < b.y && b.y < layout.canvas_size.height * 0.85)) continue;
        if (b.w < 50 * layout.scale || b.w > 260 * layout.scale) continue;
        if (b.h < 22 * layout.scale || b.h > 70 * layout.scale) continue;
        // Кнопка с надписью шире, чем высока. Иконки нижней панели — наоборот,
        // квадратные или вытянутые вверх: их не путаем с кнопками.
        if (b.w * 10 < b.h * 13) continue;
        // Но и не бесконечно широкая: лента «Победа» (475x70) — не кнопка.
        if (b.w > b.h * 4) continue;
        bool overlaps_coop = false;
        for (const UiElement& c : coops) {
            if (b.x < c.x + c.w && c.x < b.x + b.w && b.y < c.y + c.h && c.y < b.y + b.h) {
                overlaps_coop = true;
                break;
            }
        }
        if (!overlaps_coop) oks.push_back(b);
    }
    // Синяя кнопка диалога важнее крестика: на экране итогов после нижней
    // «Продолжить» выскакивает попап с сундуком, и выйти из боя можно
    // только нажав «Продолжить» внутри него. Крестик этот попап закрывает,
    // и бот снова оказывается на экране итогов.
    if (have_dialog_ok) {
        // Плашка «Понятно» чуть выше самой кнопки — жмём нижнюю часть.
        out.push_back({"ok", dialog_ok.x, dialog_ok.y + static_cast<int>(dialog_ok.h * 0.7),
                       dialog_ok.w, static_cast<int>(dialog_ok.h * 0.3)});
    }
    if (!oks.empty()) {
        // Ищем кнопку окна. Обычно она одна, но бывают окна с двумя
        // синими кнопками («Подтвердить» и «Поддержка»): тогда берём
        // правую — так устроены диалоги игры (отмена слева, подтверждение
        // справа).
        const UiElement* best = &*std::min_element(
            oks.begin(), oks.end(), [&layout](const UiElement& a, const UiElement& b) {
                const int da = std::abs(a.center().x - layout.canvas_size.width / 2);
                const int db = std::abs(b.center().x - layout.canvas_size.width / 2);
                if (da != db) {
                    // Кнопки на равном расстоянии от середины: берём правую.
                    if (std::abs(da - db) <= layout.canvas_size.width * 0.02) return a.x > b.x;
                    return da < db;
                }
                return a.y < b.y;
            });
        // Плашка «Понятно» чуть выше самой кнопки — жмём нижнюю часть.
        out.push_back({"ok", best->x, best->y + static_cast<int>(best->h * 0.7), best->w,
                       static_cast<int>(best->h * 0.3)});
    }
    if (!closes.empty()) {
        out.push_back(*std::min_element(closes.begin(), closes.end(),
                                        [](const UiElement& a, const UiElement& b) {
                                            return a.y < b.y;
                                        }));
    }

    // Кнопка «назад» внутри игры — синяя плашка со стрелкой в левом нижнем
    // углу. Системная клавиша «Назад» тут не работает, а из «Подземелья» и
    // карточки героя выйти надо именно этой кнопкой. От иконок нижней панели
    // её отличает размер: стрелка заметно меньше.
    std::vector<UiElement> backs;
    for (const rr::UiElement& b : buttons) {
        if (b.kind != "continue") continue;
        if (b.x > layout.canvas_size.width * 0.22) continue;
        if (b.y < layout.canvas_size.height * 0.88) continue;
        if (b.w > layout.canvas_size.width * 0.09) continue;
        if (b.h > layout.canvas_size.height * 0.08) continue;
        if (b.w * 10 < b.h * 7 || b.w * 10 > b.h * 14) continue;
        backs.push_back({"back", b.x, b.y, b.w, b.h});
    }
    if (!backs.empty()) {
        out.push_back(*std::min_element(backs.begin(), backs.end(),
                                        [](const rr::UiElement& a, const rr::UiElement& b) {
                                            return a.x < b.x;
                                        }));
    }

    // «Продолжить» на экране итогов — внизу по центру. В нижней полосе
    // живут иконки меню (магазин, колода, щит, «Бой») — их не трогаем:
    // иконки квадратные или выше, чем шире, а «Продолжить» — широкая
    // кнопка с надписью.
    std::vector<UiElement> continues;
    for (const UiElement& b : buttons) {
        if (b.kind != "continue" || b.y <= layout.canvas_size.height * 0.85) continue;
        if (std::abs(b.center().x - layout.canvas_size.width / 2) >= layout.canvas_size.width * 0.14) continue;
        if (b.h > layout.canvas_size.height * 0.09) continue;
        // «Продолжить» — широкая кнопка с надписью (в 2.5 раза шире, чем
        // высока). Иконки нижней панели (97x71) проходили по более мягкому
        // признаку, и бот уходил в «Подземелья».
        if (b.w * 10 < b.h * 16) continue;
        continues.push_back(b);
    }
    if (!continues.empty()) {
        out.push_back(*std::min_element(continues.begin(), continues.end(),
                                        [&layout](const UiElement& a, const UiElement& b) {
                                            const int da = std::abs(a.center().x - layout.canvas_size.width / 2);
                                            const int db = std::abs(b.center().x - layout.canvas_size.width / 2);
                                            return da != db ? da < db : a.y > b.y;
                                        }));
    }

    // Золотая кнопка «Играть» (в подземельях она такая же, как кнопки
    // покупок, но без цены — покупки отсекаются проверкой is_purchase).
    std::vector<UiElement> plays;
    for (const UiElement& b : buttons) {
        if (b.kind != "fight") continue;
        if (b.w < layout.canvas_size.width * 0.16 || b.w > layout.canvas_size.width * 0.57) continue;
        if (b.h < layout.canvas_size.height * 0.03 || b.h > layout.canvas_size.height * 0.10) continue;
        if (b.y < layout.canvas_size.height * 0.30 || b.y > layout.canvas_size.height * 0.70) continue;
        if (is_purchase(canvas, b)) continue;
        plays.push_back(b);
    }
    if (!plays.empty()) {
        out.push_back(*std::min_element(plays.begin(), plays.end(),
                                        [](const UiElement& a, const UiElement& b) {
                                            return a.x < b.x;
                                        }));
    }

    if (!pvps.empty() && !have_pvp_text) {
        const UiElement& best = *std::min_element(pvps.begin(), pvps.end(),
                                                   [](const UiElement& a, const UiElement& b) {
                                                       return a.x < b.x;
                                                   });
        // Вид «orange», а не «pvp»: ждать матч имеет смысл только там,
        // где надпись «PvP» найдена точно.
        out.push_back({"orange", best.x, best.y, best.w, best.h});
    }
    if (!coops.empty()) {
        out.push_back(*std::min_element(coops.begin(), coops.end(),
                                        [](const UiElement& a, const UiElement& b) {
                                            return a.x < b.x;
                                        }));
    }

    // Верхняя полоса HUD — не трогаем: там валюты с кнопками «+» и значки
    // меню (в том числе «три полоски» — вход в меню игры). Исключение —
    // красные крестики модальных окон, они идут выше.
    std::vector<UiElement> filtered;
    for (const UiElement& b : out) {
        const bool top_strip = b.y < 130;
        const bool safe_kind = b.kind == "close" || b.kind == "cancel";
        if (top_strip && !safe_kind) continue;
        filtered.push_back(b);
    }

    // Клик в центр — для экранов «нажмите, чтобы продолжить». На главном
    // меню такой клик попадает в портрет героя и открывает его карточку —
    // там «tap» не нужен, есть кнопки PvP и Co-Op.
    //
    // Клик в центр предлагается, только если на экране нет других кнопок:
    // иначе бот вслепую попадает в панели (колода на экране итогов) и
    // вместо выхода из боя уходит в другие экраны.
    if (!have_pvp_text && !have_coop_text && out.empty()) {
        filtered.push_back({"tap", layout.canvas_size.width / 2 - 10,
                            layout.canvas_size.height / 2 - 10, 20, 20});
    }

    // Кнопки покупки не нажимаем никогда. Кнопку, найденную по надписи,
    // не фильтруем: она опознана буквами, а не цветом.
    std::vector<UiElement> safe;
    for (const UiElement& b : filtered) {
        if (!is_purchase(canvas, b)) safe.push_back(b);
    }
    if (have_pvp_text) {
        safe.insert(safe.begin(), {"pvp", pvp_text.x, pvp_text.y, pvp_text.width, pvp_text.height});
    }
    return safe;
}

}  // namespace rr
