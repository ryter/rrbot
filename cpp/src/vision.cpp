#include "vision.hpp"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

namespace rr {
namespace {

// Медиана списка (как numpy.median: для чётного размера — среднее двух
// центральных значений).
double median_of(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2 == 1) return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Отрезок оттенка (в шкале OpenCV 0..180). Диапазон может проходить через 0.
bool hue_in_ranges(double hue, const std::vector<std::pair<double, double>>& ranges) {
    for (const auto& [low, high] : ranges) {
        if (low <= high) {
            if (low <= hue && hue <= high) return true;
        } else if (hue >= low || hue <= high) {
            return true;
        }
    }
    return false;
}

// Прямоугольник находки в виде четырёх чисел.
struct Box {
    double x1, y1, x2, y2;
};

Box box_of(const Detection& det) {
    return {static_cast<double>(det.box.x), static_cast<double>(det.box.y),
            static_cast<double>(det.box.x + det.box.width),
            static_cast<double>(det.box.y + det.box.height)};
}

// Доля перекрытия двух прямоугольников.
double iou(const Box& a, const Box& b) {
    const double ix1 = std::max(a.x1, b.x1);
    const double iy1 = std::max(a.y1, b.y1);
    const double ix2 = std::min(a.x2, b.x2);
    const double iy2 = std::min(a.y2, b.y2);
    const double iw = std::max(0.0, ix2 - ix1);
    const double ih = std::max(0.0, iy2 - iy1);
    const double inter = iw * ih;
    if (inter <= 0.0) return 0.0;
    const double area_a = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
    const double area_b = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
    const double uni = area_a + area_b - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

// ---------------------------------------------------------------- основания

// Окружность-кандидат: центр, радиус и насколько круг похож на основание.
struct RawCircle {
    cv::Point center;
    int radius = 0;
    double quality = 0.0;
};

// Насколько круг похож на настоящее основание юнита.
//
// У основания юнита граница круглая почти по всей окружности (спрайт
// закрывает только верхнюю часть основания), а у описанной окружности
// квадратной плитки граница есть только по углам. Это отличает настоящее
// основание от мусора.
//
// Возвращает долю углов (0..1), на которых радиус круга — настоящая
// граница изображения (и граница эта достаточно контрастная).
double circle_support(const cv::Mat& gray, const cv::Point& center, int radius, int angles = 24) {
    if (gray.empty() || radius < 6) return 0.0;
    int supported = 0;
    int counted = 0;
    for (int i = 0; i < angles; ++i) {
        const double angle = 2.0 * M_PI * i / angles;
        const double ca = std::cos(angle);
        const double sa = std::sin(angle);
        const int step_x = static_cast<int>(std::lround(ca));
        const int step_y = static_cast<int>(std::lround(sa));
        double best = 0.0;
        int best_radius = -1;
        for (int rr = radius - 3; rr <= radius + 3; ++rr) {
            if (rr < 4) continue;
            const int x = static_cast<int>(std::lround(center.x + rr * ca));
            const int y = static_cast<int>(std::lround(center.y + rr * sa));
            // Края: сравниваем именно те пиксели, которые читаем.
            const int xa = x + step_x;
            const int ya = y + step_y;
            const int xb = x - step_x;
            const int yb = y - step_y;
            if (xa < 0 || ya < 0 || xb < 0 || yb < 0 || xa >= gray.cols || ya >= gray.rows ||
                xb >= gray.cols || yb >= gray.rows) {
                continue;
            }
            const double g = std::abs(static_cast<double>(gray.at<uchar>(ya, xa)) -
                                      static_cast<double>(gray.at<uchar>(yb, xb)));
            if (g > best) {
                best = g;
                best_radius = rr;
            }
        }
        if (best_radius < 0) continue;
        ++counted;
        // Граница считается настоящей, если она контрастная и лежит
        // примерно на радиусе круга.
        if (best > 30.0 && std::abs(best_radius - radius) <= 1) ++supported;
    }
    if (counted == 0) return 0.0;
    return static_cast<double>(supported) / static_cast<double>(counted);
}

// Окружности на картинке: поиск по градиенту.
//
// Свой поиск вместо cv::HoughCircles: в сборке OpenCV, которой пользуется
// бот, преобразование Хафа находит на порядок меньше кругов, чем та же
// версия OpenCV из pip (IPP считает аккумулятор иначе) — на кадрах боя это
// разница между «найдено 20 юнитов» и «найден один».
//
// Работает так: по градиенту ищутся точки края, и каждая из них голосует за
// центры окружностей радиуса r, лежащие в направлении градиента. У края
// окружности градиент направлен по радиусу, поэтому центр находится сразу.
std::vector<RawCircle> detect_circles(const cv::Mat& gray, int radius_min, int radius_max,
                                       double min_distance, double min_score,
                                       double gradient_percentile, bool lower_half_only) {
    std::vector<RawCircle> out;
    if (gray.empty() || radius_max <= radius_min) return out;
    const int rows = gray.rows;
    const int cols = gray.cols;
    const int radii = radius_max - radius_min + 1;

    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    // Порог градиента: края оснований контрастные, а текстура травы — нет.
    // Порог берётся от картинки: доля самых сильных краёв.
    std::vector<float> values;
    values.reserve(static_cast<size_t>(rows) * cols / 4);
    for (int y = 0; y < rows; ++y) {
        const float* row = magnitude.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (row[x] > 60.0f) values.push_back(row[x]);
        }
    }
    if (values.empty()) return out;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(values.size() * gradient_percentile / 100.0);
    const float threshold = values[std::min(index, values.size() - 1)];

    // Накопитель голосов: (x, y) -> голосует за центр окружности радиуса r.
    std::vector<int> votes(static_cast<size_t>(rows) * cols * radii, 0);
    for (int y = 1; y < rows - 1; ++y) {
        const float* mag_row = magnitude.ptr<float>(y);
        const float* gx_row = gx.ptr<float>(y);
        const float* gy_row = gy.ptr<float>(y);
        for (int x = 1; x < cols - 1; ++x) {
            if (mag_row[x] < threshold) continue;
            const double angle = std::atan2(gy_row[x], gx_row[x]);
            const double cs = std::cos(angle);
            const double sn = std::sin(angle);
            for (int r = radius_min; r <= radius_max; ++r) {
                const size_t index = static_cast<size_t>(r - radius_min);
                for (int sign = -1; sign <= 1; sign += 2) {
                    const int cx = static_cast<int>(std::lround(x + sign * r * cs));
                    const int cy = static_cast<int>(std::lround(y + sign * r * sn));
                    if (cx < 1 || cy < 1 || cx >= cols - 1 || cy >= rows - 1) continue;
                    // Нижняя часть окружности голосует полностью: спрайт
                    // юнита закрывает основание сверху, а снизу дуга
                    // основания всегда видна целиком. Верхняя часть
                    // голосует слабее — у круга вокруг самого спрайта дуга
                    // рваная, и голосов меньше.
                    const int weight = (y > cy) ? 3 : (lower_half_only ? 0 : 1);
                    if (weight == 0) continue;
                    votes[(static_cast<size_t>(cy) * cols + cx) * radii + index] += weight;
                }
            }
        }
    }

    // Пики накопителя: локальный максимум по соседям.
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            for (int index = 0; index < radii; ++index) {
                const int value =
                    votes[(static_cast<size_t>(y) * cols + x) * radii + index];
                if (value <= 0) continue;
                const double radius = radius_min + index;
                // Доля окружности, на которой нашёлся край: у настоящего
                // основания она заметно больше, чем у мусора. Голоса нижней
                // половины считаются за три (см. веса выше), поэтому делим
                // на три — тогда шкала не зависит от способа голосования.
                const double score = value / (3.0 * 2.0 * M_PI * radius);
                if (score < min_score) continue;
                bool peak = true;
                for (int dy = -2; dy <= 2 && peak; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int ny = y + dy;
                        const int nx = x + dx;
                        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                        const int other =
                            votes[(static_cast<size_t>(ny) * cols + nx) * radii + index];
                        if (other > value) {
                            peak = false;
                            break;
                        }
                    }
                }
                if (!peak) continue;
                RawCircle circle;
                circle.center = cv::Point(x, y);
                circle.radius = static_cast<int>(std::lround(radius));
                circle.quality = score;
                out.push_back(circle);
            }
        }
    }
    return out;
}

// Строки доски, на которых есть поле.
//
// Канонический кадр мобильного клиента состоит из двух половин арены, а
// между ними — пустая полоса (там счётчик волны) и чёрные поля по краям.
// Круг, найденный в пустом месте, — мусор.
std::vector<bool> arena_rows(const cv::Mat& gray) {
    std::vector<bool> rows(gray.rows, false);
    if (gray.empty()) return rows;
    for (int y = 0; y < gray.rows; ++y) {
        int filled = 0;
        const uchar* row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (row[x] > 20) ++filled;
        }
        rows[y] = filled * 5 >= gray.cols * 3;   // не меньше 60 % строки
    }
    return rows;
}

}  // namespace

std::vector<UnitBase> find_unit_bases(const cv::Mat& board, const VisionConfig& config) {
    std::vector<UnitBase> out;
    if (board.empty()) return out;
    cv::Mat gray;
    if (board.channels() == 3) {
        cv::cvtColor(board, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = board;
    }
    cv::medianBlur(gray, gray, 5);

    const int radius_min = std::max(4, static_cast<int>(std::lround(config.base_radius_min)));
    const int radius_max = std::max(radius_min + 1,
                                     static_cast<int>(std::lround(config.base_radius_max)));
    std::vector<RawCircle> circles =
        detect_circles(gray, radius_min, radius_max, config.base_min_distance,
                       config.base_min_score, config.base_gradient_percentile,
                       config.base_lower_half_only);
    if (circles.empty()) return out;

    const std::vector<bool> rows = arena_rows(gray);
    // Лучшие круги первыми: из двух кругов на одном юните остаётся тот,
    // у которого граница круглая (настоящее основание, а не плитка).
    for (RawCircle& circle : circles) {
        const double support = circle_support(gray, circle.center, circle.radius);
        if (support < config.base_min_support) continue;
        circle.quality = circle.quality + support;
    }
    std::stable_sort(circles.begin(), circles.end(), [](const RawCircle& a, const RawCircle& b) {
        if (a.quality != b.quality) return a.quality > b.quality;
        return a.radius > b.radius;
    });
    const double merge_distance = std::max(4.0, config.base_min_distance);
    for (const RawCircle& circle : circles) {
        if (circle.center.y < 0 || circle.center.y >= board.rows) continue;
        if (circle.center.x < 0 || circle.center.x >= board.cols) continue;
        if (!rows[static_cast<size_t>(circle.center.y)]) continue;
        bool duplicate = false;
        for (const UnitBase& kept : out) {
            const double dx = kept.center.x - circle.center.x;
            const double dy = kept.center.y - circle.center.y;
            if (dx * dx + dy * dy < merge_distance * merge_distance) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        UnitBase base;
        base.center = circle.center;
        base.radius = circle.radius;
        base.quality = circle.quality;
        out.push_back(base);
    }
    if (config.base_drop_sprite_circles && out.size() > 1) {
        // Круг вокруг спрайта юнита лежит выше его основания. Если под
        // кругом нашлось другое основание — этот круг не основание, а
        // контур спрайта: убираем.
        std::vector<UnitBase> kept;
        for (const UnitBase& base : out) {
            bool sprite = false;
            for (const UnitBase& other : out) {
                if (&other == &base) continue;
                const double dy = other.center.y - base.center.y;
                const double dx = other.center.x - base.center.x;
                if (dy > 8.0 && dy < 45.0 && std::abs(dx) < 25.0) {
                    sprite = true;
                    break;
                }
            }
            if (!sprite) kept.push_back(base);
        }
        out.swap(kept);
    }
    return out;
}

// ---------------------------------------------------------------- эталоны

bool TemplateBank::load(const std::string& directory,
                        const std::map<std::string, std::string>& mapping,
                        const std::vector<std::string>& names, double scale) {
    templates_.clear();
    order_.clear();
    if (!std::filesystem::exists(directory)) return false;

    std::set<std::string> used;
    const auto add = [&](const std::string& name, const std::string& path) {
        if (templates_.count(name) > 0) return;
        // Проверяем наличие файла сами: иначе OpenCV печатает предупреждение
        // на каждый отсутствующий эталон (в живой игре типа может не быть).
        std::error_code exists_ec;
        if (!std::filesystem::exists(path, exists_ec)) return;
        cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);
        if (color.empty()) return;
        cv::Mat gray;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        if (scale != 1.0 && scale > 0.0) {
            // Масштаб эталона под конкретный клиент: в мобильном клиенте
            // юниты крупнее, чем в веб-клиенте, из которого сняты эталоны.
            const cv::Size target(std::max(1, static_cast<int>(std::lround(gray.cols * scale))),
                                  std::max(1, static_cast<int>(std::lround(gray.rows * scale))));
            cv::Mat resized_gray;
            cv::Mat resized_color;
            cv::resize(gray, resized_gray, target, 0, 0,
                       scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
            cv::resize(color, resized_color, target, 0, 0,
                       scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
            gray = resized_gray;
            color = resized_color;
        }
        Template tpl;
        tpl.name = name;
        tpl.file = path;
        tpl.image = gray;
        tpl.color = color;
        tpl.size = gray.rows;
        // Маска «содержимого» тайла: пиксели, принадлежащие юниту.
        // Чёрные углы тайла — фон, и у соседних юнитов они перекрывают друг
        // друга, из-за чего обычный отклик падает.
        cv::Mat nonzero;
        cv::threshold(gray, nonzero, 0, 255, cv::THRESH_BINARY);
        if (cv::countNonZero(nonzero) == static_cast<int>(nonzero.total())) {
            tpl.content = cv::Mat::ones(nonzero.size(), CV_8U);
        } else {
            tpl.content = nonzero;
        }
        // Где внутри эталона основание юнита: по этой точке эталон
        // совмещается с найденным на поле основанием. Если основание в
        // вырезе найти не удалось, берём середину выреза.
        tpl.base_offset = cv::Point2d(gray.cols / 2.0, gray.rows / 2.0);
        {
            const int radius_min = std::max(4, static_cast<int>(std::lround(gray.rows * 0.28)));
            const int radius_max =
                std::max(radius_min + 1, static_cast<int>(std::lround(gray.rows * 0.48)));
            const std::vector<RawCircle> circles =
                detect_circles(gray, radius_min, radius_max, 8.0, 0.03, 40.0, false);
            const RawCircle* best = nullptr;
            for (const RawCircle& circle : circles) {
                if (best == nullptr || circle.quality > best->quality) best = &circle;
            }
            if (best != nullptr) {
                tpl.base_offset = cv::Point2d(best->center.x, best->center.y);
            }
        }
        templates_[name] = tpl;
        order_.push_back(name);
    };

    // Сначала описанные в конфиге типы: для них имя файла может быть
    // переопределено явно (в живой игре одноимённого файла нет).
    for (const std::string& name : names) {
        const auto it = mapping.find(name);
        const std::string file = it != mapping.end() ? it->second : name + ".png";
        used.insert(file);
        add(name, directory + "/" + file);
    }
    // Остальные файлы каталога — дополнительные эталоны: в живой игре типов
    // юнитов больше, чем описано в конфиге, и лишние эталоны помогают
    // находить юниты вообще.
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (entry.path().extension() != ".png") continue;
        const std::string file = entry.path().filename().string();
        if (used.count(file) > 0) continue;
        add(entry.path().stem().string(), entry.path().string());
    }
    return !templates_.empty();
}

const Template* TemplateBank::get(const std::string& name) const {
    const auto it = templates_.find(name);
    return it == templates_.end() ? nullptr : &it->second;
}

std::vector<std::string> TemplateBank::names() const { return order_; }

// ------------------------------------------------------------- подавление

std::vector<Detection> non_max_suppression(std::vector<Detection> detections, double iou_threshold) {
    std::vector<Detection> kept;
    if (detections.empty()) return kept;
    std::stable_sort(detections.begin(), detections.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });
    for (const Detection& det : detections) {
        const Box box = box_of(det);
        bool overlaps = false;
        for (const Detection& other : kept) {
            if (iou(box, box_of(other)) >= iou_threshold) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) kept.push_back(det);
    }
    return kept;
}

// ---------------------------------------------------------------- детектор

VisionEngine::VisionEngine(VisionConfig config) : config_(std::move(config)) {}

std::vector<std::string> VisionEngine::template_names() const {
    // Какие эталоны искать: описанные в конфиге типы или все файлы каталога.
    std::vector<std::string> names;
    if (config_.detect_all_templates) {
        names = bank_.names();
    } else {
        for (const UnitSpec& spec : config_.units) names.push_back(spec.name);
    }
    return names;
}

bool VisionEngine::load_templates() {
    std::vector<std::string> names;
    for (const UnitSpec& spec : config_.units) names.push_back(spec.name);
    loaded_ = bank_.load(config_.templates_dir, config_.template_map, names, config_.template_scale);
    return loaded_;
}

bool VisionEngine::load_deck_map(const std::string& path) {
    deck_positions_.clear();
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first) {
            first = false;
            // Заголовок: «file deck_card card_look ...».
            if (line.rfind("file", 0) == 0) continue;
        }
        // Столбцы разделены табами: имя файла, номер карточки в колоде, ...
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string file_name = line.substr(0, tab);
        const std::string rest = line.substr(tab + 1);
        const std::size_t tab2 = rest.find('\t');
        const std::string card = tab2 == std::string::npos ? rest : rest.substr(0, tab2);
        // Номер карточки: «2 (своя колода)» -> 2.
        std::size_t digit = 0;
        while (digit < card.size() && (std::isdigit(static_cast<unsigned char>(card[digit])) == 0)) ++digit;
        if (digit >= card.size()) continue;
        const int position = std::atoi(card.c_str() + digit);
        if (position <= 0) continue;
        // Имя файла без расширения — так называются эталоны.
        const std::string stem = std::filesystem::path(file_name).stem().string();
        deck_positions_[stem] = position;
    }
    return !deck_positions_.empty();
}

const Template* VisionEngine::template_by_name(const std::string& name) const {
    return bank_.get(name);
}

int VisionEngine::deck_position(const std::string& template_name) const {
    // Название типа может быть именем файла (для «прочих» эталонов) или
    // именем типа из конфига — тогда берём файл из банка эталонов.
    std::string file = template_name;
    const Template* tpl = bank_.get(template_name);
    if (tpl != nullptr && !tpl->file.empty()) file = tpl->file;
    // В таблице карточек ключ — имя файла без расширения и без пути.
    file = std::filesystem::path(file).stem().string();
    const auto it = deck_positions_.find(file);
    return it == deck_positions_.end() ? 0 : it->second;
}

std::vector<Detection> VisionEngine::match(const cv::Mat& board, const std::string& name) const {
    return match_map(board, name, nullptr);
}

std::vector<Detection> VisionEngine::match_map(const cv::Mat& board, const std::string& name,
                                               cv::Mat* response) const {
    std::vector<Detection> detections;
    const Template* tpl = bank_.get(name);
    if (tpl == nullptr || board.empty()) return detections;

    cv::Mat haystack;
    cv::Mat needle;
    if (config_.grayscale) {
        if (board.channels() == 3) {
            cv::cvtColor(board, haystack, cv::COLOR_BGR2GRAY);
        } else {
            haystack = board;
        }
        needle = tpl->image;
    } else {
        if (board.channels() == 3) {
            haystack = board;
        } else {
            cv::cvtColor(board, haystack, cv::COLOR_GRAY2BGR);
        }
        needle = tpl->color;
    }
    if (haystack.rows < needle.rows || haystack.cols < needle.cols) return detections;

    cv::Mat result;
    cv::matchTemplate(haystack, needle, result, cv::TM_CCOEFF_NORMED);
    if (response != nullptr) *response = result;
    const int height = needle.rows;
    const int width = needle.cols;
    for (int y = 0; y < result.rows; ++y) {
        const float* row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x) {
            const double score = row[x];
            if (score < config_.match_threshold) continue;
            Detection det;
            det.name = name;
            det.deck_position = deck_position(name);
            det.score = score;
            det.box = cv::Rect(x, y, width, height);
            det.center = cv::Point2d(x + width / 2.0, y + height / 2.0);
            detections.push_back(det);
        }
    }
    detections = non_max_suppression(std::move(detections), config_.nms_iou);
    if (static_cast<int>(detections.size()) > config_.max_per_template) {
        detections.resize(config_.max_per_template);
    }
    return detections;
}

void VisionEngine::sample_color(const cv::Mat& hsv, const Detection& det, double* hue, int* saturation,
                                 int* value) {
    const cv::Rect box = det.box & cv::Rect(0, 0, hsv.cols, hsv.rows);
    *hue = 0.0;
    *saturation = 0;
    *value = 0;
    if (box.width <= 0 || box.height <= 0) return;

    // Глиф типа внутри юнита белый (насыщенность ~0), а тело — насыщенное.
    // Поэтому берём кольцо вокруг центра и отбрасываем ненасыщенные пиксели.
    const double cx = box.width / 2.0;
    const double cy = box.height / 2.0;
    const double half = std::min(box.width, box.height) / 2.0;
    std::vector<double> hues;
    std::vector<double> sats;
    std::vector<double> vals;
    for (int y = 0; y < box.height; ++y) {
        const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(box.y + y);
        for (int x = 0; x < box.width; ++x) {
            const double dx = x + 0.5 - cx;
            const double dy = y + 0.5 - cy;
            const double dist = std::hypot(dx, dy);
            if (dist < half * 0.35 || dist > half * 0.9) continue;
            const cv::Vec3b& pixel = row[box.x + x];
            hues.push_back(pixel[0]);
            sats.push_back(pixel[1]);
            vals.push_back(pixel[2]);
        }
    }
    if (hues.empty()) {
        // Кольцо пустое (слишком маленькая находка) — берём весь бокс.
        for (int y = 0; y < box.height; ++y) {
            const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(box.y + y);
            for (int x = 0; x < box.width; ++x) {
                const cv::Vec3b& pixel = row[box.x + x];
                hues.push_back(pixel[0]);
                sats.push_back(pixel[1]);
                vals.push_back(pixel[2]);
            }
        }
    }
    if (hues.empty()) return;

    // Насыщенные пиксели — это тело, белый глиф в расчёт не попадает.
    std::vector<double> kept_hues;
    std::vector<double> kept_sats;
    for (size_t i = 0; i < hues.size(); ++i) {
        if (sats[i] >= 80) {
            kept_hues.push_back(hues[i]);
            kept_sats.push_back(sats[i]);
        }
    }
    if (kept_hues.size() >= 4) {
        hues = kept_hues;
        sats = kept_sats;
    }

    // Оттенок круговой: усредняем по вектору, чтобы не ломать переход через 0.
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    for (double hue_value : hues) {
        const double rad = hue_value * 2.0 * M_PI / 180.0;
        sum_cos += std::cos(rad);
        sum_sin += std::sin(rad);
    }
    const double n = static_cast<double>(hues.size());
    double mean_hue = std::atan2(sum_sin / n, sum_cos / n) * 180.0 / M_PI / 2.0;
    if (mean_hue < 0.0) mean_hue += 180.0;
    *hue = mean_hue;
    *saturation = static_cast<int>(std::lround(median_of(sats)));
    *value = static_cast<int>(std::lround(median_of(vals)));
}

Team VisionEngine::classify_team(double hue, int saturation, int value) const {
    if (saturation < config_.faction.min_saturation || value < config_.faction.min_value) {
        return Team::Unknown;
    }
    if (hue_in_ranges(hue, config_.faction.ally_hue_ranges)) return Team::Ally;
    if (hue_in_ranges(hue, config_.faction.enemy_hue_ranges)) return Team::Enemy;
    return Team::Unknown;
}

// Разделение слипшихся спрайтов.
//
// Обычное сопоставление по шаблону даёт один максимум отклика на
// слипшуюся пару: тайлы юнитов накладываются, и шаблон перестаёт их
// различать. Видимый юнит при этом находится точно, а вот спрятанный под
// ним теряется.
//
// Спрятанный юнит виден по «второму максимуму» отклика рядом с найденной
// находкой. Его отклик сам по себе невелик (шаблон наполовину перекрыт),
// поэтому он оценивается не по всему прямоугольнику, а по видимой части:
// из шаблона выбрасываются пиксели, закрытые другими юнитами.
//
// Параметры заданы здесь, а не в конфиге, чтобы Python-прототип и C++-порт
// считали одинаково (в C++ конфиг yaml не разбирается).
namespace {
constexpr double kSplitMinSeparation = 11.0;
constexpr double kSplitPeakFloor = 0.40;
constexpr int kSplitMinVisible = 150;
constexpr double kSplitRadius = 48.0;
constexpr int kSplitRingCandidates = 4;
constexpr double kSplitMinScore = 0.75;
constexpr int kSplitExcludeMargin = 2;

// Локальные максимумы карты отклика (окно 3x3), отсортированные по убыванию.
std::vector<cv::Point3f> local_maxima(const cv::Mat& res, double floor) {
    std::vector<cv::Point3f> out;
    if (res.empty()) return out;
    cv::Mat dilated;
    cv::dilate(res, dilated, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    std::vector<cv::Point> points;
    std::vector<double> scores;
    for (int y = 0; y < res.rows; ++y) {
        const float* row = res.ptr<float>(y);
        const float* drow = dilated.ptr<float>(y);
        for (int x = 0; x < res.cols; ++x) {
            if (row[x] < floor) continue;
            if (row[x] < drow[x] - 1e-6) continue;
            points.emplace_back(x, y);
            scores.push_back(row[x]);
        }
    }
    std::vector<int> order(scores.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    for (int index : order) {
        out.emplace_back(static_cast<float>(points[index].x), static_cast<float>(points[index].y),
                         static_cast<float>(scores[index]));
    }
    return out;
}

// Корреляция шаблона с видимой частью картинки.
//
// Из шаблона выбрасываются: чёрные углы тайла (это фон, а не юнит) и те
// пиксели, которые закрыты другими юнитами (occupied — зоны перекрытия в
// координатах поля).
double visible_score(const cv::Mat& gray, const Template& tpl, int x, int y,
                     const cv::Mat& occupied) {
    const int size = tpl.size;
    if (x < 0 || y < 0 || x + size > gray.cols || y + size > gray.rows) return 0.0;
    const cv::Mat patch = gray(cv::Rect(x, y, size, size));
    cv::Mat mask;
    if (occupied.empty()) {
        mask = tpl.content.clone();
    } else {
        cv::bitwise_and(tpl.content, 255 - occupied(cv::Rect(x, y, size, size)), mask);
    }
    const int visible = cv::countNonZero(mask);
    if (visible < kSplitMinVisible) return 0.0;
    std::vector<uchar> values;
    std::vector<double> a;
    std::vector<double> b;
    values.reserve(visible);
    for (int row = 0; row < size; ++row) {
        const uchar* mask_row = mask.ptr<uchar>(row);
        const uchar* patch_row = patch.ptr<uchar>(row);
        for (int col = 0; col < size; ++col) {
            if (!mask_row[col]) continue;
            a.push_back(patch_row[col]);
            b.push_back(tpl.image.at<uchar>(row, col));
        }
    }
    if (a.size() < 2) return 0.0;
    double mean_a = 0.0;
    double mean_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<double>(a.size());
    mean_b /= static_cast<double>(b.size());
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = a[i] - mean_a;
        const double db = b[i] - mean_b;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    const double den = std::sqrt(norm_a * norm_b);
    if (den <= 0.0) return 0.0;
    return dot / den;
}

// Лучшие точки отклика в кольце вокруг уже найденных находок.
std::vector<cv::Point2f> ring_candidates(const cv::Mat& res, const std::vector<Detection>& dets,
                                          int size) {
    std::vector<cv::Point2f> out;
    const double half = size / 2.0;
    for (const Detection& det : dets) {
        int x0 = static_cast<int>(det.center.x - half - kSplitRadius);
        int x1 = static_cast<int>(det.center.x - half + kSplitRadius) + 1;
        int y0 = static_cast<int>(det.center.y - half - kSplitRadius);
        int y1 = static_cast<int>(det.center.y - half + kSplitRadius) + 1;
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(res.cols, x1);
        y1 = std::min(res.rows, y1);
        if (x1 - x0 < 3 || y1 - y0 < 3) continue;
        const cv::Mat win = res(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        const double ccx = det.center.x - half - x0;
        const double ccy = det.center.y - half - y0;
        // Точки кольца: внутри самого бокса находки искать нечего.
        std::vector<cv::Point> candidates;
        std::vector<double> values;
        const double min_dist = kSplitMinSeparation + 2.0;
        for (int y = 0; y < win.rows; ++y) {
            const float* row = win.ptr<float>(y);
            for (int x = 0; x < win.cols; ++x) {
                if (row[x] < kSplitPeakFloor) continue;
                const double dx = x + 0.5 - ccx;
                const double dy = y + 0.5 - ccy;
                if (dx * dx + dy * dy <= min_dist * min_dist) continue;
                candidates.emplace_back(x0 + x, y0 + y);
                values.push_back(row[x]);
            }
        }
        std::vector<int> order(values.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return values[a] > values[b]; });
        std::vector<cv::Point2f> picked;
        for (int index : order) {
            const cv::Point& point = candidates[index];
            bool too_close = false;
            for (const cv::Point2f& other : picked) {
                if (std::abs(point.x - other.x) <= 6 && std::abs(point.y - other.y) <= 6) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) continue;
            picked.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
            if (static_cast<int>(picked.size()) >= kSplitRingCandidates) break;
        }
        for (const cv::Point2f& point : picked) out.push_back(point);
    }
    return out;
}
}  // namespace

std::vector<Detection> VisionEngine::split_merged(
    const cv::Mat& board, const std::vector<Detection>& detections,
    const std::map<std::string, cv::Mat>& responses) const {
    std::vector<Detection> extra;
    if (detections.empty() || responses.empty()) return extra;

    cv::Mat gray;
    if (config_.grayscale) {
        if (board.channels() == 3) {
            cv::cvtColor(board, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = board;
        }
    } else {
        gray = board;
    }

    // Зоны перекрытия и «зона поиска» рисуем один раз в координатах поля.
    cv::Mat occupied = cv::Mat::zeros(gray.size(), CV_8U);
    cv::Mat near = cv::Mat::zeros(gray.size(), CV_8U);
    const auto mark = [](cv::Mat& mask, int x0, int y0, int x1, int y1) {
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(mask.cols, x1);
        y1 = std::min(mask.rows, y1);
        if (x1 > x0 && y1 > y0) mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)).setTo(cv::Scalar(255));
    };
    for (const Detection& det : detections) {
        const cv::Rect& box = det.box;
        mark(occupied, box.x - kSplitExcludeMargin, box.y - kSplitExcludeMargin,
             box.x + box.width + kSplitExcludeMargin, box.y + box.height + kSplitExcludeMargin);
        mark(near, box.x - static_cast<int>(kSplitRadius), box.y - static_cast<int>(kSplitRadius),
             box.x + box.width + static_cast<int>(kSplitRadius),
             box.y + box.height + static_cast<int>(kSplitRadius));
    }

    struct Candidate {
        double score;
        std::string name;
        int x;
        int y;
        int size;
    };
    std::vector<Candidate> candidates;
    for (const auto& [name, res] : responses) {
        const Template* tpl = bank_.get(name);
        if (tpl == nullptr) continue;
        std::vector<cv::Point2f> spots;
        for (const cv::Point3f& peak : local_maxima(res, kSplitPeakFloor)) {
            // Спрятанный юнит может прятаться только рядом с уже найденным:
            // вдали от находок смотреть нечего.
            if (!near.at<uchar>(static_cast<int>(peak.y), static_cast<int>(peak.x))) continue;
            spots.emplace_back(peak.x, peak.y);
        }
        for (const cv::Point2f& point : ring_candidates(res, detections, tpl->size)) {
            spots.push_back(point);
        }
        for (const cv::Point2f& spot : spots) {
            const int x = static_cast<int>(spot.x);
            const int y = static_cast<int>(spot.y);
            const double score = visible_score(gray, *tpl, x, y, occupied);
            if (score >= kSplitMinScore) candidates.push_back({score, name, x, y, tpl->size});
        }
    }
    // Лучшие кандидаты побеждают: сначала по оценке, потом жадно, чтобы
    // один юнит не дал несколько находок.
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<cv::Point2d> taken;
    for (const Detection& det : detections) taken.push_back(det.center);
    for (const Candidate& candidate : candidates) {
        const cv::Point2d center(candidate.x + candidate.size / 2.0, candidate.y + candidate.size / 2.0);
        bool close = false;
        for (const cv::Point2d& other : taken) {
            if (cv::norm(center - other) < kSplitMinSeparation) {
                close = true;
                break;
            }
        }
        if (close) continue;
        Detection det;
        det.name = candidate.name;
        det.score = candidate.score;
        det.box = cv::Rect(candidate.x, candidate.y, candidate.size, candidate.size);
        det.center = center;
        extra.push_back(det);
        taken.push_back(center);
    }
    return extra;
}

namespace {

// Корреляция эталона с картинкой по видимой части.
//
// Пиксели эталона, которые закрыты (краем доски или другим юнитом), в
// сравнении не участвуют: иначе совпадение падает из-за того, что видно
// меньше половины юнита. Возвращает коэффициент корреляции (-1..1) или
// -1, если сравнивать нечего.
double masked_ncc(const cv::Mat& gray, const Template& tpl, int x, int y) {
    const int size = tpl.size;
    const int x0 = std::max(0, x);
    const int y0 = std::max(0, y);
    const int x1 = std::min(gray.cols, x + size);
    const int y1 = std::min(gray.rows, y + size);
    if (x1 <= x0 || y1 <= y0) return -1.0;
    int count = 0;
    double sum_a = 0.0;
    double sum_b = 0.0;
    for (int row = y0; row < y1; ++row) {
        const uchar* mask_row = tpl.content.ptr<uchar>(row - y);
        const uchar* tpl_row = tpl.image.ptr<uchar>(row - y);
        const uchar* gray_row = gray.ptr<uchar>(row);
        for (int col = x0; col < x1; ++col) {
            if (!mask_row[col - x]) continue;
            const double a = tpl_row[col - x];
            const double b = gray_row[col];
            sum_a += a;
            sum_b += b;
            ++count;
        }
    }
    if (count < 64) return -1.0;
    const double mean_a = sum_a / count;
    const double mean_b = sum_b / count;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int row = y0; row < y1; ++row) {
        const uchar* mask_row = tpl.content.ptr<uchar>(row - y);
        const uchar* tpl_row = tpl.image.ptr<uchar>(row - y);
        const uchar* gray_row = gray.ptr<uchar>(row);
        for (int col = x0; col < x1; ++col) {
            if (!mask_row[col - x]) continue;
            const double da = tpl_row[col - x] - mean_a;
            const double db = gray_row[col] - mean_b;
            dot += da * db;
            norm_a += da * da;
            norm_b += db * db;
        }
    }
    const double den = std::sqrt(norm_a * norm_b);
    if (den <= 0.0) return -1.0;
    return dot / den;
}

}  // namespace

std::vector<Detection> VisionEngine::detect_by_bases(const cv::Mat& board) const {
    return detect_by_bases(board, find_unit_bases(board, config_));
}

std::vector<Detection> VisionEngine::detect_by_bases(const cv::Mat& board,
                                                     const std::vector<UnitBase>& bases) const {
    std::vector<Detection> out;
    if (bases.empty()) return out;

    cv::Mat gray;
    if (board.channels() == 3) {
        cv::cvtColor(board, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = board;
    }

    const std::vector<std::string> names = template_names();
    for (const UnitBase& base : bases) {
        // Первая стадия: совпадение ровно по центру основания.
        struct Candidate {
            std::string name;
            double score;
            int x;
            int y;
        };
        std::vector<Candidate> candidates;
        for (const std::string& name : names) {
            const Template* tpl = bank_.get(name);
            if (tpl == nullptr) continue;
            const int x = static_cast<int>(std::lround(base.center.x - tpl->base_offset.x));
            const int y = static_cast<int>(std::lround(base.center.y - tpl->base_offset.y));
            const double score = masked_ncc(gray, *tpl, x, y);
            candidates.push_back({name, score, x, y});
        }
        if (candidates.empty()) continue;
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        // Вторая стадия: небольшой поиск положения для лучших кандидатов —
        // центр основания определяется с точностью до пары пикселей, и
        // эталон встаёт на место не идеально.
        const size_t refine_count = std::min<size_t>(candidates.size(), 3);
        const int radius = std::max(0, config_.classify_radius);
        for (size_t i = 0; i < refine_count && radius > 0; ++i) {
            Candidate& candidate = candidates[i];
            const Template* tpl = bank_.get(candidate.name);
            if (tpl == nullptr) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const double score = masked_ncc(gray, *tpl, candidate.x + dx, candidate.y + dy);
                    if (score > candidate.score) {
                        candidate.score = score;
                        candidate.x += dx;
                        candidate.y += dy;
                    }
                }
            }
        }

        const Candidate& best = candidates.front();
        const Template* tpl = bank_.get(best.name);
        Detection det;
        det.name = best.name;
        det.deck_position = deck_position(best.name);
        det.score = best.score;
        // Положение юнита — центр его основания: спрайт смещён вверх, а
        // основание стоит ровно там, где юнит стоит на земле.
        det.center = cv::Point2d(base.center.x, base.center.y);
        if (tpl != nullptr) {
            const cv::Rect raw(best.x, best.y, tpl->size, tpl->size);
            det.box = raw & cv::Rect(0, 0, board.cols, board.rows);
        }
        out.push_back(det);
    }
    return out;
}

std::vector<Detection> VisionEngine::detect_by_template(const cv::Mat& board) const {
    std::vector<Detection> all;
    if (board.empty()) return all;

    // Какие эталоны искать: описанные в конфиге типы или все файлы каталога.
    const std::vector<std::string> names = template_names();
    std::map<std::string, cv::Mat> responses;
    for (const std::string& name : names) {
        cv::Mat response;
        const std::vector<Detection> found = match_map(board, name, &response);
        all.insert(all.end(), found.begin(), found.end());
        if (!response.empty()) responses[name] = response;
    }
    // Общий NMS: один юнит не должен давать находку сразу от нескольких
    // шаблонов.
    all = non_max_suppression(std::move(all), config_.nms_iou);
    // Юниты, спрятанные под другими: слипшиеся спрайты дают одну находку.
    {
        const std::vector<Detection> extra = split_merged(board, all, responses);
        all.insert(all.end(), extra.begin(), extra.end());
    }
    return all;
}

std::vector<Detection> VisionEngine::detect(const cv::Mat& board) const {
    std::vector<Detection> all;
    if (board.empty()) return all;

    if (config_.detect_by_base) {
        // Мобильный клиент: юнит стоит на круглом основании, и по нему
        // находится и его место, и его тип.
        all = detect_by_bases(board);
        if (!all.empty() && config_.loose_template_threshold < 1.0) {
            // Часть юнитов стоит не на круге (основание квадратное или
            // спрятано): их находят только по эталону.
            const std::vector<Detection> by_template = detect_by_template(board);
            for (const Detection& det : by_template) {
                if (det.score < config_.loose_template_threshold) continue;
                bool near_base = false;
                for (const Detection& found : all) {
                    const double dx = found.center.x - det.center.x;
                    const double dy = found.center.y - det.center.y;
                    if (dx * dx + dy * dy < config_.base_match_radius * config_.base_match_radius) {
                        near_base = true;
                        break;
                    }
                }
                if (near_base) continue;
                all.push_back(det);
            }
        }
    }
    if (all.empty()) {
        // Веб-клиент и симулятор: оснований там нет, юниты ищутся по
        // эталонам.
        all = detect_by_template(board);
    }
    assign_teams(board, all);
    return all;
}

void VisionEngine::assign_teams(const cv::Mat& board, std::vector<Detection>& detections) const {
    if (detections.empty()) return;
    cv::Mat hsv;
    if (board.channels() == 3) {
        cv::cvtColor(board, hsv, cv::COLOR_BGR2HSV);
    } else {
        cv::cvtColor(board, hsv, cv::COLOR_GRAY2BGR);
        cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
    }
    for (Detection& det : detections) {
        int saturation = 0;
        int value = 0;
        double hue = 0.0;
        sample_color(hsv, det, &hue, &saturation, &value);
        det.hue = hue;
        det.saturation = saturation;
        if (config_.team_by_position) {
            const double split = board.rows * config_.team_split;
            det.team = det.center.y < split ? Team::Enemy : Team::Ally;
        } else {
            det.team = classify_team(hue, saturation, value);
        }
    }
}

GameState VisionEngine::analyze(const cv::Mat& canvas, const ArenaLayout& arena, double mana,
                                double mana_confidence,
                                const std::vector<std::string>& available) const {
    const cv::Mat board = extract_field(canvas, arena);
    if (board.empty()) {
        GameState empty;
        empty.mana = mana;
        empty.mana_confidence = mana_confidence;
        empty.available_units = available;
        return empty;
    }
    return build_state(detect(board), mana, mana_confidence, available);
}

GameState VisionEngine::analyze_board(const cv::Mat& board, double mana, double mana_confidence,
                                       const std::vector<std::string>& available) const {
    if (board.empty()) {
        GameState empty;
        empty.mana = mana;
        empty.mana_confidence = mana_confidence;
        empty.available_units = available;
        return empty;
    }
    // Мобильный клиент: канонический кадр собран из половин арены, и у
    // каждого юнита есть круглое основание. Если оснований на доске много,
    // работаем по ним: положение юнита тогда известно точно.
    std::vector<Detection> detections;
    if (config_.detect_by_base || config_.auto_detect_by_base) {
        const std::vector<UnitBase> bases = find_unit_bases(board, config_);
        if (config_.detect_by_base ||
            static_cast<int>(bases.size()) >= config_.base_min_count) {
            detections = detect_by_bases(board, bases);
            assign_teams(board, detections);
        }
    }
    if (detections.empty()) detections = detect(board);
    return build_state(detections, mana, mana_confidence, available);
}

GameState VisionEngine::build_state(const std::vector<Detection>& detections, double mana,
                                    double mana_confidence,
                                    const std::vector<std::string>& available) const {
    GameState state;
    state.mana = mana;
    state.mana_confidence = mana_confidence;
    state.detections = detections;
    state.available_units = available;
    int uid = 1;
    for (const Detection& det : detections) {
        const UnitSpec* spec = nullptr;
        for (const UnitSpec& candidate : config_.units) {
            if (candidate.name == det.name) {
                spec = &candidate;
                break;
            }
        }
        if (spec == nullptr) continue;
        Unit unit;
        unit.uid = uid++;
        unit.kind = det.name;
        unit.team = det.team == Team::Unknown ? Team::Enemy : det.team;
        unit.x = det.center.x;
        unit.y = det.center.y;
        unit.hp = spec->hp;
        unit.max_hp = spec->hp;
        unit.spec = spec;
        if (unit.team == Team::Ally) {
            state.ally_units.push_back(unit);
        } else {
            state.enemy_units.push_back(unit);
        }
    }
    return state;
}

// ----------------------------------------------------------------- геометрия

ArenaLayout web_arena() {
    // Веб-клиент: холст 492x804, арена — прямоугольник (66,52)-(400,664).
    // Размер арены совпадает с каноническим полем 334x612.
    ArenaLayout arena;
    arena.quad = {cv::Point2f(66.0f, 52.0f), cv::Point2f(400.0f, 52.0f),
                  cv::Point2f(400.0f, 664.0f), cv::Point2f(66.0f, 664.0f)};
    arena.board_size = cv::Size(334, 612);
    // Половина арены: в веб-клиенте сверху противник, снизу свои.
    arena.split = (348.0 - 52.0) / 612.0;
    return arena;
}

ArenaLayout emulator_arena() {
    // Мобильный клиент: холст 950x1034. В бою видно две половины арены —
    // сверху чужая, снизу своя; берём нижнюю.
    //
    // Камера в бою ходит (арена то приближается, то отдаляется), поэтому
    // прямоугольник задан приблизительно: для точного разбора нужен поиск
    // арены по кадру.
    ArenaLayout arena;
    arena.quad = {cv::Point2f(238.0f, 510.0f), cv::Point2f(697.0f, 510.0f),
                  cv::Point2f(697.0f, 858.0f), cv::Point2f(238.0f, 858.0f)};
    arena.board_size = cv::Size(459, 348);
    // В нижней половине арены находятся только свои юниты.
    arena.split = 0.0;
    return arena;
}

cv::Mat extract_field(const cv::Mat& canvas, const ArenaLayout& arena) {
    if (canvas.empty() || arena.quad.size() != 4) return {};
    cv::Point2f src[4];
    for (int i = 0; i < 4; ++i) src[i] = arena.quad[static_cast<size_t>(i)];
    const cv::Point2f dst[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(arena.board_size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(arena.board_size.width - 1),
                    static_cast<float>(arena.board_size.height - 1)),
        cv::Point2f(0.0f, static_cast<float>(arena.board_size.height - 1)),
    };
    const cv::Mat matrix = cv::getPerspectiveTransform(src, dst);
    cv::Mat field;
    cv::warpPerspective(canvas, field, matrix, arena.board_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return field;
}

void add_deploy_lanes(GameState* state) {
    if (state == nullptr) return;
    // Доли взяты из симулятора: свои юниты появляются внизу поля, уходят к
    // вражеской базе через мост своей линии.
    Lane left;
    left.name = "left";
    left.x = 83.0;
    left.y = 526.0;
    Lane right;
    right.name = "right";
    right.x = 250.0;
    right.y = 526.0;
    state->spawn_points["left"] = left;
    state->spawn_points["right"] = right;
}

// ------------------------------------------------- характеристики типов юнитов
std::vector<UnitSpec> default_live_specs() {
    // Значения из config/live.yaml.
    UnitSpec knight;
    knight.name = "knight";
    knight.cost = 100;
    knight.hp = 220;
    knight.damage = 28;
    knight.attack_interval = 0.9;
    knight.speed = 26;
    knight.attack_range = 18;
    knight.radius = 9;
    knight.family = "melee";
    knight.counters = {"warden"};

    UnitSpec archer;
    archer.name = "archer";
    archer.cost = 100;
    archer.hp = 130;
    archer.damage = 34;
    archer.attack_interval = 1.2;
    archer.speed = 22;
    archer.attack_range = 42;
    archer.radius = 8;
    archer.family = "ranged";
    archer.counters = {"samurai"};

    UnitSpec golem;
    golem.name = "golem";
    golem.cost = 100;
    golem.hp = 520;
    golem.damage = 22;
    golem.attack_interval = 1.4;
    golem.speed = 14;
    golem.attack_range = 16;
    golem.radius = 11;
    golem.family = "tank";
    golem.counters = {"knight"};

    UnitSpec samurai;
    samurai.name = "samurai";
    samurai.cost = 100;
    samurai.hp = 260;
    samurai.damage = 46;
    samurai.attack_interval = 1.1;
    samurai.speed = 30;
    samurai.attack_range = 20;
    samurai.radius = 9;
    samurai.family = "melee";
    samurai.counters = {"golem"};

    UnitSpec warden;
    warden.name = "warden";
    warden.cost = 100;
    warden.hp = 600;
    warden.damage = 30;
    warden.attack_interval = 1.3;
    warden.speed = 12;
    warden.attack_range = 24;
    warden.radius = 12;
    warden.family = "tank";
    warden.counters = {"bomber"};

    return {knight, archer, golem, samurai, warden};
}

std::map<std::string, std::string> default_live_template_map() {
    // Значения из config/live.yaml (vision.template_map).
    return {
        {"knight", "unit_00.png"},
        {"archer", "unit_01.png"},
        {"golem", "unit_02.png"},
        {"samurai", "unit_03.png"},
        {"warden", "unit_04.png"},
    };
}

std::map<std::string, std::string> default_emulator_template_map() {
    // Эталоны мобильного клиента (assets/real_emulator): спрайты сняты с
    // настоящих боевых кадров и названы по цвету тела и основания.
    //
    // Соответствие «тип из конфига → спрайт» приблизительное: конфиг
    // описывает веб-клиент, а в мобильном спрайты другие. Подробности —
    // в assets/real_emulator/types.tsv (там же видно, какой спрайт какому
    // юниту соответствует).
    return {
        {"knight", "unit_em_00_teal_on_gray.png"},
        {"archer", "unit_em_01_orange_on_green.png"},
        {"golem", "unit_em_02_orange_on_orange.png"},
        {"samurai", "unit_em_03_red_on_green.png"},
        {"warden", "unit_em_04_orange_on_lime.png"},
    };
}

}  // namespace rr
