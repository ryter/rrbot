#include "merge.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace rr {
namespace {

// Число ячеек гистограммы: оттенок, насыщенность, яркость.
inline const int kHistHue = 18;
inline const int kHistSat = 4;
inline const int kHistVal = 4;
inline const int kHistSize = kHistHue * kHistSat * kHistVal;

// Портрет пешки: квадрат со стороной 2R вокруг основания.
//
// Если пешка стоит у края кадра, портрет дополняем нулями — иначе сравнение
// с соседней пешкой ломается.
cv::Mat portrait(const cv::Mat& canvas, const Pawn& pawn, int radius) {
    const int side = radius * 2;
    const int x0 = pawn.center.x - radius;
    const int y0 = pawn.center.y - radius;
    const int sx = std::max(0, -x0);
    const int sy = std::max(0, -y0);
    const int width = std::min(side - sx, canvas.cols - std::max(0, x0));
    const int height = std::min(side - sy, canvas.rows - std::max(0, y0));
    if (width <= 0 || height <= 0) return {};
    cv::Mat patch = canvas(cv::Rect(std::max(0, x0), std::max(0, y0), width, height)).clone();
    if (patch.rows == side && patch.cols == side) return patch;
    cv::Mat padded(side, side, CV_8UC3, cv::Scalar(0, 0, 0));
    patch.copyTo(padded(cv::Rect(sx, sy, width, height)));
    return padded;
}

// Палитра портрета: гистограмма цветов по кругу основания.
//
// Именно палитра, а не пиксели: у юнитов есть анимация (шестерёнки крутятся,
// пушка поворачивается), и по пикселям два одинаковых юнита дают 0.2-0.5,
// а два разных — 0.3. По палитре одинаковые дают 0.99, разные — 0.2-0.3.
cv::Mat color_histogram(const cv::Mat& canvas, const Pawn& pawn, int radius) {
    cv::Mat hist = cv::Mat::zeros(kHistSize, 1, CV_32F);
    const cv::Mat patch = portrait(canvas, pawn, radius);
    if (patch.empty()) return hist;
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    const int center = radius;
    const int inner = std::max(4, radius - 3);
    for (int y = 0; y < hsv.rows; ++y) {
        const int dy = y - center;
        for (int x = 0; x < hsv.cols; ++x) {
            const int dx = x - center;
            if (dx * dx + dy * dy > inner * inner) continue;
            const cv::Vec3b& pixel = hsv.at<cv::Vec3b>(y, x);
            const int h = std::min(kHistHue - 1, pixel[0] * kHistHue / 180);
            const int s = std::min(kHistSat - 1, pixel[1] * kHistSat / 256);
            const int v = std::min(kHistVal - 1, pixel[2] * kHistVal / 256);
            hist.at<float>(h * kHistSat * kHistVal + s * kHistVal + v) += 1.0f;
        }
    }
    const float total = cv::sum(hist)[0];
    if (total > 0.0f) hist /= total;
    return hist;
}

// Похожество двух палитр: косинус между гистограммами.
double similarity(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return -1.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (int i = 0; i < a.rows; ++i) {
        const float va = a.at<float>(i);
        const float vb = b.at<float>(i);
        dot += static_cast<double>(va) * vb;
        norm_a += static_cast<double>(va) * va;
        norm_b += static_cast<double>(vb) * vb;
    }
    if (norm_a < 1e-9 || norm_b < 1e-9) return -1.0;
    return dot / std::sqrt(norm_a * norm_b);
}

}  // namespace

// Заполненность основания: см. merge.hpp.
double pedestal_fill(const cv::Mat& canvas, const Pawn& pawn) {
    const int radius = 38;
    const int x0 = pawn.center.x - radius;
    const int y0 = pawn.center.y - radius;
    if (x0 < 0 || y0 < 0 || x0 + 2 * radius > canvas.cols || y0 + 2 * radius > canvas.rows) {
        return 1.0;   // не разобрали — считаем обычным
    }
    const cv::Mat patch = canvas(cv::Rect(x0, y0, 2 * radius, 2 * radius));
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    // Плитка — зелёная (фон). Всё остальное вокруг юнита — основание.
    cv::Mat tile;
    cv::inRange(hsv, cv::Scalar(26, 130, 110), cv::Scalar(56, 255, 255), tile);
    cv::Mat mask = cv::Mat::ones(patch.size(), CV_8U);
    mask.setTo(0, tile > 0);
    // Оставляем только круг радиуса radius: иначе область основания
    // захватывает соседние клетки и заполненность считается неверно.
    cv::Mat circle_mask(patch.size(), CV_8U, cv::Scalar(0));
    cv::circle(circle_mask, cv::Point(radius, radius), radius, cv::Scalar(255), -1);
    mask.setTo(0, circle_mask == 0);
    // Область, связанная с центром юнита.
    cv::Mat labels;
    cv::connectedComponents(mask, labels, 8);
    int center_label = labels.at<int>(radius, radius);
    if (center_label == 0) {
        // Центр попал на плитку: нулевой номер у самой плитки, а это не
        // основание — берём ближайшую по центру область.
        double best = 1e9;
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                const int label = labels.at<int>(y, x);
                if (label == 0) continue;
                const double d = std::hypot(x - radius, y - radius);
                if (d < best) {
                    best = d;
                    center_label = label;
                }
            }
        }
    }
    int area = 0;
    double rmax = 0.0;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) != center_label) continue;
            ++area;
            const double dx = x - radius;
            const double dy = y - radius;
            rmax = std::max(rmax, std::hypot(dx, dy));
        }
    }
    if (area < 100 || rmax < 8.0) return 1.0;
    return area / (M_PI * rmax * rmax);
}

// Форма основания по заполненности: см. merge.hpp.
PedestalShape pedestal_shape_class(double fill) {
    if (fill >= kRank1Fill) return PedestalShape::Circle;
    if (fill >= kRank2Fill) return PedestalShape::Oval;
    return PedestalShape::Rhombus;
}

const char* pedestal_shape_name(PedestalShape shape) {
    switch (shape) {
        case PedestalShape::Circle: return "круг";
        case PedestalShape::Oval: return "капля";
        case PedestalShape::Rhombus: return "щит";
    }
    return "круг";
}

// Круглое ли основание: см. merge.hpp.
bool is_round_pedestal(const cv::Mat& canvas, const Pawn& pawn) {
    // У круга заполненность ~0.9 и больше, у ромба/овала — меньше.
    return pedestal_fill(canvas, pawn) >= 0.88;
}

std::vector<Pawn> find_pawns(const cv::Mat& canvas, const ArenaFrame& arena) {
    std::vector<Pawn> pawns;
    if (canvas.empty() || !arena.found) return pawns;
    const std::vector<cv::Rect> halves = {arena.top, arena.bottom};
    const std::vector<PawnBase> bases = find_unit_bases(canvas, halves);
    for (const PawnBase& base : bases) {
        // Своя половина — нижняя, чужая — верхняя. В «Подземельях» (Co-Op)
        // своё поле — комната: там по высоте не различить, потому что по
        // дороге к комнате идут и свои, и чужие юниты.
        bool ours;
        if (arena.bounded) {
            ours = base.center.x >= arena.bottom.x && base.center.x < arena.bottom.x + arena.bottom.width &&
                   base.center.y >= arena.bottom.y && base.center.y < arena.bottom.y + arena.bottom.height;
        } else {
            ours = base.center.y >= arena.bottom.y && base.center.y < arena.bottom.y + arena.bottom.height;
        }
        Pawn pawn;
        pawn.center = base.center;
        pawn.radius = base.radius;
        pawn.ours = ours;
        pawn.fill = pedestal_fill(canvas, pawn);
        pawn.round_pedestal = pawn.fill >= 0.88;
        // Один юнит иногда даёт два круга (обруч и внутренний круг) —
        // такие дубли убираем.
        bool duplicate = false;
        for (const Pawn& other : pawns) {
            const double dx = other.center.x - pawn.center.x;
            const double dy = other.center.y - pawn.center.y;
            if (std::hypot(dx, dy) < 0.85 * std::max(other.radius, pawn.radius)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) pawns.push_back(pawn);
    }
    return pawns;
}

std::vector<MergePair> find_merge_pairs(const cv::Mat& canvas, const std::vector<Pawn>& pawns) {
    std::vector<MergePair> pairs;
    if (canvas.empty() || pawns.size() < 2) return pairs;
    // Радиус, по которому берётся портрет: медианный по своим пешкам. Так
    // портреты получаются одного масштаба (камера в бою ходит) и не захватывают
    // плитки вокруг юнита.
    std::vector<int> radii;
    for (const Pawn& pawn : pawns) {
        if (pawn.ours) radii.push_back(pawn.radius);
    }
    if (radii.size() < 2) return pairs;
    std::sort(radii.begin(), radii.end());
    const int common = std::max(6, static_cast<int>(radii[radii.size() / 2] * 0.87));
    // Палитры своих пешек.
    std::vector<cv::Mat> hists(pawns.size());
    for (std::size_t i = 0; i < pawns.size(); ++i) {
        if (!pawns[i].ours) continue;
        hists[i] = color_histogram(canvas, pawns[i], common);
    }
    for (int i = 0; i < static_cast<int>(pawns.size()); ++i) {
        if (!pawns[i].ours || hists[i].empty()) continue;
        for (int j = i + 1; j < static_cast<int>(pawns.size()); ++j) {
            if (!pawns[j].ours || hists[j].empty()) continue;
            // Соседние клетки: шаг сетки — примерно 2.5 радиуса основания.
            const double limit = 3.3 * std::max(pawns[i].radius, pawns[j].radius);
            const double dx = pawns[i].center.x - pawns[j].center.x;
            const double dy = pawns[i].center.y - pawns[j].center.y;
            const double distance = std::hypot(dx, dy);
            if (distance < 1.0) continue;
            const double score = similarity(hists[i], hists[j]);
            if (score < 0.0) continue;
            // Одинаковый размер: у слитого юнита портрет больше, и с ним
            // сливать нельзя (ранг 1 с рангом 2 игра не принимает). Точный
            // порог выбирает вызывающий (см. kMergeSizeMax): здесь только
            // считаем, насколько пешки разного размера.
            const double size_diff =
                std::abs(pawns[i].radius - pawns[j].radius) /
                std::max(1.0, static_cast<double>(std::max(pawns[i].radius, pawns[j].radius)));
            // Форму основания в выборе пар не используем: она кодирует тип
            // карты, а не ранг (notes/ЗАДАЧА1_РАНГ.md). Ранг на поле не
            // виден — если ранги разные, игра сама откажет в слиянии.
            MergePair pair;
            // Тянем пешку, которая ближе к врагу, на пешку у базы: сливленный
            // юнит встаёт на место того, на кого перетащили, — то есть в тыл,
            // где его не достанут, а в тылу ставят новых.
            const bool i_is_forward = pawns[i].center.y < pawns[j].center.y;
            pair.source = i_is_forward ? i : j;
            pair.target = i_is_forward ? j : i;
            pair.score = score;
            pair.distance = distance;
            pair.adjacent = std::abs(dx) <= limit && std::abs(dy) <= limit;
            pair.size_diff = size_diff;
            // Ранг юнита — по форме подставки (см. merge.hpp): круг — ранг 1,
            // капля — ранг 2, щит — ранг 3. Игра сливает только юнитов
            // одного ранга, поэтому разные ранги — плохая пара.
            pair.rank_source = pedestal_rank(pawns[pair.source].fill);
            pair.rank_target = pedestal_rank(pawns[pair.target].fill);
            pair.same_rank = pair.rank_source == pair.rank_target;
            pairs.push_back(pair);
        }
    }
    // Самая похожая пара; при равенстве — самая близкая.
    std::stable_sort(pairs.begin(), pairs.end(), [](const MergePair& a, const MergePair& b) {
        if (std::abs(a.score - b.score) > 1e-6) return a.score > b.score;
        return a.distance < b.distance;
    });
    return pairs;
}

// Оценка успеха слияния: см. merge.hpp.
double merge_success_estimate(const MergePair& pair) {
    // Разные ранги игра не сливает: «ранг 1 + ранг 2» — отказ (живой опыт
    // 24.10). Такие пары бот не пробует вовсе, оценка близка к нулю.
    if (!pair.same_rank) return 0.05;
    double probability;
    // Таблица — по замерам 25.10 (46 попыток, PvP): >= 0.90 — 96 %,
    // 0.85..0.90 — 92 %, 0.78..0.85 — 58 %, ниже 0.78 — 12 %.
    if (pair.score >= 0.90) {
        probability = 0.96;
    } else if (pair.score >= 0.85) {
        probability = 0.92;
    } else if (pair.score >= 0.78) {
        probability = 0.58;
    } else {
        probability = 0.12;
    }
    // Пешки заметно разного размера — скорее всего, разные ранги (юнит
    // второго ранга нарисован крупнее). Замер 25.10: разница размеров до
    // 8 % — успех в 95 % случаев, 8..12 % — в 66 %, больше — в 50 %.
    if (pair.size_diff > kMergeSizeMax) {
        probability *= 0.55;
    } else if (pair.size_diff > kMergeSizeGood) {
        probability *= 0.75;
    }
    return probability;
}

int count_same_pawns(const cv::Mat& canvas, const std::vector<Pawn>& pawns, const Pawn& reference) {
    if (canvas.empty() || !reference.ours) return 0;
    std::vector<int> radii;
    for (const Pawn& pawn : pawns) {
        if (pawn.ours) radii.push_back(pawn.radius);
    }
    if (radii.empty()) return 0;
    std::sort(radii.begin(), radii.end());
    const int common = std::max(6, static_cast<int>(radii[radii.size() / 2] * 0.87));
    const cv::Mat ref = color_histogram(canvas, reference, common);
    if (ref.empty()) return 0;
    int count = 0;
    for (const Pawn& pawn : pawns) {
        if (!pawn.ours) continue;
        const cv::Mat other = color_histogram(canvas, pawn, common);
        if (other.empty()) continue;
        if (similarity(ref, other) >= kMergeMatchMin) ++count;
    }
    return count;
}

double pawn_similarity(const cv::Mat& canvas, const Pawn& a, const Pawn& b) {
    if (canvas.empty()) return -1.0;
    const int radius = std::max(6, static_cast<int>(std::max(a.radius, b.radius) * 0.87));
    const cv::Mat ha = color_histogram(canvas, a, radius);
    const cv::Mat hb = color_histogram(canvas, b, radius);
    return similarity(ha, hb);
}

cv::Point estimate_pawn_shift(const std::vector<Pawn>& before, const std::vector<Pawn>& after,
                               int max_shift) {
    // Совпадающие пешки: для каждой «до» ищем ближайшую «после».
    std::vector<int> dx;
    std::vector<int> dy;
    for (const Pawn& from : before) {
        int best = -1;
        double best_dist = 1e9;
        for (int i = 0; i < static_cast<int>(after.size()); ++i) {
            if (after[i].ours != from.ours) continue;
            const double d = std::hypot(after[i].center.x - from.center.x,
                                        after[i].center.y - from.center.y);
            if (d < best_dist) {
                best_dist = d;
                best = i;
            }
        }
        if (best < 0 || best_dist > max_shift) continue;
        dx.push_back(after[best].center.x - from.center.x);
        dy.push_back(after[best].center.y - from.center.y);
    }
    if (dx.empty()) return {0, 0};
    // Медиана: единичная ошибка сопоставления не должна портить сдвиг.
    std::sort(dx.begin(), dx.end());
    std::sort(dy.begin(), dy.end());
    return {dx[dx.size() / 2], dy[dy.size() / 2]};
}

int find_pawn_near(const cv::Mat& canvas, const std::vector<Pawn>& pawns,
                   const Pawn& reference, cv::Point predicted, int radius,
                   double min_similarity) {
    int best = -1;
    double best_score = min_similarity;
    double best_dist = 1e9;
    for (int i = 0; i < static_cast<int>(pawns.size()); ++i) {
        const Pawn& pawn = pawns[i];
        if (pawn.ours != reference.ours) continue;
        const double dist = std::hypot(pawn.center.x - predicted.x, pawn.center.y - predicted.y);
        if (dist > radius) continue;
        const double sim = pawn_similarity(canvas, reference, pawn);
        if (sim < best_score) continue;
        // Из похожих берём ближайшую к предсказанной точке.
        if (sim > best_score || dist < best_dist) {
            best_score = std::max(best_score, sim);
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

}  // namespace rr
