#include "arena.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

namespace rr {
namespace {

// Кусок плитки: прямоугольник и его центр.
struct Piece {
    cv::Rect rect;
    cv::Point2f center() const {
        return {static_cast<float>(rect.x + rect.width / 2.0),
                static_cast<float>(rect.y + rect.height / 2.0)};
    }
};

// Собрать общий прямоугольник арены из кусков плиток.
//
// Поле арены — сплошная жёлто-зелёная плитка, но юниты, стоящие на ней,
// разбивают её на несколько кусков. Поэтому берём объединение всех
// достаточно крупных кусков, а не только тех, что рядом с самым большим:
// промежутки между кусками как раз и есть места, где стоят юниты.
//
// Мелкие куски (иконки карточек, портал) отбрасываем: их площадь много
// меньше площади арены.
bool merge_near(const std::vector<Piece>& pieces, const cv::Size& frame, cv::Rect* out,
                int* count) {
    if (pieces.empty()) return false;
    const Piece& biggest = *std::max_element(pieces.begin(), pieces.end(),
                                              [](const Piece& a, const Piece& b) {
                                                  return a.rect.area() < b.rect.area();
                                              });
    const double min_area = std::max(400.0, biggest.rect.area() * 0.03);
    // По ширине берём объединение кусков (юниты разбивают плитку), а по
    // высоте — только главный кусок: узкие полосы по краям кадра (рамки,
    // декорации) иначе затягивают половину арены вниз.
    int x1 = biggest.rect.x;
    int x2 = biggest.rect.x + biggest.rect.width;
    int taken = 0;
    for (const Piece& p : pieces) {
        if (p.rect.area() < min_area) continue;
        ++taken;
        x1 = std::min(x1, p.rect.x);
        x2 = std::max(x2, p.rect.x + p.rect.width);
    }
    if (x2 - x1 < frame.width * 0.25) return false;
    *out = cv::Rect(x1, biggest.rect.y, x2 - x1, biggest.rect.height);
    *count = taken;
    return true;
}

// Собрать прямоугольник половины арены «посевом и расширением».
//
// Правило другое, чем у merge_near: самый крупный кусок — «семя», к нему
// присоединяются только соседние куски. Именно так половины арены находят
// в tools/prepare_real_frames.py, а от этого зависит канонический кадр, а
// значит и совпадение с эталонами юнитов. Правило merge_near даёт другой
// прямоугольник (шире, но ниже), и эталоны с ним не совпадают.
bool merge_seed_grow(const std::vector<Piece>& pieces, cv::Rect* out) {
    if (pieces.empty()) return false;
    // Крупные куски: ищем семя — самый большой из них.
    const Piece* seed = nullptr;
    for (const Piece& piece : pieces) {
        const bool wide = piece.rect.width >= 150 && piece.rect.height >= 100 &&
                          piece.rect.area() >= 8000;
        if (!wide) continue;
        if (seed == nullptr || piece.rect.area() > seed->rect.area()) seed = &piece;
    }
    if (seed == nullptr) return false;

    int x1 = seed->rect.x;
    int y1 = seed->rect.y;
    int x2 = seed->rect.x + seed->rect.width;
    int y2 = seed->rect.y + seed->rect.height;
    for (int round = 0; round < 4; ++round) {
        bool changed = false;
        for (const Piece& piece : pieces) {
            const double cx = piece.rect.x + piece.rect.width / 2.0;
            const double cy = piece.rect.y + piece.rect.height / 2.0;
            const bool near_y = cy >= y1 - 40 && cy <= y2 + 40;
            const bool near_x = cx >= x1 - 90 && cx <= x2 + 90;
            if (!near_y || !near_x) continue;
            const int nx1 = std::min(x1, piece.rect.x);
            const int ny1 = std::min(y1, piece.rect.y);
            const int nx2 = std::max(x2, piece.rect.x + piece.rect.width);
            const int ny2 = std::max(y2, piece.rect.y + piece.rect.height);
            if (nx1 != x1 || ny1 != y1 || nx2 != x2 || ny2 != y2) {
                x1 = nx1;
                y1 = ny1;
                x2 = nx2;
                y2 = ny2;
                changed = true;
            }
        }
        if (!changed) break;
    }
    *out = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    return true;
}

}  // namespace

namespace {

// Диапазон цветов плиток арены.
struct TileRange {
    cv::Scalar low;
    cv::Scalar high;
};

// Обычная арена: яркие жёлто-зелёные плитки.
const TileRange kBrightTiles = {cv::Scalar(28, 170, 170), cv::Scalar(40, 255, 255)};
// Тёмная арена (ночной стиль, бои в режиме «Трибунал»): те же плитки, но
// вдвое темнее — по яркости они не подходят.
const TileRange kDimTiles = {cv::Scalar(35, 80, 55), cv::Scalar(60, 255, 255)};

}  // namespace

namespace {

// Общая часть поиска арены: куски плиток делятся на верхнюю и нижнюю
// половины, и каждая собирается в прямоугольник.
//
// seed_grow — собирать прямоугольник «посевом и расширением» (как в
// tools/prepare_real_frames.py) или по объединению кусков (merge_near).
ArenaFrame find_arena_impl(const cv::Mat& canvas, bool seed_grow, const TileRange& tiles) {
    ArenaFrame result;
    if (canvas.empty()) return result;
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, tiles.low, tiles.high, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, {9, 9}));

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    // Порог площади куска: в канонической геометрии — как в
    // tools/prepare_real_frames.py (400), иначе мелкие куски не годятся в
    // «семя» и половина арены выходит другого размера.
    const int min_area = seed_grow ? 400 : std::max(400, canvas.rows * canvas.cols / 1000);
    std::vector<Piece> top;
    std::vector<Piece> bottom;
    const double middle = canvas.rows / 2.0;
    for (int i = 1; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < min_area) continue;
        const cv::Rect rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                            stats.at<int>(i, cv::CC_STAT_WIDTH),
                            stats.at<int>(i, cv::CC_STAT_HEIGHT));
        Piece piece;
        piece.rect = rect;
        if (rect.y + rect.height / 2.0 < middle) {
            top.push_back(piece);
        } else {
            bottom.push_back(piece);
        }
    }
    if (top.empty() && bottom.empty()) return result;

    cv::Rect top_rect;
    cv::Rect bottom_rect;
    int top_count = 0;
    int bottom_count = 0;
    if (seed_grow) {
        // Половина может не найтись: в тёмной арене (ночной стиль, режим
        // «Трибунал») часть поля закрыта, и тогда верхняя половина не
        // находится вовсе. С половиной можно работать и без второй.
        if (!top.empty()) {
            if (!merge_seed_grow(top, &top_rect)) top_rect = cv::Rect();
            top_count = static_cast<int>(top.size());
        }
        if (!bottom.empty()) {
            if (!merge_seed_grow(bottom, &bottom_rect)) bottom_rect = cv::Rect();
            bottom_count = static_cast<int>(bottom.size());
        }
    } else {
        if (!top.empty() && !merge_near(top, canvas.size(), &top_rect, &top_count)) {
            top_rect = cv::Rect();
        }
        if (!bottom.empty() && !merge_near(bottom, canvas.size(), &bottom_rect, &bottom_count)) {
            bottom_rect = cv::Rect();
        }
    }
    if (top_rect.width <= 0 && bottom_rect.width <= 0) return result;
    // Проверка правдоподобия: половина арены — это широкий прямоугольник в
    // середине экрана. Если получилось что-то странное (например, трава
    // вокруг поля), такой результат не годится. На канонических кадрах
    // (маленький кадр) проверка не нужна — там арена занимает его целиком.
    if (canvas.cols >= 700) {
        for (const cv::Rect& rect : {top_rect, bottom_rect}) {
            if (rect.width <= 0) continue;
            if (rect.width < canvas.cols * 0.2 || rect.width > canvas.cols * 0.8) return result;
            if (rect.height < canvas.rows * 0.1 || rect.height > canvas.rows * 0.5) return result;
        }
    }
    // Между половинами должна быть полоса со счётчиком волны.
    if (top_rect.width > 0 && bottom_rect.width > 0 &&
        bottom_rect.y - (top_rect.y + top_rect.height) < canvas.rows * 0.04) {
        return result;
    }

    result.found = true;
    result.top = top_rect;
    result.bottom = bottom_rect;
    result.tiles_top = top_count;
    result.tiles_bottom = bottom_count;
    // Общий прямоугольник обеих половин (пустой половины в расчёт не берём).
    const bool has_top = top_rect.width > 0;
    const bool has_bottom = bottom_rect.width > 0;
    const int x = has_top && has_bottom ? std::min(top_rect.x, bottom_rect.x)
                                        : (has_top ? top_rect.x : bottom_rect.x);
    const int y = has_top ? top_rect.y : bottom_rect.y;
    const int right = has_top && has_bottom
                          ? std::max(top_rect.x + top_rect.width, bottom_rect.x + bottom_rect.width)
                          : (has_top ? top_rect.x + top_rect.width : bottom_rect.x + bottom_rect.width);
    const int bottom_edge = has_top && has_bottom
                                ? bottom_rect.y + bottom_rect.height
                                : (has_top ? top_rect.y + top_rect.height
                                           : bottom_rect.y + bottom_rect.height);
    result.full = cv::Rect(x, y, right - x, bottom_edge - y);
    return result;
}

}  // namespace

ArenaFrame find_arena(const cv::Mat& canvas) {
    ArenaFrame result = find_arena_impl(canvas, false, kBrightTiles);
    if (!result.found) result = find_arena_impl(canvas, false, kDimTiles);
    // В «Подземельях» (Co-Op) цветных плиток нет: там комнаты, обнесённые
    // кремовыми стенами. По плиткам там находится одна половина (зелёные
    // пятна врага), и бот считал бы поле найденным, а пешек не видел бы.
    // Поэтому: если по плиткам нашлась не обе половины — пробуем комнаты.
    if (!result.found || result.top.width <= 0 || result.bottom.width <= 0) {
        ArenaFrame dungeon = find_arena_dungeon(canvas);
        if (dungeon.found) result = dungeon;
    }
    return result;
}

ArenaFrame find_arena_dungeon(const cv::Mat& canvas) {
    ArenaFrame result;
    if (canvas.empty()) return result;
    cv::Mat hsv;
    cv::cvtColor(canvas, hsv, cv::COLOR_BGR2HSV);
    // Кремовая маска: стены и угловые столбики комнат.
    //
    // Стена — бледный бежевый: светлота высокая, насыщенность низкая. В
    // старом дизайне она почти белая (V~250), в новом — темнее и желтее
    // (V~185, S~89), а плитки поля чуть более насыщенные (S~105): по
    // насыщенности их и различаем.
    cv::Mat cream;
    cv::inRange(hsv, cv::Scalar(10, 40, 175), cv::Scalar(30, 100, 255), cream);
    // Плитки: рыжевато-песочные квадраты поля.
    cv::Mat tiles;
    cv::inRange(hsv, cv::Scalar(8, 70, 150), cv::Scalar(24, 180, 226), tiles);
    // Стена комнаты — вертикальная полоса кремового цвета. Среди них
    // выбираем самую правую слева от центра кадра и самую левую справа:
    // между ними и есть поле (слева от него — дорога, по которой ходят юниты).
    const int row_from = static_cast<int>(canvas.rows * 0.45);
    const int row_to = static_cast<int>(canvas.rows * 0.95);
    std::vector<std::pair<int, int>> walls;
    {
        int start = -1;
        for (int x = 0; x <= canvas.cols; ++x) {
            int count = 0;
            if (x < canvas.cols) {
                for (int y = row_from; y < row_to; ++y) {
                    if (cream.at<uchar>(y, x) > 0) ++count;
                }
            }
            const bool solid = count > (row_to - row_from) * 0.35;
            if (solid && start < 0) start = x;
            if (!solid && start >= 0) {
                if (x - start >= 4) walls.emplace_back(start, x);
                start = -1;
            }
        }
    }
    if (walls.size() < 2) return result;
    const int center = canvas.cols / 2;
    int left = -1;
    int right = -1;
    for (const auto& wall : walls) {
        if (wall.second <= center && wall.second > left) left = wall.second;
        if (wall.first > center && (right < 0 || wall.first < right)) right = wall.first;
    }
    if (left < 0 || right < 0 || right - left < canvas.cols * 0.25) return result;
    // Высота комнат — по вертикальной стене: строка относится к стене, если
    // кремового в ней больше половины ширины стены. Самый нижний кусок —
    // это стена нашей комнаты (выше — стены чужой комнаты и дороги между
    // ними).
    //
    // По плиткам height искать нельзя: когда комната забита юнитами, плитки
    // от них закрыты, и строки распадаются на мелкие куски.
    std::vector<std::pair<int, int>> runs;
    {
        // Правая стена — та группа, что начинается ближе всего к центру
        // кадра справа от него.
        int wall_index = -1;
        for (std::size_t i = 0; i < walls.size(); ++i) {
            if (walls[i].first <= center) continue;
            if (wall_index < 0 || walls[i].first < walls[wall_index].first) {
                wall_index = static_cast<int>(i);
            }
        }
        if (wall_index < 0) return result;
        const int wall_x1 = walls[wall_index].first;
        const int wall_x2 = std::min(canvas.cols, walls[wall_index].second);
        const int wall_w = std::max(1, wall_x2 - wall_x1);
        std::vector<bool> is_wall(canvas.rows, false);
        for (int y = 0; y < canvas.rows; ++y) {
            int count = 0;
            for (int x = wall_x1; x < wall_x2; ++x) {
                if (cream.at<uchar>(y, x) > 0) ++count;
            }
            is_wall[y] = count > wall_w * 0.5;
        }
        int start = -1;
        for (int y = 0; y <= canvas.rows; ++y) {
            const bool solid = y < canvas.rows && is_wall[y];
            if (solid && start < 0) start = y;
            if (!solid && start >= 0) {
                if (y - start >= 20) runs.emplace_back(start, y);
                start = -1;
            }
        }
        // Куски стены, разделённые небольшими промежутками (юниты у края),
        // склеиваем: стена одна и та же.
        std::vector<std::pair<int, int>> merged;
        for (const auto& run : runs) {
            if (!merged.empty() && run.first - merged.back().second < 80) {
                merged.back().second = run.second;
            } else {
                merged.push_back(run);
            }
        }
        runs.swap(merged);
    }
    if (runs.size() < 2) return result;
    const auto& ours = runs.back();
    if (ours.second - ours.first < canvas.rows * 0.10) return result;
    // Чужая комната — всё, что выше нашей (дорога между комнатами в это
    // попадает: для поиска чужих юнитов это не важно).
    const int enemy_top = runs.front().first;
    const int enemy_bottom = std::max(enemy_top + 1, ours.first - 60);
    result.found = true;
    result.top = cv::Rect(left, enemy_top, right - left, enemy_bottom - enemy_top);
    result.bottom = cv::Rect(left, ours.first, right - left, ours.second - ours.first);
    result.bounded = true;
    result.tiles_top = 1;
    result.tiles_bottom = 1;
    result.full = cv::Rect(left, result.top.y, right - left,
                           result.bottom.y + result.bottom.height - result.top.y);
    return result;
}

ArenaFrame find_arena_canonical(const cv::Mat& canvas) {
    ArenaFrame result = find_arena_impl(canvas, true, kBrightTiles);
    if (!result.found) result = find_arena_impl(canvas, true, kDimTiles);
    return result;
}

std::vector<PawnBase> find_unit_bases(const cv::Mat& canvas,
                                       const std::vector<cv::Rect>& halves) {
    std::vector<PawnBase> found;
    if (canvas.empty() || halves.empty()) return found;
    cv::Mat gray;
    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    // Параметры — как в tools/prepare_real_frames.py: иначе основания
    // находятся не те же и масштаб кадра получается другим.
    cv::Mat circles;
    // В OpenCV 5 результат HoughCircles в cv::Mat пишется построчно (одна
    // строка на все круги), поэтому берём std::vector: иначе читается только
    // первый круг, и основания юнитов не находятся вовсе.
    std::vector<cv::Vec3f> found_circles;
    cv::HoughCircles(gray, found_circles, cv::HOUGH_GRADIENT, 1.0, 20, 100, 30, 26, 44);
    if (found_circles.empty()) return found;
    for (const cv::Vec3f& circle : found_circles) {
        const cv::Point center(static_cast<int>(circle[0]), static_cast<int>(circle[1]));
        const int radius = static_cast<int>(circle[2]);
        for (const cv::Rect& half : halves) {
            if (center.x >= half.x - 12 && center.x <= half.x + half.width + 12 &&
                center.y >= half.y - 12 && center.y <= half.y + half.height + 12) {
                found.push_back({center, radius});
                break;
            }
        }
    }
    return found;
}

cv::Mat canonical_board(const cv::Mat& canvas, BoardMapping* mapping) {
    cv::Mat board;
    if (canvas.empty()) return board;
    // Арена ищется именно в канонической геометрии: от правила зависит
    // размер кадра, а значит и совпадение с эталонами юнитов.
    const ArenaFrame found = find_arena_canonical(canvas);
    if (!found.found) return board;
    const std::vector<cv::Rect> halves = {found.top, found.bottom};
    const std::vector<PawnBase> bases = find_unit_bases(canvas, halves);
    if (bases.empty()) return board;

    // Масштаб кадра: приводим медианный радиус оснований к опорному, чтобы
    // размер юнита не зависел от положения камеры.
    std::vector<int> radii;
    radii.reserve(bases.size());
    for (const PawnBase& base : bases) radii.push_back(base.radius);
    std::sort(radii.begin(), radii.end());
    const double median_radius = radii[radii.size() / 2];
    const double factor = kCanonicalRefRadius / std::max(1.0, static_cast<double>(median_radius));

    board = cv::Mat::zeros(kCanonicalH, kCanonicalW, CV_8UC3);
    const cv::Rect halves_rect[2] = {found.top, found.bottom};
    for (int index = 0; index < 2; ++index) {
        const cv::Rect& rect = halves_rect[index];
        // Половина могла не найтись (тёмная арена) — тогда в канонический
        // кадр она просто не попадает.
        if (rect.width <= 0 || rect.height <= 0) continue;
        const cv::Mat half = canvas(rect).clone();
        const cv::Size target(std::max(1, static_cast<int>(std::lround(rect.width * factor))),
                              std::max(1, static_cast<int>(std::lround(rect.height * factor))));
        cv::Mat resized;
        cv::resize(half, resized, target, 0, 0,
                   factor < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
        // Верхняя половина — с нуля, нижняя — после полосы волны.
        const int top = index == 0 ? 0 : kCanonicalHalfH + kCanonicalGap;
        // Центрируем по горизонтали: кадр может быть обрезан по-разному.
        const int left = std::max(0, (kCanonicalW - resized.cols) / 2);
        const int src_x = std::max(0, -left);
        const int src_y = std::max(0, -top);
        const int dst_x = std::max(0, left);
        const int dst_y = std::max(0, top);
        const int width = std::min(kCanonicalW - dst_x, resized.cols - src_x);
        const int height = std::min(kCanonicalH - dst_y, resized.rows - src_y);
        if (width > 0 && height > 0) {
            resized(cv::Rect(src_x, src_y, width, height))
                .copyTo(board(cv::Rect(dst_x, dst_y, width, height)));
        }
        if (mapping != nullptr) {
            mapping->half[index] = rect;
            mapping->factor = factor;
            mapping->left[index] = left;
            mapping->top[index] = top;
        }
    }
    return board;
}

cv::Point BoardMapping::to_frame(cv::Point board_point) const {
    // Половина определяется по вертикали: сверху — чужая, снизу — своя.
    int index = -1;
    if (board_point.y < kCanonicalHalfH) {
        index = 0;
    } else if (board_point.y >= kCanonicalHalfH + kCanonicalGap) {
        index = 1;
    }
    if (index < 0) return {-1, -1};
    const double scale = factor > 0.0 ? 1.0 / factor : 1.0;
    const cv::Rect& rect = half[index];
    const double x = rect.x + (board_point.x - left[index]) * scale;
    const double y = rect.y + (board_point.y - top[index]) * scale;
    return {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))};
}

}  // namespace rr
