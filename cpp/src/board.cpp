// Сетка высадки и занятость доски: см. board.hpp.

#include "board.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>

namespace rr {
namespace {

// Остаток по модулю шага (всегда в [0, spacing)).
double wrap(double value, double spacing) {
    double phase = std::fmod(value, spacing);
    if (phase < 0.0) phase += spacing;
    return phase;
}

// Насколько значение отстоит от ближайшей точки решётки с шагом spacing.
double deviation(double value, double phase, double spacing) {
    double d = std::fmod(value - phase, spacing);
    if (d < 0.0) d += spacing;
    if (d > spacing / 2.0) d -= spacing;
    return std::abs(d);
}

// Начало решётки по кругу: среднее остатков (обычное среднее здесь нельзя —
// остаток перескакивает через ноль).
double circular_phase(const std::vector<double>& values, double spacing) {
    double sum_sin = 0.0;
    double sum_cos = 0.0;
    for (const double value : values) {
        const double angle = 2.0 * M_PI * value / spacing;
        sum_sin += std::sin(angle);
        sum_cos += std::cos(angle);
    }
    return wrap(std::atan2(sum_sin, sum_cos) * spacing / (2.0 * M_PI), spacing);
}

// Точка внутри прямоугольника (с запасом).
bool inside(const cv::Rect& rect, cv::Point point, int margin) {
    return point.x >= rect.x - margin && point.x <= rect.x + rect.width + margin &&
           point.y >= rect.y - margin && point.y <= rect.y + rect.height + margin;
}

// «Пестрота» клетки: разброс яркости в пятне вокруг центра.
//
// У пустой клетки плитка ровная — разброс маленький. На занятой клетке лежит
// основание с юнитом, и картинка пёстрая — разброс большой.
double cell_busy(const cv::Mat& gray, cv::Point2d center, double radius) {
    const int r = std::max(3, static_cast<int>(std::lround(radius)));
    const int x = static_cast<int>(std::lround(center.x));
    const int y = static_cast<int>(std::lround(center.y));
    const int x0 = std::max(0, x - r);
    const int y0 = std::max(0, y - r);
    const int x1 = std::min(gray.cols, x + r);
    const int y1 = std::min(gray.rows, y + r);
    if (x1 - x0 < 4 || y1 - y0 < 4) return 0.0;
    const cv::Mat patch = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(patch, mean, deviation);
    return deviation[0];
}

// Точка решётки по индексам клетки.
cv::Point2d grid_point(const CellGrid& grid, int column, int row) {
    return cv::Point2d(grid.phase_x + column * grid.spacing,
                       grid.phase_y + row * grid.spacing);
}

}  // namespace

std::pair<int, int> CellGrid::cell_of(cv::Point point) const {
    if (!valid || spacing <= 0.0) return {-1, -1};
    const int column = static_cast<int>(std::lround((point.x - phase_x) / spacing));
    const int row = static_cast<int>(std::lround((point.y - phase_y) / spacing));
    return {column, row};
}

cv::Point CellGrid::point_of(std::pair<int, int> cell) const {
    if (!valid || spacing <= 0.0) return {-1, -1};
    return {static_cast<int>(std::lround(phase_x + cell.first * spacing)),
            static_cast<int>(std::lround(phase_y + cell.second * spacing))};
}

CellGrid estimate_grid(const std::vector<Pawn>& pawns) {
    CellGrid grid;
    std::vector<const Pawn*> ours;
    for (const Pawn& pawn : pawns) {
        if (pawn.ours) ours.push_back(&pawn);
    }
    // По одной пешке сетку не определить: неизвестен ни шаг, ни начало.
    if (ours.size() < 2) return grid;

    std::vector<int> radii;
    radii.reserve(ours.size());
    for (const Pawn* pawn : ours) radii.push_back(pawn->radius);
    std::sort(radii.begin(), radii.end());
    const double median_radius = radii[radii.size() / 2];
    if (median_radius < 4) return grid;
    // Шаг сетки — около 2.5 радиуса основания (замер живых боёв: 78 px при
    // радиусе 31). Точное значение неважно: ниже шаг уточняется перебором.
    const double base = 2.5 * median_radius;

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(ours.size());
    ys.reserve(ours.size());
    for (const Pawn* pawn : ours) {
        xs.push_back(pawn->center.x);
        ys.push_back(pawn->center.y);
    }

    // Перебор шага: для каждого считаем, сколько пешек попадает в одну
    // решётку. Лучший шаг — тот, что подтверждает больше всего пешек.
    double best_spacing = 0.0;
    double best_phase_x = 0.0;
    double best_phase_y = 0.0;
    int best_inliers = 0;
    double best_spread = 1e9;
    for (int step = -12; step <= 12; ++step) {
        const double spacing = base * (1.0 + 0.02 * step);
        if (spacing < 8.0) continue;
        const double phase_x = circular_phase(xs, spacing);
        const double phase_y = circular_phase(ys, spacing);
        int inliers = 0;
        double spread = 0.0;
        for (const Pawn* pawn : ours) {
            const double dx = deviation(pawn->center.x, phase_x, spacing);
            const double dy = deviation(pawn->center.y, phase_y, spacing);
            if (dx < 0.2 * spacing && dy < 0.2 * spacing) {
                ++inliers;
                spread += dx + dy;
            }
        }
        // Больше подтверждений — лучше; при равенстве — меньше разброс
        // (пешки ближе к решётке).
        if (inliers > best_inliers || (inliers == best_inliers && spread < best_spread)) {
            best_inliers = inliers;
            best_spread = spread;
            best_spacing = spacing;
            best_phase_x = phase_x;
            best_phase_y = phase_y;
        }
    }
    if (best_inliers < 2 || best_spacing <= 0.0) return grid;

    // Уточнение начала сетки по подтвердившим пешкам: из них остатки
    // считаются точнее (юниты, идущие по полю, в решётку не попадают).
    for (int iteration = 0; iteration < 2; ++iteration) {
        std::vector<double> in_x;
        std::vector<double> in_y;
        for (const Pawn* pawn : ours) {
            if (deviation(pawn->center.x, best_phase_x, best_spacing) < 0.2 * best_spacing &&
                deviation(pawn->center.y, best_phase_y, best_spacing) < 0.2 * best_spacing) {
                in_x.push_back(pawn->center.x);
                in_y.push_back(pawn->center.y);
            }
        }
        if (in_x.size() < 2) break;
        best_phase_x = circular_phase(in_x, best_spacing);
        best_phase_y = circular_phase(in_y, best_spacing);
    }

    int inliers = 0;
    for (const Pawn* pawn : ours) {
        if (deviation(pawn->center.x, best_phase_x, best_spacing) < 0.2 * best_spacing &&
            deviation(pawn->center.y, best_phase_y, best_spacing) < 0.2 * best_spacing) {
            ++inliers;
        }
    }
    // Если решётку подтверждают меньше половины пешек — оценка ненадёжна
    // (пешки могут стоять не на клетках вовсе).
    if (inliers * 2 < static_cast<int>(ours.size())) return grid;

    grid.valid = true;
    grid.spacing = best_spacing;
    grid.phase_x = best_phase_x;
    grid.phase_y = best_phase_y;
    grid.inliers = inliers;
    grid.considered = static_cast<int>(ours.size());
    return grid;
}

BoardState analyze_board(const cv::Mat& canvas,
                         const ArenaFrame& arena,
                         const std::vector<Pawn>& pawns) {
    BoardState state;
    const CellGrid grid = estimate_grid(pawns);
    if (!grid.valid || !arena.found || canvas.empty()) return state;

    // Клетки высадки: точки решётки внутри своей половины арены.
    //
    // Половина арены — это стены комнаты (в «Подземельях») или половина
    // поля, а сами клетки стоят внутри, не по краю. Поэтому берём все точки
    // решётки, что попали внутрь, и оставляем нужное число: 5 столбцов и
    // 3 ряда (ряды — ближние к своей базе, то есть нижние).
    const cv::Rect& zone = arena.bottom;
    std::vector<int> columns;
    std::vector<int> rows;
    for (int column = -12; column <= 24; ++column) {
        for (int row = -12; row <= 24; ++row) {
            const cv::Point2d point = grid_point(grid, column, row);
            if (!inside(zone, cv::Point(static_cast<int>(std::lround(point.x)),
                                        static_cast<int>(std::lround(point.y))),
                        0)) {
                continue;
            }
            if (std::find(columns.begin(), columns.end(), column) == columns.end()) {
                columns.push_back(column);
            }
            if (std::find(rows.begin(), rows.end(), row) == rows.end()) {
                rows.push_back(row);
            }
        }
    }
    std::sort(columns.begin(), columns.end());
    std::sort(rows.begin(), rows.end());
    if (columns.size() < kDeployCols || rows.size() < kDeployRows) return state;
    // Лишние столбцы (если половина арены накрыла больше, чем нужно) —
    // берём центральные.
    while (columns.size() > kDeployCols) {
        columns.erase(columns.begin());
    }
    // Ряды — нижние: своя база внизу.
    rows.erase(rows.begin(), rows.end() - kDeployRows);

    cv::Mat gray;
    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
    // Пятно клетки — примерно треть радиуса основания юнита.
    const double patch_radius = grid.spacing * 0.38;
    std::vector<std::pair<int, int>> occupied;
    std::vector<double> busy_values;
    for (const int row : rows) {
        for (const int column : columns) {
            const cv::Point2d point = grid_point(grid, column, row);
            const double busy = cell_busy(gray, point, patch_radius);
            busy_values.push_back(busy);
            if (busy > kCellBusyThreshold) occupied.emplace_back(column, row);
        }
    }
    // Проверка: пешки стоят на занятых клетках. Если нет — решётка посчитана
    // не по той фазе, и оценке доски верить нельзя.
    int checked = 0;
    int on_busy = 0;
    for (const Pawn& pawn : pawns) {
        if (!pawn.ours) continue;
        ++checked;
        if (std::find(occupied.begin(), occupied.end(), grid.cell_of(pawn.center)) !=
            occupied.end()) {
            ++on_busy;
        }
    }
    if (checked > 0 && on_busy * 2 < checked) return state;
    state.valid = true;
    state.grid = grid;
    state.capacity = kDeployCapacity;
    state.occupied = static_cast<int>(occupied.size());
    state.free_cells = std::max(0, state.capacity - state.occupied);
    state.occupied_cells = occupied;
    state.cell_busy = busy_values;
    return state;
}

}  // namespace rr
