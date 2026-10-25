// Замер точности детекции юнитов: тот же смысл, что у tools/evaluate_vision.py,
// но на C++-детекторе.
//
// Вход: каталог с кадрами и файл эталонной разметки (TSV):
//     <имя кадра> <TAB> <тип> <TAB> <сторона> <TAB> <x> <TAB> <y>
// Координаты — в системе канонического поля.
//
// Вывод: recall, precision, точность типа и стороны, ошибка позиции —
// сводно и по типам.
//
// Запуск:
//     ./build/rrvision --frames DIR --truth FILE --templates assets/templates
//         --arena 0,0,360,640 --board 360x640 --threshold 0.78

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "models.hpp"
#include "planner.hpp"
#include "vision.hpp"

namespace {

// Одна эталонная метка.
struct Truth {
    std::string frame;
    std::string kind;
    rr::Team team = rr::Team::Unknown;
    double x = 0.0;
    double y = 0.0;
};

// Счётчики по одному типу (или по всем сразу).
struct Stats {
    int tp = 0;
    int fp = 0;
    int fn = 0;
    // Ошибки позиции (в пикселях поля).
    std::vector<double> position_errors;
    // Ошибки типа: «истина -> предсказано».
    std::map<std::string, int> confusions;
    // Ошибки стороны.
    int team_errors = 0;

    double recall() const {
        const int total = tp + fn;
        return total > 0 ? static_cast<double>(tp) / total : 0.0;
    }
    double precision() const {
        const int total = tp + fp;
        return total > 0 ? static_cast<double>(tp) / total : 0.0;
    }
    double mean_position_error() const {
        if (position_errors.empty()) return 0.0;
        double sum = 0.0;
        for (double value : position_errors) sum += value;
        return sum / static_cast<double>(position_errors.size());
    }
};

// Разбор аргументов «ключ=значение».
bool parse_pairs(const std::string& text, std::map<std::string, std::string>* out) {
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t eq = item.find('=');
        if (eq == std::string::npos) return false;
        out->emplace(item.substr(0, eq), item.substr(eq + 1));
    }
    return true;
}

// Разбор диапазона оттенков «a,b» или «a-b,c-d».
bool parse_ranges(const std::string& text, std::vector<std::pair<double, double>>* out) {
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t dash = item.find('-', item.empty() ? 0 : 1);
        if (dash == std::string::npos) return false;
        out->emplace_back(std::stod(item.substr(0, dash)), std::stod(item.substr(dash + 1)));
    }
    return true;
}

// Разбор «x0,y0,x1,y1» в прямоугольник.
bool parse_rect(const std::string& text, cv::Rect* out) {
    std::stringstream stream(text);
    std::string item;
    std::vector<int> values;
    while (std::getline(stream, item, ',')) values.push_back(std::stoi(item));
    if (values.size() != 4) return false;
    *out = cv::Rect(values[0], values[1], values[2] - values[0], values[3] - values[1]);
    return true;
}

// Прочитать эталонную разметку.
std::vector<Truth> load_truth(const std::string& path) {
    std::vector<Truth> truth;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "разметка не прочитана: " << path << std::endl;
        return truth;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream stream(line);
        std::string frame;
        std::string kind;
        std::string team;
        std::string x;
        std::string y;
        if (!std::getline(stream, frame, '\t')) continue;
        if (!std::getline(stream, kind, '\t')) continue;
        if (!std::getline(stream, team, '\t')) continue;
        if (!std::getline(stream, x, '\t')) continue;
        if (!std::getline(stream, y, '\t')) continue;
        Truth item;
        item.frame = frame;
        item.kind = kind;
        item.team = team == "ally" ? rr::Team::Ally
                                   : (team == "enemy" ? rr::Team::Enemy : rr::Team::Unknown);
        item.x = std::stod(x);
        item.y = std::stod(y);
        truth.push_back(item);
    }
    return truth;
}

// Показать число с тремя знаками.
std::string ratio(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string frames_dir;
    std::string truth_path;
    std::string templates_dir = "assets/templates";
    std::string map_text;
    std::string types_text;
    std::string arena_text;
    std::string board_text;
    std::string ally_text;
    std::string enemy_text;
    double threshold = 0.78;
    double nms_iou = 0.30;
    double scale = 1.0;
    double radius = 26.0;
    double split = 0.5;
    int max_per_template = 24;
    bool all_templates = false;
    bool team_by_position = false;
    bool print_detections = false;
    bool by_base = false;
    double base_radius_min = 24.0;
    double base_radius_max = 40.0;
    double base_min_score = 0.045;
    double base_min_support = 0.0;
    double base_gradient = 40.0;
    bool keep_sprite = false;
    bool full_vote = false;
    int base_min_count = 4;
    double base_match_radius = 22.0;
    int classify_radius = 3;
    double loose_threshold = 1.0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "не хватает значения у " << name << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--frames") frames_dir = next("--frames");
        else if (arg == "--truth") truth_path = next("--truth");
        else if (arg == "--templates") templates_dir = next("--templates");
        else if (arg == "--map") map_text = next("--map");
        else if (arg == "--types") types_text = next("--types");
        else if (arg == "--arena") arena_text = next("--arena");
        else if (arg == "--board") board_text = next("--board");
        else if (arg == "--threshold") threshold = std::stod(next("--threshold"));
        else if (arg == "--nms") nms_iou = std::stod(next("--nms"));
        else if (arg == "--scale") scale = std::stod(next("--scale"));
        else if (arg == "--radius") radius = std::stod(next("--radius"));
        else if (arg == "--split") split = std::stod(next("--split"));
        else if (arg == "--max-per-template") max_per_template = std::stoi(next("--max-per-template"));
        else if (arg == "--all-templates") all_templates = true;
        else if (arg == "--team-by-position") team_by_position = true;
        else if (arg == "--print-detections") print_detections = true;
        else if (arg == "--by-base") by_base = true;
        else if (arg == "--base-radius") {
            // «min,max» или одно число (тогда min=max=число).
            const std::string value = next("--base-radius");
            const size_t comma = value.find(',');
            if (comma == std::string::npos) {
                base_radius_min = std::stod(value);
                base_radius_max = base_radius_min;
            } else {
                base_radius_min = std::stod(value.substr(0, comma));
                base_radius_max = std::stod(value.substr(comma + 1));
            }
        } else if (arg == "--base-radius-min") base_radius_min = std::stod(next("--base-radius-min"));
        else if (arg == "--base-radius-max") base_radius_max = std::stod(next("--base-radius-max"));
        else if (arg == "--base-min-score") base_min_score = std::stod(next("--base-min-score"));
        else if (arg == "--base-min-support") base_min_support = std::stod(next("--base-min-support"));
        else if (arg == "--base-gradient") base_gradient = std::stod(next("--base-gradient"));
        else if (arg == "--base-keep-sprite") keep_sprite = true;
        else if (arg == "--base-full-vote") full_vote = true;
        else if (arg == "--base-min-count") base_min_count = std::stoi(next("--base-min-count"));
        else if (arg == "--base-match-radius") base_match_radius = std::stod(next("--base-match-radius"));
        else if (arg == "--classify-radius") classify_radius = std::stoi(next("--classify-radius"));
        else if (arg == "--loose-threshold") loose_threshold = std::stod(next("--loose-threshold"));
        else if (arg == "--ally-hues") ally_text = next("--ally-hues");
        else if (arg == "--enemy-hues") enemy_text = next("--enemy-hues");
        else if (arg == "--print-defaults") {
            // Значения по умолчанию (продублированы из config/live.yaml):
            // тест сверяет их с конфигом, чтобы они не разошлись.
            for (const auto& [kind, file] : rr::default_live_template_map()) {
                std::cout << "ЭТАЛОН " << kind << " " << file << std::endl;
            }
            for (const rr::UnitSpec& spec : rr::default_live_specs()) {
                std::cout << "ТИП " << spec.name << " cost=" << spec.cost << " hp=" << spec.hp
                          << " damage=" << spec.damage << " interval=" << spec.attack_interval
                          << " speed=" << spec.speed << " range=" << spec.attack_range
                          << " radius=" << spec.radius << " family=" << spec.family << " counters=";
                for (size_t i = 0; i < spec.counters.size(); ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << spec.counters[i];
                }
                std::cout << std::endl;
            }
            return 0;
        }
        else {
            std::cerr << "неизвестный аргумент: " << arg << std::endl;
            return 2;
        }
    }
    if (frames_dir.empty() || truth_path.empty()) {
        std::cerr << "нужны --frames и --truth" << std::endl;
        return 2;
    }

    rr::VisionConfig config;
    config.templates_dir = templates_dir;
    config.match_threshold = threshold;
    config.nms_iou = nms_iou;
    config.max_per_template = max_per_template;
    config.template_scale = scale;
    config.detect_all_templates = all_templates;
    config.team_by_position = team_by_position;
    config.team_split = split;
    // Поиск юнитов по круглому основанию (мобильный клиент).
    config.detect_by_base = by_base;
    config.auto_detect_by_base = false;
    config.base_radius_min = base_radius_min;
    config.base_radius_max = base_radius_max;
    config.base_min_score = base_min_score;
    config.base_min_support = base_min_support;
    config.base_gradient_percentile = base_gradient;
    config.base_drop_sprite_circles = !keep_sprite;
    config.base_lower_half_only = !full_vote;
    config.base_min_count = base_min_count;
    config.base_match_radius = base_match_radius;
    config.classify_radius = classify_radius;
    config.loose_template_threshold = loose_threshold;
    if (!map_text.empty()) {
        std::map<std::string, std::string> mapping;
        if (!parse_pairs(map_text, &mapping)) {
            std::cerr << "не разобран --map: " << map_text << std::endl;
            return 2;
        }
        config.template_map = mapping;
    }
    if (!ally_text.empty() && !parse_ranges(ally_text, &config.faction.ally_hue_ranges)) {
        std::cerr << "не разобраны диапазоны своих: " << ally_text << std::endl;
        return 2;
    }
    if (!enemy_text.empty() && !parse_ranges(enemy_text, &config.faction.enemy_hue_ranges)) {
        std::cerr << "не разобраны диапазоны чужих: " << enemy_text << std::endl;
        return 2;
    }
    // Типы юнитов: имена совпадают с именами в --map, иначе эталоны не ищутся.
    std::vector<std::string> types;
    if (!types_text.empty()) {
        std::stringstream stream(types_text);
        std::string item;
        while (std::getline(stream, item, ',')) {
            if (!item.empty()) types.push_back(item);
        }
    } else {
        for (const auto& [name, file] : config.template_map) {
            (void)file;
            types.push_back(name);
        }
    }
    for (const std::string& name : types) {
        rr::UnitSpec spec;
        spec.name = name;
        config.units.push_back(spec);
    }

    rr::VisionEngine engine(config);
    if (!engine.load_templates()) {
        std::cerr << "эталоны не загружены из " << templates_dir << std::endl;
        return 1;
    }

    const std::vector<Truth> truth = load_truth(truth_path);
    std::map<std::string, std::vector<Truth>> truth_by_frame;
    for (const Truth& item : truth) truth_by_frame[item.frame].push_back(item);

    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(frames_dir)) {
        if (entry.path().extension() != ".png") continue;
        files.push_back(entry.path().filename().string());
    }
    std::sort(files.begin(), files.end());

    Stats total;
    std::map<std::string, Stats> per_kind;
    int frames = 0;

    for (const std::string& file : files) {
        cv::Mat canvas = cv::imread((std::filesystem::path(frames_dir) / file).string(), cv::IMREAD_COLOR);
        if (canvas.empty()) {
            std::cerr << "кадр не прочитан: " << file << std::endl;
            continue;
        }
        rr::ArenaLayout arena;
        if (!arena_text.empty()) {
            cv::Rect rect;
            if (!parse_rect(arena_text, &rect)) {
                std::cerr << "не разобран --arena: " << arena_text << std::endl;
                return 2;
            }
            arena.quad = {cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)),
                          cv::Point2f(static_cast<float>(rect.x + rect.width),
                                      static_cast<float>(rect.y)),
                          cv::Point2f(static_cast<float>(rect.x + rect.width),
                                      static_cast<float>(rect.y + rect.height)),
                          cv::Point2f(static_cast<float>(rect.x),
                                      static_cast<float>(rect.y + rect.height))};
        } else {
            arena.quad = {cv::Point2f(0.0f, 0.0f),
                          cv::Point2f(static_cast<float>(canvas.cols), 0.0f),
                          cv::Point2f(static_cast<float>(canvas.cols), static_cast<float>(canvas.rows)),
                          cv::Point2f(0.0f, static_cast<float>(canvas.rows))};
        }
        if (!board_text.empty()) {
            const size_t x = board_text.find('x');
            arena.board_size = cv::Size(std::stoi(board_text.substr(0, x)), std::stoi(board_text.substr(x + 1)));
        } else {
            arena.board_size = cv::Size(canvas.cols, canvas.rows);
        }
        arena.split = split;

        const cv::Mat board = rr::extract_field(canvas, arena);
        if (board.empty()) continue;
        const std::vector<rr::Detection> detections = engine.detect(board);
        ++frames;

        if (print_detections) {
            std::cout << "НАХОДКИ " << file;
            for (const rr::Detection& det : detections) {
                std::cout << " | " << det.name << " " << ratio(det.score) << " (" << det.center.x << ","
                          << det.center.y << ") " << rr::team_name(det.team);
            }
            std::cout << std::endl;
        }

        // Сопоставление с истиной: как в tools/evaluate_vision.py — ближайшая
        // ещё не использованная находка.
        const auto frame_truth = truth_by_frame.find(file);
        std::vector<bool> used(detections.size(), false);
        Stats frame_stats;
        if (frame_truth != truth_by_frame.end()) {
            for (const Truth& item : frame_truth->second) {
                int best_index = -1;
                double best_distance = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < detections.size(); ++i) {
                    if (used[i]) continue;
                    const double d = std::hypot(detections[i].center.x - item.x,
                                                detections[i].center.y - item.y);
                    if (d < best_distance) {
                        best_distance = d;
                        best_index = static_cast<int>(i);
                    }
                }
                Stats& kind_stats = per_kind[item.kind];
                if (best_index >= 0 && best_distance <= radius) {
                    used[static_cast<size_t>(best_index)] = true;
                    const rr::Detection& det = detections[static_cast<size_t>(best_index)];
                    ++frame_stats.tp;
                    ++kind_stats.tp;
                    frame_stats.position_errors.push_back(best_distance);
                    kind_stats.position_errors.push_back(best_distance);
                    if (det.name != item.kind && item.kind != "-" && !item.kind.empty()) {
                        ++kind_stats.confusions[item.kind + "->" + det.name];
                        ++frame_stats.confusions[item.kind + "->" + det.name];
                    }
                    if (det.team != item.team) {
                        ++kind_stats.team_errors;
                        ++frame_stats.team_errors;
                    }
                } else {
                    ++frame_stats.fn;
                    ++kind_stats.fn;
                }
            }
        }
        for (size_t i = 0; i < detections.size(); ++i) {
            if (used[i]) continue;
            ++frame_stats.fp;
            Stats& kind_stats = per_kind[detections[i].name];
            ++kind_stats.fp;
        }
        total.tp += frame_stats.tp;
        total.fp += frame_stats.fp;
        total.fn += frame_stats.fn;
        total.team_errors += frame_stats.team_errors;
        total.position_errors.insert(total.position_errors.end(), frame_stats.position_errors.begin(),
                                     frame_stats.position_errors.end());
        for (const auto& [key, value] : frame_stats.confusions) total.confusions[key] += value;
    }

    const int truth_total = total.tp + total.fn;
    const int detected_total = total.tp + total.fp;
    int confusion_total = 0;
    for (const auto& [key, value] : total.confusions) {
        (void)key;
        confusion_total += value;
    }
    const double type_accuracy =
        total.tp > 0 ? 1.0 - static_cast<double>(confusion_total) / total.tp : 0.0;
    const double team_accuracy =
        total.tp > 0 ? 1.0 - static_cast<double>(total.team_errors) / total.tp : 0.0;

    std::cout << "ИТОГО: кадров=" << frames << " | истина=" << truth_total << " | найдено="
              << detected_total << " | recall=" << ratio(total.recall())
              << " | precision=" << ratio(total.precision()) << " | тип=" << ratio(type_accuracy)
              << " | сторона=" << ratio(team_accuracy) << " | ошибка позиции="
              << ratio(total.mean_position_error()) << "px | TP=" << total.tp << " | FP=" << total.fp
              << " | FN=" << total.fn << std::endl;
    for (const auto& [kind, stats] : per_kind) {
        // Тип «-» означает «в разметке не указан» (настоящая игра): отдельно
        // по нему считать нечего.
        if (kind == "-") continue;
        std::cout << "ПО ТИПАМ: " << kind << " | TP=" << stats.tp << " | FP=" << stats.fp << " | FN="
                  << stats.fn << " | recall=" << ratio(stats.recall())
                  << " | precision=" << ratio(stats.precision()) << std::endl;
    }
    if (!total.confusions.empty()) {
        std::cout << "ПУТАНИЦЫ:";
        for (const auto& [key, value] : total.confusions) std::cout << " " << key << "=" << value;
        std::cout << std::endl;
    }
    return 0;
}
