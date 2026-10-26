// Проверочная программа: что делает игра, если перетащить карточку из колоды
// на пустую клетку своего поля.
//
// Бот в эту программу не входит. Программа ждёт бой (три стабильных кадра:
// HUD с энергией и рядом карточек, арена с полем), читает энергию, тянет
// карточку (слот колоды K, 1..5) настоящей мышью на клетку своего поля и
// снимает кадры до и после. Печатает энергию, доступность карточек
// (насыщенность слотов: серая карточка не готова) и пешек рядом с клеткой.
//
//   build/deck_probe --slot 1 --cell 0,2 [--out КАТАЛОГ]
//     клетка: столбец 0..4 слева направо, ряд 0..2 сверху вниз (свой ряд
//     клеток: ряд 2 — у нижней кромки поля, ближе к базе);
//   build/deck_probe --slot 2 --tap      # только нажатие на карточку
//
// Нужен эмулятор с игрой и окно эмулятора на экране (клик настоящей мышью).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "arena.hpp"
#include "common.hpp"
#include "game.hpp"
#include "hud.hpp"
#include "merge.hpp"

namespace {

void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// Средняя насыщенность центральной части слота: серая карточка — около нуля.
void print_slots(const cv::Mat& canvas, const rr::Hud& hud) {
    std::cout << "    слоты колоды (насыщенность):";
    for (int slot = 1; slot <= 5; ++slot) {
        std::cout << " " << slot << "=" << static_cast<int>(hud.deck_slot_saturation(canvas, slot));
    }
    std::cout << std::endl;
}

// Бой: HUD (энергия, флакон, ряд карточек) и поле — арена нужного размера.
// Одного HUD мало: в магазине ряд внизу тоже даёт «карточки».
struct FieldView {
    bool battle = false;
    rr::ArenaFrame arena;
};

FieldView field_view(const cv::Mat& canvas, const rr::Hud& hud) {
    FieldView view;
    if (canvas.empty() || !hud.in_battle(canvas)) return view;
    view.arena = rr::find_arena(canvas);
    view.battle = view.arena.found && view.arena.bottom.width > 250 && view.arena.bottom.height > 150;
    return view;
}

// Клетка своего поля по прямоугольнику нижней половины арены: 5 столбцов,
// 3 ряда. Прямоугольник арены в кадре гуляет на несколько пикселей, поэтому
// клетка попадает в центр ячейки с запасом (ячейка около 78 px).
cv::Point cell_point(const cv::Rect& field, int col, int row) {
    const double x = field.x + (col + 0.5) * field.width / 5.0;
    const double y = field.y + (row + 0.5) * field.height / 3.0;
    return cv::Point(static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y)));
}

}  // namespace

int main(int argc, char** argv) {
    int slot = 1;
    bool tap = false;
    int col = -1;
    int row = -1;
    std::string out_dir = "/tmp/opencode/deck_probe";
    int wait_battle_s = 180;
    bool summon_first = false;
    bool select_test = false;
    bool move_test = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--slot" && i + 1 < argc) {
            slot = std::atoi(argv[++i]);
        } else if (arg == "--tap") {
            tap = true;
        } else if (arg == "--summon-first") {
            summon_first = true;
        } else if (arg == "--select-test") {
            select_test = true;
        } else if (arg == "--move-test") {
            move_test = true;
        } else if (arg == "--cell" && i + 1 < argc) {
            const std::string value = argv[++i];
            const std::size_t comma = value.find(',');
            if (comma == std::string::npos) {
                std::cerr << "--cell КОЛОНКА,РЯД" << std::endl;
                return 2;
            }
            col = std::atoi(value.substr(0, comma).c_str());
            row = std::atoi(value.substr(comma + 1).c_str());
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--wait" && i + 1 < argc) {
            wait_battle_s = std::atoi(argv[++i]);
        } else {
            std::cerr << "неизвестный ключ: " << arg << std::endl;
            return 2;
        }
    }
    if (slot < 1 || slot > 5) {
        std::cerr << "--slot 1..5" << std::endl;
        return 2;
    }
    if (!tap && (col < 0 || col > 4 || row < 0 || row > 2)) {
        std::cerr << "нужен --cell КОЛОНКА,РЯД (колонка 0..4, ряд 0..2)" << std::endl;
        return 2;
    }
    std::filesystem::create_directories(out_dir);

    const std::filesystem::path root = "/home/lennistyle/rushroyale-lab";
    auto backend = rr::make_adb_backend(true);
    rr::Game game(backend.get());
    cv::Mat canvas = game.grab_canvas();
    if (canvas.empty()) {
        std::cerr << "кадр не получен (эмулятор не отвечает?)" << std::endl;
        return 1;
    }
    rr::Hud hud((root / "assets").string(), rr::emulator_layout().scaled_to(canvas.size()));

    // Бой подтверждаем тремя кадрами подряд, и арена не должна ехать:
    // в начале боя камера уходит в анимацию, и клетка смещается.
    int stable = 0;
    cv::Rect last_field;
    bool battle = false;
    for (int i = 0; i < wait_battle_s && !battle; ++i) {
        canvas = game.grab_canvas();
        const FieldView view = field_view(canvas, hud);
        if (view.battle) {
            const bool moved = last_field.area() > 0 &&
                               (std::abs(view.arena.bottom.x - last_field.x) > 6 ||
                                std::abs(view.arena.bottom.y - last_field.y) > 6);
            stable = moved ? 1 : stable + 1;
            last_field = view.arena.bottom;
            battle = stable >= 3;
        } else {
            stable = 0;
            last_field = cv::Rect();
        }
        if (i % 5 == 0) {
            std::cout << "    ждём бой: HUD " << (hud.in_battle(canvas) ? "да" : "нет") << ", поле "
                      << (view.battle ? "да" : "нет") << std::endl;
        }
        if (!battle) sleep_ms(1000);
    }
    if (!battle) {
        std::cerr << "бой не начался за " << wait_battle_s << " с" << std::endl;
        return 1;
    }
    std::cout << "бой идёт, кадр " << canvas.cols << "x" << canvas.rows << ", поле "
              << last_field.x << "," << last_field.y << " " << last_field.width << "x"
              << last_field.height << std::endl;
    cv::imwrite(out_dir + "/detect.png", canvas);

    if (summon_first && hud.summon_ready(canvas)) {
        // Призыв виджетом: дешёвый юнит на случайную клетку. Нужен, чтобы поле
        // не было пустым и бой шёл дольше (без юнитов база падает за ~20 с).
        game.click_canvas(hud.summon_button(canvas));
        std::cout << "призыв виджетом (случайная клетка)" << std::endl;
        sleep_ms(1500);
        canvas = game.grab_canvas();
    }

    // Энергия должна читаться: иначе непонятно, сколько стоит действие.
    double confidence = 0.0;
    double energy_before = hud.read_mana(canvas, &confidence);
    for (int i = 0; i < 20 && confidence < 0.5; ++i) {
        sleep_ms(1000);
        canvas = game.grab_canvas();
        energy_before = hud.read_mana(canvas, &confidence);
    }
    std::cout << "энергия до: " << energy_before << " (уверенность " << confidence << ")" << std::endl;
    print_slots(canvas, hud);
    cv::imwrite(out_dir + "/before.png", canvas);

    // Попытки жеста: (слот, клетка, сколько держать палец до движения, шагов).
    // Быстрый жест (как у слияния) и медленный с долгим нажатием — игра может
    // принимать перетаскивание карточки только при таком жесте.
    struct Attempt {
        int slot;
        int col;
        int row;
        int hold_ms;
        int steps;
    };
    if (move_test) {
        // Перенос своего юнита: тянем пешку на пустую клетку (как при слиянии).
        const FieldView view = field_view(canvas, hud);
        std::vector<rr::Pawn> pawns = view.battle ? rr::find_pawns(canvas, view.arena) : std::vector<rr::Pawn>{};
        const rr::Pawn* source = nullptr;
        for (const rr::Pawn& pawn : pawns) {
            if (pawn.ours) {
                source = &pawn;
                break;
            }
        }
        if (source == nullptr) {
            std::cout << "перенос: своих пешек на кадре нет" << std::endl;
            return 1;
        }
        const cv::Point from = source->center;
        // Пустая клетка подальше от пешки — перемещение видно хорошо.
        cv::Point target = from;
        double best = -1.0;
        for (int c = 0; c < 5; ++c) {
            for (int r = 0; r < 3; ++r) {
                const cv::Point candidate = cell_point(view.arena.bottom, c, r);
                bool busy = false;
                for (const rr::Pawn& pawn : pawns) {
                    if (std::abs(pawn.center.x - candidate.x) < 40 && std::abs(pawn.center.y - candidate.y) < 40) {
                        busy = true;
                    }
                }
                const double d = std::hypot(candidate.x - from.x, candidate.y - from.y);
                if (!busy && d > best) {
                    best = d;
                    target = candidate;
                }
            }
        }
        std::cout << "перенос: своя пешка (" << from.x << "," << from.y << ") -> пустая клетка ("
                  << target.x << "," << target.y << ")" << std::endl;
        cv::imwrite(out_dir + "/move_before.png", canvas);
        if (!game.drag_start_canvas(from)) {
            std::cerr << "мышь не работает" << std::endl;
            return 1;
        }
        for (int s = 1; s <= 5; ++s) {
            const double t = static_cast<double>(s) / 5.0;
            game.drag_move_canvas(cv::Point(static_cast<int>(from.x + (target.x - from.x) * t),
                                            static_cast<int>(from.y + (target.y - from.y) * t)));
            sleep_ms(40);
        }
        sleep_ms(200);
        game.drag_end_canvas(target);
        sleep_ms(2500);
        const cv::Mat moved = game.grab_canvas();
        if (!moved.empty()) {
            cv::imwrite(out_dir + "/move_after.png", moved);
            const rr::ArenaFrame moved_arena = rr::find_arena(moved);
            if (moved_arena.found) {
                bool at_target = false;
                bool at_source = false;
                for (const rr::Pawn& pawn : rr::find_pawns(moved, moved_arena)) {
                    if (!pawn.ours) continue;
                    if (std::abs(pawn.center.x - target.x) < 40 && std::abs(pawn.center.y - target.y) < 40) at_target = true;
                    if (std::abs(pawn.center.x - from.x) < 30 && std::abs(pawn.center.y - from.y) < 30) at_source = true;
                }
                std::cout << "перенос ИТОГ: пешка в цели=" << (at_target ? "да" : "нет")
                          << ", на старом месте=" << (at_source ? "да" : "нет") << std::endl;
            }
        }
        std::cout << "кадры: " << out_dir << std::endl;
        return 0;
    }
    std::vector<Attempt> attempts;
    if (select_test) {
        // Выбор нажатием: карточку — одиночным нажатием, затем свободную клетку.
        attempts.push_back({slot, col, row, -1, 0});
        attempts.push_back({slot == 1 ? 2 : slot, 3, 1, -1, 0});
    } else if (tap) {
        attempts.push_back({slot, 0, 0, 0, 0});
    } else {
        attempts.push_back({slot, col, row, 200, 6});
        attempts.push_back({slot == 1 ? 2 : slot, 1, 2, 800, 20});
        attempts.push_back({slot == 1 ? 3 : slot, 4, 2, 800, 20});
    }
    for (std::size_t n = 0; n < attempts.size(); ++n) {
        const Attempt& attempt = attempts[n];
        const std::string tag = "попытка " + std::to_string(n + 1);
        // Карточку ждём цветной (доступна), бой должен идти.
        const cv::Rect card = hud.deck_card_rect(attempt.slot);
        for (int i = 0; i < 40 && !hud.deck_slot_ready(canvas, attempt.slot); ++i) {
            sleep_ms(1000);
            canvas = game.grab_canvas();
            if (canvas.empty()) break;
        }
        const double card_before = hud.deck_slot_saturation(canvas, attempt.slot);
        const FieldView field = field_view(canvas, hud);
        if (!field.battle) {
            std::cout << tag << ": поле пропало — конец боя" << std::endl;
            break;
        }
        if (card_before < 58.0) {
            std::cout << tag << ": карточка " << attempt.slot << " не стала доступной" << std::endl;
            continue;
        }
        cv::Point target = cell_point(field.arena.bottom, attempt.col, attempt.row);
        const cv::Point from(card.x + card.width / 2, card.y + card.height / 2);
        const std::vector<rr::Pawn> pawns_before = rr::find_pawns(canvas, field.arena);
        int ours_before = 0;
        for (const rr::Pawn& pawn : pawns_before) {
            if (pawn.ours) ++ours_before;
        }
        double conf = 0.0;
        const double energy_before = hud.read_mana(canvas, &conf);
        std::cout << tag << ": карточка " << attempt.slot << " (sat " << static_cast<int>(card_before)
                  << ") из (" << from.x << "," << from.y << ") в клетку (" << attempt.col << ","
                  << attempt.row << ")=(" << target.x << "," << target.y << "), держу " << attempt.hold_ms
                  << " мс, шагов " << attempt.steps << "; своих " << ours_before << ", энергия "
                  << energy_before << std::endl;
        cv::imwrite(out_dir + "/" + std::to_string(n + 1) + "_before.png", canvas);
        if (attempt.hold_ms < 0) {
            // Выбор нажатием: карточка, затем клетка (если клетка занята — соседняя).
            game.click_canvas(from);
            sleep_ms(1200);
            const cv::Mat picked = game.grab_canvas();
            if (!picked.empty()) cv::imwrite(out_dir + "/" + std::to_string(n + 1) + "_picked.png", picked);
            std::cout << tag << ": нажал карточку " << attempt.slot << ", сохранил кадр выбора" << std::endl;
            game.click_canvas(target);
        } else if (tap) {
            game.click_canvas(from);
        } else {
            if (!game.drag_start_canvas(from)) {
                std::cerr << "мышь не работает (нет окна эмулятора?)" << std::endl;
                return 1;
            }
            sleep_ms(attempt.hold_ms);
            for (int s = 1; s <= attempt.steps; ++s) {
                const double t = static_cast<double>(s) / attempt.steps;
                game.drag_move_canvas(cv::Point(static_cast<int>(from.x + (target.x - from.x) * t),
                                                static_cast<int>(from.y + (target.y - from.y) * t)));
                sleep_ms(40);
            }
            // Кадр в середине жеста: видно, поднята ли карточка и куда она тянется.
            sleep_ms(300);
            const cv::Mat mid = game.grab_canvas();
            if (!mid.empty()) cv::imwrite(out_dir + "/" + std::to_string(n + 1) + "_mid.png", mid);
            game.drag_end_canvas(target);
        }
        sleep_ms(2500);
        const cv::Mat after = game.grab_canvas();
        if (after.empty()) {
            std::cout << tag << ": кадр после не получен" << std::endl;
            break;
        }
        cv::imwrite(out_dir + "/" + std::to_string(n + 1) + "_after.png", after);
        const rr::ArenaFrame after_arena = rr::find_arena(after);
        int ours_after = -1;
        if (after_arena.found) {
            ours_after = 0;
            for (const rr::Pawn& pawn : rr::find_pawns(after, after_arena)) {
                if (pawn.ours) ++ours_after;
            }
        }
        const double energy_after = hud.read_mana(after, &conf);
        std::cout << tag << " ИТОГ: своих " << ours_before << " -> " << ours_after << ", карточка "
                  << static_cast<int>(card_before) << " -> "
                  << static_cast<int>(hud.deck_slot_saturation(after, attempt.slot))
                  << ", энергия " << energy_before << " -> " << energy_after << ", HUD "
                  << (hud.in_battle(after) ? "да" : "нет") << std::endl;
        canvas = after;
    }
    std::cout << "кадры: " << out_dir << std::endl;
    return 0;
}
