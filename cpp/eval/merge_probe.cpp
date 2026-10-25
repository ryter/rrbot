// Проверка поиска пар для слияния: на сохранённом кадре боя показывает
// пешки на поле и пары, которые бот стал бы сливать.
//
//     ./build/merge_probe /tmp/opencode/battle_loop/b7_20.png
//
// Печатает пешки (кадр), пары-кандидаты с оценкой похожести и — если
// задан порог — только те пары, что бот счёл бы слиянием.
//
// Режим --csv: данные для разбора правил выбора пары (tools/
// merge_policy_offline.py). Первая строка — служебная («our_pawns;N» —
// сколько своих пешек на кадре), дальше по строке на пару, разделитель «;».
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "arena.hpp"
#include "merge.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "нужен кадр: merge_probe <файл.png>" << std::endl;
        return 1;
    }
    const std::string path = argv[1];
    const double min_score = argc > 2 ? std::atof(argv[2]) : rr::kMergeMatchMin;
    bool csv = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--csv") csv = true;
    }
    const cv::Mat canvas = cv::imread(path, cv::IMREAD_COLOR);
    if (canvas.empty()) {
        std::cerr << "кадр не прочитан: " << path << std::endl;
        return 1;
    }
    const rr::ArenaFrame arena = rr::find_arena(canvas);
    if (!arena.found) {
        std::cerr << "арена не найдена" << std::endl;
        return 1;
    }
    if (!csv) {
        std::cout << "арена: своя " << arena.bottom.x << "," << arena.bottom.y << " "
                  << arena.bottom.width << "x" << arena.bottom.height << "; чужая " << arena.top.x
                  << "," << arena.top.y << " " << arena.top.width << "x" << arena.top.height
                  << std::endl;
    }
    if (argc > 3 && std::string(argv[3]) == "--dungeon") {
        // Как выглядит поле в «Подземельях»: комнаты по кремовым стенам.
        const rr::ArenaFrame rooms = rr::find_arena_dungeon(canvas);
        std::cout << "комнаты: " << (rooms.found ? "найдены" : "не найдены");
        if (rooms.found) {
            std::cout << ": своя " << rooms.bottom.x << "," << rooms.bottom.y << " "
                      << rooms.bottom.width << "x" << rooms.bottom.height << "; чужая " << rooms.top.x
                      << "," << rooms.top.y << " " << rooms.top.width << "x" << rooms.top.height;
        }
        std::cout << std::endl;
    }
    const std::vector<rr::Pawn> pawns = rr::find_pawns(canvas, arena);
    int ours = 0;
    for (const rr::Pawn& pawn : pawns) {
        if (pawn.ours) ++ours;
    }
    if (csv) {
        // Служебная строка: сколько своих пешек на кадре (бот по этому
        // признаку решает, забита ли доска).
        std::cout << "our_pawns;" << ours << std::endl;
    } else {
        for (const rr::Pawn& pawn : pawns) {
            std::cout << (pawn.ours ? "  своя " : "  чужая ") << pawn.center.x << "," << pawn.center.y
                      << " радиус " << pawn.radius
                      << (pawn.round_pedestal ? " круглое" : " НЕ круглое")
                      << " заполненность " << rr::pedestal_fill(canvas, pawn) << std::endl;
        }
        std::cout << "пешек всего " << pawns.size() << ", своих " << ours << std::endl;
    }

    const std::vector<rr::MergePair> pairs = rr::find_merge_pairs(canvas, pawns);
    if (!csv) {
        std::cout << "пар-кандидатов: " << pairs.size() << std::endl;
    }
    for (const rr::MergePair& pair : pairs) {
        const rr::Pawn& source = pawns[pair.source];
        const rr::Pawn& target = pawns[pair.target];
        if (csv) {
            std::cout << pair.source << ";" << pair.target << ";" << source.center.x << ";"
                      << source.center.y << ";" << target.center.x << ";" << target.center.y << ";"
                      << pair.score << ";" << pair.distance << ";" << pair.size_diff << ";"
                      << pair.rank_source << ";" << pair.rank_target << ";"
                      << (pair.same_rank ? 1 : 0)
                      << ";" << (pair.adjacent ? 1 : 0) << ";"
                      << rr::merge_success_estimate(pair) << std::endl;
            continue;
        }
        std::cout << "  [" << pair.source << "->" << pair.target << "] (" << source.center.x << ","
                  << source.center.y << ") -> (" << target.center.x << "," << target.center.y
                  << "): похожесть " << pair.score << ", расстояние " << pair.distance
                  << ", разница размеров " << static_cast<int>(pair.size_diff * 100)
                  << " %, ранги " << pair.rank_source << "/" << pair.rank_target
                  << ", оценка успеха "
                  << static_cast<int>(rr::merge_success_estimate(pair) * 100) << " %"
                  << (pair.adjacent ? ", соседние" : "")
                  << (pair.score >= min_score ? "  СЛИВАЕМ" : "") << std::endl;
    }
    return 0;
}
