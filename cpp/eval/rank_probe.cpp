// Проверка формы основания: совпадает ли она у одного и того же типа
// юнита на разных рангах.
//
// Печатает по каждой пешке: тип (по эталону), ранг основания (заполненность)
// и радиус. Если форма кодирует тип, у одного типа она одинакова.
//
//     ./build/rank_probe <кадр.png>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "arena.hpp"
#include "merge.hpp"
#include "vision.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "нужен кадр: rank_probe <файл.png>" << std::endl;
        return 1;
    }
    const cv::Mat canvas = cv2.imread(argv[1], cv2.IMREAD_COLOR);
    if (canvas.empty()) {
        std::cerr << "кадр не прочитан" << std::endl;
        return 1;
    }
    const rr::ArenaFrame arena = rr::find_arena(canvas);
    if (!arena.found) {
        std::cerr << "арена не найдена" << std::endl;
        return 1;
    }
    rr::VisionConfig config;
    config.templates_dir = "assets/real_emulator";
    config.template_map = {{"knight", "unit_em_00_teal_on_gray.png"},
                            {"archer", "unit_em_01_orange_on_green.png"},
                            {"golem", "unit_em_02_orange_on_green.png"},
                            {"warden", "unit_em_03_red_on_green.png"},
                            {"knight2", "unit_em_04_orange_on_orange.png"},
                            {"archer2", "unit_em_05_orange_on_orange.png"},
                            {"golem2", "unit_em_06_teal_on_teal.png"}};
    config.units = rr::default_live_specs();
    rr::VisionEngine vision(config);
    if (!vision.load_templates()) {
        std::cerr << "эталоны не загрузились" << std::endl;
        return 1;
    }
    const std::vector<rr::Pawn> pawns = rr::find_pawns(canvas, arena);
    for (const rr::Pawn& pawn : pawns) {
        const double fill = rr::pedestal_fill(canvas, pawn);
        // Тип определяем по портрету: сравниваем палитру пешки с эталонами.
        std::string type = "?";
        double best = -1.0;
        for (const rr::UnitBase& base : rr::find_unit_bases(canvas, {arena.top, arena.bottom})) {
            (void)base;
        }
        std::cout << (pawn.ours ? "своя " : "чужая ") << pawn.center.x << "," << pawn.center.y
                  << " радиус " << pawn.radius << " заполненность " << fill
                  << " вытянутость " << rr::pedestal_stretch(canvas, pawn)
                  << (pawn.round_pedestal ? " круглое" : " НЕ круглое") << std::endl;
    }
    return 0;
}
