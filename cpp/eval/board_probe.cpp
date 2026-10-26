// Проверочная программа: сколько клеток доски занято.
//
// Печатает по каждому кадру: найденную арену, пешки, оценку сетки и
// занятость доски (сколько клеток высадки занято и сколько свободно).
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "arena.hpp"
#include "board.hpp"
#include "merge.hpp"

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string path = argv[i];
        const cv::Mat canvas = cv::imread(path, cv::IMREAD_COLOR);
        if (canvas.empty()) {
            std::cout << path << ": кадр не прочитан" << std::endl;
            continue;
        }
        const rr::ArenaFrame arena = rr::find_arena(canvas);
        const std::vector<rr::Pawn> pawns = rr::find_pawns(canvas, arena);
        const rr::BoardState board = rr::analyze_board(canvas, arena, pawns);
        std::cout << path << ": арена ";
        if (!arena.found) {
            std::cout << "не найдена" << std::endl;
            continue;
        }
        std::cout << "своя " << arena.bottom.x << "," << arena.bottom.y << " "
                  << arena.bottom.width << "x" << arena.bottom.height;
        int ours = 0;
        for (const rr::Pawn& pawn : pawns) {
            if (pawn.ours) ++ours;
        }
        std::cout << ", своих пешек " << ours;
        if (!board.valid) {
            std::cout << ", доска не оценена" << std::endl;
            continue;
        }
        std::cout << ", сетка: шаг " << board.grid.spacing << ", начало ("
                  << board.grid.phase_x << "," << board.grid.phase_y << "), подтвердили "
                  << board.grid.inliers << " из " << board.grid.considered;
        std::cout << ", занято " << board.occupied << "/" << board.capacity
                  << ", свободно " << board.free_cells
                  << ", клеток " << board.cells.size() << std::endl;
        if (!board.cell_busy.empty()) {
            std::cout << "   пестрота клеток:";
            for (const double busy : board.cell_busy) {
                std::cout << " " << static_cast<int>(busy);
            }
            std::cout << std::endl;
        }
    }
    return 0;
}
