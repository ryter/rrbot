// Проверочная программа: печатает найденные на кадре половины арены.
//
// Используется тестом tests/test_cpp_arena.py и при отладке разбора поля:
//     ./build/arena_probe assets/eval_real/frames/*.png
//
// Формат вывода на кадр:
//     верх X,Y WxH (плиток N), низ X,Y WxH (плиток N), вместе X,Y WxH
//
// С ключом --board <каталог> ещё сохраняется канонический кадр (400x522) —
// так можно сравнить его с тем, что делает tools/prepare_real_frames.py.
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

#include "arena.hpp"

int main(int argc, char** argv) {
    std::string board_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--board" && i + 1 < argc) {
            board_dir = argv[++i];
            continue;
        }
        const cv::Mat image = cv::imread(arg, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::printf("%s: не прочитан\n", arg.c_str());
            continue;
        }
        const rr::ArenaFrame arena = rr::find_arena(image);
        std::printf("%s (%dx%d): ", arg.c_str(), image.cols, image.rows);
        if (!arena.found) {
            std::printf("не найдена\n");
            continue;
        }
        std::printf(
            "верх %d,%d %dx%d (плиток %d), низ %d,%d %dx%d (плиток %d), вместе %d,%d %dx%d\n",
            arena.top.x, arena.top.y, arena.top.width, arena.top.height, arena.tiles_top,
            arena.bottom.x, arena.bottom.y, arena.bottom.width, arena.bottom.height,
            arena.tiles_bottom, arena.full.x, arena.full.y, arena.full.width, arena.full.height);
        if (!board_dir.empty()) {
            const cv::Mat board = rr::canonical_board(image);
            if (!board.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(board_dir, ec);
                const std::string name = std::filesystem::path(arg).stem().string() + "_board.png";
                cv::imwrite(board_dir + "/" + name, board);
                std::printf("    канонический кадр: %s (%dx%d)\n", name.c_str(), board.cols,
                            board.rows);
            }
        }
    }
    return 0;
}
