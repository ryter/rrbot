// Оценка вероятности успеха слияния — для тестов и разбора замеров.
//
// Печатает оценку для пары с заданными параметрами:
//     merge_estimate <похожесть> <один_ранг 0|1> <разница_размеров>
//
// Пример:
//     merge_estimate 0.93 1 0.05   -> 0.92
#include <cstdlib>
#include <iostream>

#include "merge.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "usage: merge_estimate <score> <same_rank 0|1> <size_diff>\n";
        return 1;
    }
    rr::MergePair pair;
    pair.score = std::atof(argv[1]);
    pair.same_rank = std::atoi(argv[2]) != 0;
    pair.size_diff = std::atof(argv[3]);
    std::cout << rr::merge_success_estimate(pair) << "\n";
    return 0;
}
