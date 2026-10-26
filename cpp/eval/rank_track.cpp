// Самопроверка трекера ранга: призыв = 1, слияние на месте цели = 2.
//
//     ./build/rank_track
#include <iostream>

#include "merge.hpp"

int main() {
    rr::RankTracker tracker;
    std::vector<rr::Pawn> pawns(2);
    pawns[0].ours = true;
    pawns[0].center = {100, 200};
    pawns[0].radius = 20;
    pawns[1].ours = true;
    pawns[1].center = {180, 200};
    pawns[1].radius = 20;
    auto ranks = tracker.update(pawns);
    if (ranks.size() != 2 || ranks[0] != 1 || ranks[1] != 1) {
        std::cerr << "ожидал ранг 1 у обеих новых пешек\n";
        return 1;
    }
    tracker.promote({180, 200}, 2);
    ranks = tracker.update(pawns);
    if (ranks[1] != 2) {
        std::cerr << "ожидал ранг 2 на месте цели, получил " << ranks[1] << "\n";
        return 1;
    }
    pawns[1].center = {185, 270};
    ranks = tracker.update(pawns);
    if (ranks[1] != 2) {
        std::cerr << "ранг должен ехать за пешкой, получил " << ranks[1] << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
