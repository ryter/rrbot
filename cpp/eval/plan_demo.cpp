// Демонстрация планировщика C++: печатает решение по нескольким сценариям.
//
// Нужна для проверки порта rrbot/planner.py: тест запускает эту утилиту и
// смотрит, что решение ожидаемое.
//
// Запуск:
//     ./build/rrplan [--scenario A|B|C]

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "models.hpp"
#include "planner.hpp"
#include "vision.hpp"

namespace {

// Состояние сценария: мана, доступные юниты и юниты на поле.
struct Scenario {
    std::string name;
    double mana = 500.0;
    std::vector<std::string> available;
    // Юниты: тип, сторона, x, y, доля здоровья.
    std::vector<std::tuple<std::string, rr::Team, double, double, double>> units;
};

// Линии высадки: как в симуляторе, левая и правая.
void add_lanes(rr::GameState* state) {
    rr::Lane left;
    left.name = "left";
    left.x = 90.0;
    left.y = 560.0;
    rr::Lane right;
    right.name = "right";
    right.x = 270.0;
    right.y = 560.0;
    state->spawn_points["left"] = left;
    state->spawn_points["right"] = right;
}

// Характеристики типов из config/default.yaml (у них разная цена —
// как в симуляторе).
std::vector<rr::UnitSpec> sim_specs() {
    std::vector<rr::UnitSpec> specs;
    const auto add = [&specs](const char* name, double cost, double hp, double damage,
                               double interval, double radius, const char* family,
                               const char* counter) {
        rr::UnitSpec spec;
        spec.name = name;
        spec.cost = cost;
        spec.hp = hp;
        spec.damage = damage;
        spec.attack_interval = interval;
        spec.radius = radius;
        spec.family = family;
        spec.counters = {counter};
        specs.push_back(spec);
    };
    add("knight", 70, 220, 28, 0.9, 9, "melee", "warden");
    add("archer", 90, 130, 34, 1.2, 8, "ranged", "samurai");
    add("golem", 120, 520, 22, 1.4, 11, "tank", "knight");
    add("bomber", 100, 180, 95, 2.4, 9, "splash", "archer");
    add("samurai", 110, 260, 46, 1.1, 9, "melee", "golem");
    add("warden", 140, 600, 30, 1.3, 12, "tank", "bomber");
    return specs;
}

}  // namespace

int main(int argc, char** argv) {
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scenario" && i + 1 < argc) only = argv[++i];
    }

    // Сценарии повторяют проверки из tests/test_core.py (Python-версия), поэтому
    // характеристики юнитов — из config/default.yaml (у них разная цена).
    const rr::PlannerConfig weights;
    const std::vector<rr::UnitSpec> specs = sim_specs();
    rr::Planner planner(weights, specs);
    std::vector<Scenario> scenarios = {
        // A: враг-warden слева, наш knight рядом. Knight перебивает warden —
        //    планировщик должен выбрать его.
        {"A", 500.0, {"knight", "archer", "golem", "samurai", "warden"},
         {{"warden", rr::Team::Enemy, 100.0, 300.0, 1.0},
          {"knight", rr::Team::Ally, 120.0, 500.0, 1.0}}},
        // B: маны нет — ждём.
        {"B", 0.0, {"knight", "archer", "golem", "samurai", "warden"},
         {{"warden", rr::Team::Enemy, 100.0, 300.0, 1.0}}},
        // C: доступен только knight, маны на всех не хватает — копим под тем,
        //    кто выгоднее по отдаче на единицу маны.
        {"C", 100.0, {"knight"},
         {{"knight", rr::Team::Ally, 120.0, 500.0, 1.0}}},
        // E: враг прорвался в нашу половину по левой линии — защищаем её.
        {"E", 500.0, {"knight", "archer", "golem", "samurai", "warden"},
         {{"samurai", rr::Team::Enemy, 100.0, 400.0, 1.0}}},
    };

    for (const Scenario& scenario : scenarios) {
        if (!only.empty() && scenario.name != only) continue;
        rr::GameState state;
        state.mana = scenario.mana;
        state.available_units = scenario.available;
        // Список карточек в сценарии задан точно: планировщик знает, кто готов.
        state.available_known = true;
        add_lanes(&state);
        int uid = 1;
        for (const auto& [kind, team, x, y, hp_fraction] : scenario.units) {
            rr::Unit unit;
            unit.uid = uid++;
            unit.kind = kind;
            unit.team = team;
            unit.x = x;
            unit.y = y;
            for (const rr::UnitSpec& spec : specs) {
                if (spec.name == kind) {
                    unit.spec = &spec;
                    unit.hp = spec.hp * hp_fraction;
                    unit.max_hp = spec.hp;
                    break;
                }
            }
            if (team == rr::Team::Ally) {
                state.ally_units.push_back(unit);
            } else {
                state.enemy_units.push_back(unit);
            }
        }

        const rr::Action action = planner.decide(state);
        std::cout << "СЦЕНАРИЙ " << scenario.name << " | мана=" << scenario.mana
                  << " | решение=" << action.kind;
        if (action.kind == "spawn") {
            std::cout << " юнит=" << action.unit_kind << " линия=" << action.lane;
        }
        std::cout << " оценка=" << action.score << " причина=\"" << action.reason << "\"" << std::endl;

        // Оценки всех типов — видно, почему выбран именно этот.
        std::cout << "ОЦЕНКИ";
        const rr::Lane* lane = nullptr;
        if (!state.spawn_points.empty()) lane = &state.spawn_points.begin()->second;
        for (const rr::UnitSpec& spec : specs) {
            const double score = planner.score_unit(spec.name, state, lane);
            std::cout << " " << spec.name << "=" << score;
        }
        std::cout << std::endl;

        // Накопление маны: отдельно, чтобы решение не зависело от него.
        rr::Planner saving(weights, specs);
        rr::Action wait = saving.should_save_mana(state, "knight");
        std::cout << "НАКОПЛЕНИЕ " << (wait.kind == "wait" ? "да" : "нет") << " | " << wait.reason
                  << std::endl;
    }

    // Сценарий D: проверка накопления маны. Здесь типы из config/default.yaml
    // (у них разная цена) и механизм накопления включён.
    //
    // Готов только стрелок: он дешевле, но по отдаче на единицу маны хуже
    // рыцаря, самурая и вардена — под кого-то из них мана и копится.
    // Сценарий повторяет тест test_нет_подходящего_юнита_копим_ману
    // из tests/test_core.py — числа должны совпадать.
    {
        rr::PlannerConfig saving_weights = weights;
        saving_weights.save_mana_enabled = true;
        saving_weights.weight_cost_efficiency = 3.0;
        const std::vector<rr::UnitSpec> specs = sim_specs();
        rr::Planner planner(saving_weights, specs);
        rr::GameState state;
        state.mana = 90.0;
        state.available_units = {"archer"};
        state.available_known = true;
        add_lanes(&state);
        const rr::Action action = planner.decide(state);
        const rr::Action wait = planner.should_save_mana(state, action.unit_kind);
        std::cout << "СЦЕНАРИЙ D | мана=90 | решение=" << action.kind
                  << " юнит=" << action.unit_kind << " оценка=" << action.score << " причина=\""
                  << action.reason << "\"" << std::endl;
        std::cout << "ОЦЕНКИ D";
        const rr::Lane* lane = &state.spawn_points.begin()->second;
        for (const rr::UnitSpec& spec : specs) {
            const double score = planner.score_unit(spec.name, state, lane);
            std::cout << " " << spec.name << "=" << score;
        }
        std::cout << std::endl;
        std::cout << "НАКОПЛЕНИЕ " << (wait.kind == "wait" ? "да" : "нет") << " | " << wait.reason
                  << std::endl;
    }
    return 0;
}
