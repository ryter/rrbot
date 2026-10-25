#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace rr {
namespace {

// Число с двумя знаками после запятой — для пояснений в логах.
std::string num(double value, int precision = 2) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << value;
    return out.str();
}

// Число со знаком: «+4.78», «-0.30» — так виден выигрыш от ожидания.
std::string num_signed(double value, int precision = 2) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.setf(std::ios::showpos);
    out.precision(precision);
    out << value;
    return out.str();
}

// Расстояние между точками.
double distance(double x1, double y1, double x2, double y2) { return std::hypot(x1 - x2, y1 - y2); }

}  // namespace

Planner::Planner(PlannerConfig weights, std::vector<UnitSpec> specs)
    : weights_(weights), specs_(std::move(specs)) {}

const UnitSpec* Planner::spec_of(const std::string& kind) const {
    for (const UnitSpec& spec : specs_) {
        if (spec.name == kind) return &spec;
    }
    return nullptr;
}

std::vector<LanePressure> Planner::lane_pressure(const GameState& state) const {
    std::vector<LanePressure> result;
    for (const auto& [name, lane] : state.spawn_points) {
        LanePressure pressure;
        pressure.lane = name;
        for (const Unit& enemy : state.enemy_units) {
            const double dist = distance(enemy.x, enemy.y, lane.x, lane.y);
            // Вклад падает с расстоянием: чем ближе к нашей базе, тем опаснее.
            const double weight = 1.0 / (1.0 + dist / 90.0);
            pressure.enemy_threat += enemy.dps() * weight * enemy.hp_fraction();
        }
        for (const Unit& ally : state.ally_units) {
            const double dist = distance(ally.x, ally.y, lane.x, lane.y);
            const double weight = 1.0 / (1.0 + dist / 90.0);
            pressure.ally_support += ally.dps() * weight * ally.hp_fraction();
        }
        result.push_back(pressure);
    }
    return result;
}

std::vector<std::string> Planner::roster(const GameState& state) const {
    std::vector<std::string> result;
    for (const UnitSpec& spec : specs_) {
        if (!state.available_known) {
            // Про карточки ничего не известно — доступны все типы.
            result.push_back(spec.name);
            continue;
        }
        if (std::find(state.available_units.begin(), state.available_units.end(), spec.name) !=
            state.available_units.end()) {
            result.push_back(spec.name);
        }
    }
    return result;
}

double Planner::card_cooldown(const GameState& state, const std::string& kind) const {
    const auto it = state.card_cooldowns.find(kind);
    if (it == state.card_cooldowns.end()) return 0.0;
    return std::max(0.0, it->second);
}

bool Planner::is_ready(const GameState& state, const std::string& kind) const {
    if (card_cooldown(state, kind) > 0.0) return false;
    if (state.available_known &&
        std::find(state.available_units.begin(), state.available_units.end(), kind) ==
            state.available_units.end()) {
        return false;
    }
    return true;
}

double Planner::cost_efficiency(const UnitSpec& spec) const {
    if (weights_.cost_metric == CostMetric::HpDps) {
        return spec.value_per_cost() / kValueScale;
    }
    return spec.hp_per_cost();
}

double Planner::score_unit(const std::string& kind, const GameState& state, const Lane* lane,
                           bool assume_available, std::map<std::string, double>* parts) const {
    std::map<std::string, double> local;
    std::map<std::string, double>& out = parts != nullptr ? *parts : local;

    const UnitSpec* spec = spec_of(kind);
    if (spec == nullptr) {
        out.clear();
        return 0.0;
    }

    // 1. Контр-счётство: какую долю вражеского урона мы перебиваем.
    // Считаем долю, а не сумму: иначе слагаемое зависит от числа врагов
    // на поле и перебивает все остальные слагаемые.
    double counter_value = 0.0;
    double total_threat = 0.0;
    for (const Unit& enemy : state.enemy_units) {
        const double threat = enemy.hp_fraction() * enemy.dps();
        total_threat += threat;
        const bool countered = std::find(spec->counters.begin(), spec->counters.end(), enemy.kind) !=
                               spec->counters.end();
        if (countered) counter_value += threat;
    }
    const double share = total_threat > 0.0 ? counter_value / total_threat : 0.0;
    out["counter"] = weights_.weight_counter * share;

    // 2. Давление на линии: спавн туда, где враг сильнее.
    // Слагаемое получается одинаковым для всех типов, поэтому на выбор
    // юнита оно не влияет — только показывает, насколько важна линия.
    double pressure = 0.0;
    if (lane != nullptr) {
        for (const Unit& enemy : state.enemy_units) {
            const double dist = distance(enemy.x, enemy.y, lane->x, lane->y);
            pressure += enemy.dps() * enemy.hp_fraction() / (1.0 + dist / 120.0);
        }
    }
    out["pressure"] = weights_.weight_pressure * pressure;

    // 3. Синергия: не забивать линию одним типом.
    int same = 0;
    for (const Unit& ally : state.ally_units) {
        if (ally.kind == kind) ++same;
    }
    const double total_ally = std::max(1.0, static_cast<double>(state.ally_units.size()));
    out["synergy"] = weights_.weight_synergy * (1.0 - 2.0 * same / total_ally);

    // 4. Эффективность по мане: отдача юнита на единицу маны (по метрике
    // из конфига). Метрики нормированы к одному порядку величины, поэтому
    // вес weight_cost_efficiency значит одно и то же при любой из них.
    out["cost"] = weights_.weight_cost_efficiency * cost_efficiency(*spec);

    // 5. Бонус за то, что юнит вообще доступен для спавна.
    // Если про карточки ничего не известно — считаем, что доступны все.
    const bool is_available =
        assume_available || !state.available_known ||
        std::find(state.available_units.begin(), state.available_units.end(), kind) !=
            state.available_units.end();
    out["available"] = is_available ? weights_.weight_spawn_value : 0.0;

    double total = 0.0;
    for (const auto& [name, value] : out) {
        (void)name;
        total += value;
    }
    return total;
}

double Planner::lane_threat(const GameState& state, const Lane& lane) const {
    double threat = 0.0;
    for (const Unit& enemy : state.enemy_units) {
        const double dist = distance(enemy.x, enemy.y, lane.x, lane.y);
        threat += enemy.dps() * enemy.hp_fraction() / (1.0 + dist / 90.0);
    }
    return threat;
}

const Lane* Planner::guard_lane(const GameState& state) const {
    // Враг прорвался в нашу половину — защищаем его линию.
    //
    // Волны врага идут по одной линии и числом превосходят наши
    // подкрепления: юниты, отправленные в ту же линию, гибнут в стычке и
    // не доходят до базы врага. Поэтому, пока враг держится в своей
    // половине, бот не лезет в его линию, а толкает в другую (контр-пуш);
    // как только враг перешёл середину поля — защищаем его линию.
    const double half = weights_.defend_line_height * 0.5;
    if (half <= 0.0) return nullptr;
    const Unit* closest = nullptr;
    for (const Unit& enemy : state.enemy_units) {
        if (enemy.y <= half) continue;
        if (closest == nullptr || enemy.y > closest->y) closest = &enemy;
    }
    if (closest == nullptr) return nullptr;
    const Lane* best = nullptr;
    double best_distance = 0.0;
    for (const auto& [name, lane] : state.spawn_points) {
        (void)name;
        // Линии спавна лежат на одной высоте, поэтому линию врага узнаём по
        // горизонтали.
        const double d = std::abs(closest->x - lane.x);
        if (best == nullptr || d < best_distance) {
            best = &lane;
            best_distance = d;
        }
    }
    return best;
}

const Lane* Planner::pick_lane(const GameState& state,
                                const std::vector<LanePressure>& pressures) const {
    if (pressures.empty()) return nullptr;
    std::vector<const Lane*> lanes;
    for (const auto& [name, lane] : state.spawn_points) {
        (void)name;
        lanes.push_back(&lane);
    }
    if (lanes.empty()) return nullptr;

    // 1. Враг прорвался в нашу половину — защищаем его линию.
    const Lane* guard = guard_lane(state);
    if (guard != nullptr) return guard;

    // 2. Иначе толкаем в линию, где врага меньше (контр-пуш).
    double least = std::numeric_limits<double>::infinity();
    for (const Lane* lane : lanes) least = std::min(least, lane_threat(state, *lane));
    std::vector<const Lane*> candidates;
    for (const Lane* lane : lanes) {
        if (lane_threat(state, *lane) <= least + 1e-9) candidates.push_back(lane);
    }
    const Lane* best = candidates.front();
    if (candidates.size() > 1) {
        // 3. При равной угрозе берём линию, где меньше наших юнитов: не
        // забиваем одну линию. Считаем по горизонтали — линии спавна лежат
        // на одной высоте, и по вертикали каждый юнит попадал бы сразу в
        // обе линии.
        std::map<std::string, int> counts;
        for (const auto& [name, lane] : state.spawn_points) {
            (void)lane;
            counts[name] = 0;
        }
        for (const Unit& ally : state.ally_units) {
            const Lane* nearest = nullptr;
            double nearest_distance = 0.0;
            for (const Lane* lane : lanes) {
                const double d = std::abs(ally.x - lane->x);
                if (nearest == nullptr || d < nearest_distance) {
                    nearest = lane;
                    nearest_distance = d;
                }
            }
            if (nearest != nullptr) ++counts[nearest->name];
        }
        int least_count = std::numeric_limits<int>::max();
        for (const Lane* lane : candidates) {
            const int count = counts[lane->name];
            if (count < least_count) {
                least_count = count;
                best = lane;
            }
        }
    }
    return best;
}

std::string Planner::explain(const std::string& kind, const std::map<std::string, double>& parts,
                              const Lane* lane) {
    std::vector<std::pair<std::string, double>> sorted;
    for (const auto& [name, value] : parts) sorted.emplace_back(name, value);
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return std::abs(a.second) > std::abs(b.second);
    });
    std::string contrib;
    for (size_t i = 0; i < sorted.size() && i < 2; ++i) {
        if (!contrib.empty()) contrib += ", ";
        contrib += sorted[i].first + "=" + num_signed(sorted[i].second);
    }
    std::string where;
    if (lane != nullptr) where = " на линию «" + lane->name + "»";
    return kind + where + " (" + contrib + ")";
}

Action Planner::decide(const GameState& state) const {
    // Ростер: юниты, о которых зрение сообщило как о доступных. Если
    // информации нет, считаем доступными всех.
    const std::vector<std::string> roster = this->roster(state);
    if (roster.empty()) {
        // Карточки известны и не горят ни одна — звать некого.
        Action action;
        action.kind = "wait";
        action.reason = "нет доступных юнитов";
        return action;
    }

    // Кого можно заспавнить прямо сейчас: хватает маны.
    const double usable = state.mana - weights_.reserve_mana;
    std::vector<std::string> spawnable;
    for (const std::string& kind : roster) {
        const UnitSpec* spec = spec_of(kind);
        if (spec == nullptr) continue;
        if (spec->cost <= usable) spawnable.push_back(kind);
    }
    if (spawnable.empty()) {
        double need = 0.0;
        bool first = true;
        for (const std::string& kind : roster) {
            const UnitSpec* spec = spec_of(kind);
            if (spec == nullptr) continue;
            if (first || spec->cost < need) {
                need = spec->cost;
                first = false;
            }
        }
        Action action;
        action.kind = "wait";
        action.reason = "мана " + num(state.mana, 0) + " < стоимости " + num(need, 0);
        if (weights_.reserve_mana > 0.0) {
            action.reason += " + резерва " + num(weights_.reserve_mana, 0);
        }
        return action;
    }

    const std::vector<LanePressure> pressures = lane_pressure(state);
    const Lane* lane = pick_lane(state, pressures);

    std::string best_kind;
    double best_score = -std::numeric_limits<double>::infinity();
    std::map<std::string, double> best_parts;
    for (const std::string& kind : spawnable) {
        std::map<std::string, double> parts;
        const double score = score_unit(kind, state, lane, false, &parts);
        if (score > best_score) {
            best_score = score;
            best_kind = kind;
            best_parts = parts;
        }
    }
    if (best_kind.empty()) {
        Action action;
        action.kind = "wait";
        action.reason = "нет кандидатов";
        return action;
    }
    if (best_score < 0.0) {
        Action action;
        action.kind = "wait";
        action.reason = "нет выгодного спавна (оценка " + num(best_score) + ")";
        action.score = best_score;
        action.scores = best_parts;
        return action;
    }

    Action action;
    action.kind = "spawn";
    action.unit_kind = best_kind;
    action.score = best_score;
    action.scores = best_parts;
    action.reason = explain(best_kind, best_parts, lane);
    if (lane != nullptr) {
        action.lane = lane->name;
        action.has_position = true;
        action.position_x = lane->x;
        action.position_y = lane->y;
    }
    return action;
}

Action Planner::should_save_mana(const GameState& state, const std::string& best_kind) const {
    Action none;
    none.kind = "none";
    if (!weights_.save_mana_enabled) return none;
    if (spec_of(best_kind) == nullptr) return none;

    const std::vector<LanePressure> pressures = lane_pressure(state);
    const Lane* lane = pick_lane(state, pressures);
    const double current_score = score_unit(best_kind, state, lane);

    const double usable = state.mana - weights_.reserve_mana;
    const double regen = state.mana_regen > 0.0 ? state.mana_regen : 0.0;

    // Ждать бывает по двум причинам: карточка юнита ещё горит (не закончилась
    // перезарядка) или маны пока не хватает. Время ожидания — в секундах:
    // max(время до готовности карточки, время накопления маны).
    std::string best_kind_wait;
    double best_gain = 0.0;
    double best_deficit = 0.0;
    double best_card = 0.0;
    std::map<std::string, double> best_parts;
    for (const UnitSpec& spec : specs_) {
        const bool ready = is_ready(state, spec.name);
        const double deficit = spec.cost - usable;
        if (ready && deficit <= 0.0) continue;  // доступен прямо сейчас
        const double card_left = card_cooldown(state, spec.name);
        const double mana_wait = (deficit > 0.0 && regen > 0.0) ? deficit / regen : 0.0;
        const double wait = std::max(card_left, mana_wait);
        if (wait > weights_.save_mana_max_wait) continue;
        std::map<std::string, double> parts;
        const double future_score = score_unit(spec.name, state, lane, true, &parts);
        const double gain = future_score - current_score;
        if (gain < weights_.save_mana_min_gain) continue;
        // Из кандидатов выбираем тот, у которого выигрыш больше, а не первый
        // попавшийся: иначе порядок типов в конфиге определял бы решение.
        if (best_kind_wait.empty() || gain > best_gain) {
            best_gain = gain;
            best_kind_wait = spec.name;
            best_parts = parts;
            best_deficit = std::max(0.0, deficit);
            best_card = card_left;
        }
    }
    if (best_kind_wait.empty()) return none;

    Action action;
    action.kind = "wait";
    action.score = current_score;
    action.scores = best_parts;
    std::string reason;
    if (best_deficit > 0.0) {
        reason = "копим ману на " + best_kind_wait + " (выигрыш " + num_signed(best_gain) +
                 ", не хватает " + num(best_deficit, 0) + ")";
    } else {
        reason = "ждём карточку " + best_kind_wait + " (выигрыш " + num_signed(best_gain) + ")";
    }
    if (best_card > 0.0) {
        reason += ", карточка горит ещё " + num(best_card, 0) + " с";
    }
    action.reason = reason;
    return action;
}

Action Planner::plan(const GameState& state) const {
    Action action = decide(state);
    if (action.kind != "spawn") return action;
    const Action wait = should_save_mana(state, action.unit_kind);
    if (wait.kind == "wait") return wait;
    return action;
}

Action Planner::suggest_focus(const GameState& state) const {
    Action action;
    if (state.enemy_units.empty()) {
        action.kind = "wait";
        action.reason = "нет целей";
        return action;
    }
    double cx = 0.0;
    double cy = 0.0;
    if (!state.ally_units.empty()) {
        for (const Unit& ally : state.ally_units) {
            cx += ally.x;
            cy += ally.y;
        }
        cx /= static_cast<double>(state.ally_units.size());
        cy /= static_cast<double>(state.ally_units.size());
    }
    const Unit* target = &state.enemy_units.front();
    double best = std::numeric_limits<double>::infinity();
    for (const Unit& enemy : state.enemy_units) {
        const double value = distance(enemy.x, enemy.y, cx, cy) - 40.0 * enemy.hp_fraction();
        if (value < best) {
            best = value;
            target = &enemy;
        }
    }
    action.kind = "focus";
    action.target_uid = target->uid;
    action.reason = "близкая цель " + target->kind + " (hp " + num(target->hp_fraction() * 100.0, 0) + "%)";
    return action;
}

}  // namespace rr
