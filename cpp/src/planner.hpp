// Планировщик: оценка полезности спавна и выбор цели.
//
// Порт rrbot/planner.py на C++. Логика прозрачна и легко тюнится: каждая
// составляющая полезности — отдельный вес, поэтому наборы весов можно
// сравнивать замерами.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "models.hpp"

namespace rr {

// Метрика эффективности по мане: сколько юнит даёт на единицу потраченной
// маны. Выбирается замером в симуляторе (tools/compare_cost_metric.py).
enum class CostMetric {
    // Живучесть на единицу цены: hp / цена.
    Hp,
    // Живучесть × урон на единицу цены, нормированные так же, как Hp.
    HpDps,
};

// Шкала нормировки метрики HpDps: значения сопоставимы с Hp, поэтому вес
// weight_cost_efficiency значит одно и то же при любой метрике.
constexpr double kValueScale = 25.0;

// Веса полезности юнита.
struct PlannerConfig {
    // Перебиваем ли мы врагов этого типа.
    double weight_counter = 3.0;
    // Насколько велик враг у выбранной линии.
    double weight_pressure = 2.5;
    // Не забиваем ли линию одним типом.
    double weight_synergy = 1.2;
    // Эффективность по мане (по метрике cost_metric).
    double weight_cost_efficiency = 1.0;
    // Метрика эффективности по мане.
    // По умолчанию hp_dps: так выиграла замер в симуляторе
    // (tools/compare_cost_metric.py) — 36.2% побед против 31.0% у hp.
    CostMetric cost_metric = CostMetric::HpDps;
    // Бонус юниту, который вообще доступен к спавну.
    double weight_spawn_value = 0.6;
    // Мана, которую держим в запасе.
    double reserve_mana = 0.0;
    // Сколько юнитов можно позвать за шаг.
    int max_spawn_per_tick = 1;
    // Копить ли ману под более сильного юнита.
    bool save_mana_enabled = false;
    // Минимальный выигрыш от ожидания.
    double save_mana_min_gain = 1.5;
    // Максимальное ожидание в секундах (ждать дольше — поздно: враг успевает
    // пройти).
    double save_mana_max_wait = 8.0;
    // Высота поля в канонических координатах: где середина поля. Если враг
    // перешёл середину — защищаем его линию, иначе толкаем в другую
    // (контр-пуш). Ноль — правило выключено (линия выбирается по давлению).
    double defend_line_height = 640.0;
};

// Оценка давления противника на конкретной линии.
struct LanePressure {
    // Имя линии.
    std::string lane;
    // Суммарная угроза врагов на линии.
    double enemy_threat = 0.0;
    // Суммарная поддержка своих юнитов.
    double ally_support = 0.0;
    // Чистый перевес: угроза минус поддержка.
    double net() const { return enemy_threat - ally_support; }
};

// Планировщик: превращает наблюдаемое состояние в одно решение.
class Planner {
public:
    Planner(PlannerConfig weights, std::vector<UnitSpec> specs);

    // Оценить давление по каждой линии спавна.
    std::vector<LanePressure> lane_pressure(const GameState& state) const;
    // Полезность конкретного типа юнита в текущей ситуации.
    //
    // ``assume_available`` нужен для оценки «будущего» спавна: когда юнит
    // станет доступен, он получит бонус доступности, иначе сравнение с
    // текущим спавном было бы некорректным.
    double score_unit(const std::string& kind, const GameState& state, const Lane* lane,
                      bool assume_available = false,
                      std::map<std::string, double>* parts = nullptr) const;
    // Выбрать действие для текущего состояния.
    Action decide(const GameState& state) const;
    // Решить, стоит ли подождать и накопить ману под более сильного юнита.
    //
    // Возвращает действие "wait", если ожидание выгодно, иначе — пустое
    // действие (kind == "none").
    Action should_save_mana(const GameState& state, const std::string& best_kind) const;
    // Полное решение: спавн, ожидание или копим ману.
    Action plan(const GameState& state) const;
    // Подсказать юнита, на который стоит сосредоточить урон.
    Action suggest_focus(const GameState& state) const;

    // Веса планировщика.
    const PlannerConfig& weights() const { return weights_; }
    // Задать высоту канонического поля: по ней решается, где середина поля
    // (враг перешёл середину — защищаем его линию). Раскладка арены бывает
    // разной, поэтому высота задаётся перед разбором.
    void set_field_height(double height) { weights_.defend_line_height = height; }

private:
    // Типы юнитов, которых можно рассматривать: известные зрением карточки
    // либо все типы, если про карточки ничего не известно.
    std::vector<std::string> roster(const GameState& state) const;
    // Сколько секунд ещё горит карточка этого типа (0 — готова).
    double card_cooldown(const GameState& state, const std::string& kind) const;
    // Готова ли карточка этого типа к призыву.
    bool is_ready(const GameState& state, const std::string& kind) const;
    // Отдача юнита на единицу маны — по метрике из конфига.
    double cost_efficiency(const UnitSpec& spec) const;
    // Угроза врага у линии (без нашей поддержки).
    double lane_threat(const GameState& state, const Lane& lane) const;
    // Линия, которую надо защищать: враг прорвался в нашу половину.
    const Lane* guard_lane(const GameState& state) const;
    // Выбрать линию: где больше врагов и меньше своих сил.
    const Lane* pick_lane(const GameState& state, const std::vector<LanePressure>& pressures) const;
    // Пояснение для человека: две главные составляющие оценки.
    static std::string explain(const std::string& kind, const std::map<std::string, double>& parts,
                                const Lane* lane);
    // Характеристики типа юнита (nullptr, если типа нет).
    const UnitSpec* spec_of(const std::string& kind) const;

    PlannerConfig weights_;
    std::vector<UnitSpec> specs_;
};

}  // namespace rr
