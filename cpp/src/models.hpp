// Доменные модели: характеристики юнитов, находки, состояние поля, решение.
//
// Это порт rrbot/models.py на C++: те же сущности и те же названия полей,
// чтобы результаты замеров Python-прототипа и C++-бота можно было сравнивать.
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace rr {

// Сторона юнита на поле.
enum class Team {
    Unknown,
    Ally,
    Enemy,
};

// Название стороны по-русски (для логов и отчётов).
inline const char* team_name(Team team) {
    switch (team) {
        case Team::Ally: return "ally";
        case Team::Enemy: return "enemy";
        default: return "unknown";
    }
}

// Статические характеристики типа юнита.
struct UnitSpec {
    std::string name;
    // Цена призыва в мане.
    double cost = 100.0;
    // Запас здоровья.
    double hp = 100.0;
    // Урон за удар.
    double damage = 0.0;
    // Пауза между ударами, с.
    double attack_interval = 1.0;
    // Скорость движения, пикселей/с.
    double speed = 0.0;
    // Дальность атаки, пикселей.
    double attack_range = 0.0;
    // Радиус юнита, пикселей.
    double radius = 9.0;
    // Семейство (melee, ranged, tank, splash) — для отчётов.
    std::string family = "melee";
    // Типы юнитов, которые этот юнит перебивает.
    std::vector<std::string> counters;

    // Урон в секунду.
    double dps() const { return damage / std::max(attack_interval, 1e-6); }
    // Запас здоровья на единицу маны.
    double hp_per_cost() const { return hp / std::max(cost, 1.0); }
    // Живучесть × урон на единицу маны.
    double value_per_cost() const { return hp * dps() / std::max(cost, 1.0); }
};

// Находка детектора в канонических координатах поля.
struct Detection {
    // Имя типа юнита (для «прочих» эталонов — имя файла).
    std::string name;
    // Игровой тип юнита, если он известен (по таблице соответствия
    // «эталон -> тип»). Пусто, если соответствия нет.
    std::string type;
    // Номер карточки этого типа в колоде (1..N), если известен.
    int deck_position = 0;
    // Уверенность совпадения, 0..1.
    double score = 0.0;
    // Прямоугольник находки в координатах поля.
    cv::Rect box;
    // Центр находки в координатах поля.
    cv::Point2d center{0.0, 0.0};
    // Сторона (определяется по цвету или по положению на арене).
    Team team = Team::Unknown;
    // Оттенок и насыщенность «тела» юнита.
    double hue = 0.0;
    double saturation = 0.0;
};

// Линия высадки: точка, где можно поставить юнита.
struct Lane {
    std::string name;
    double x = 0.0;
    double y = 0.0;
};

// Юнит на поле.
struct Unit {
    // Идентификатор в пределах одного кадра.
    int uid = 0;
    // Тип юнита.
    std::string kind;
    // Сторона.
    Team team = Team::Unknown;
    // Положение в канонических координатах поля.
    double x = 0.0;
    double y = 0.0;
    // Здоровье и его максимум.
    double hp = 0.0;
    double max_hp = 0.0;
    // Характеристики типа (не владеет, указатель на таблицу спецификаций).
    const UnitSpec* spec = nullptr;

    // Доля оставшегося здоровья, 0..1.
    double hp_fraction() const {
        if (max_hp <= 0.0) return 0.0;
        return std::max(0.0, std::min(1.0, hp / max_hp));
    }
    // Урон в секунду (0, если характеристики нет).
    double dps() const { return spec != nullptr ? spec->dps() : 0.0; }
};

// Состояние поля — результат работы зрения.
struct GameState {
    // Энергия и уверенность её чтения.
    double mana = 0.0;
    double mana_max = 500.0;
    double mana_confidence = 0.0;
    // Сколько маны набегает в секунду: нужно, чтобы оценить, сколько ждать
    // до нужной суммы маны.
    double mana_regen = 34.0;
    // Юниты своих и чужих.
    std::vector<Unit> ally_units;
    std::vector<Unit> enemy_units;
    // Все находки детектора.
    std::vector<Detection> detections;
    // Точки высадки по линиям.
    std::map<std::string, Lane> spawn_points;
    // Типы юнитов, которых можно заспавнить прямо сейчас.
    std::vector<std::string> available_units;
    // Известен ли список доступных юнитов. Если список пуст и про него ничего
    // не известно, доступны все типы (в Python это None).
    bool available_known = false;
    // Сколько секунд ещё горит каждая карточка (до готовности).
    std::map<std::string, double> card_cooldowns;
    // Откуда получено состояние (для логов).
    std::string source = "live";

    // Все юниты поля.
    std::vector<Unit> all_units() const {
        std::vector<Unit> out = ally_units;
        out.insert(out.end(), enemy_units.begin(), enemy_units.end());
        return out;
    }
};

// Решение планировщика.
struct Action {
    // "spawn" — звать юнита, "focus" — бить цель, "wait" — ждать.
    std::string kind = "wait";
    // Тип юнита для спавна.
    std::string unit_kind;
    // Линия высадки.
    std::string lane;
    // Точка высадки в канонических координатах поля.
    bool has_position = false;
    double position_x = 0.0;
    double position_y = 0.0;
    // Идентификатор цели (для "focus").
    int target_uid = -1;
    // Пояснение для человека.
    std::string reason;
    // Итоговая оценка.
    double score = 0.0;
    // Составляющие оценки.
    std::map<std::string, double> scores;
};

}  // namespace rr
