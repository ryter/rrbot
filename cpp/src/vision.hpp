// Зрение: поиск юнитов по эталонам, подавление дубликатов, определение стороны.
//
// Порт rrbot/vision.py на C++. Шаблоны ищутся всегда в каноническом
// пространстве поля, поэтому детектор не зависит от разрешения экрана и от
// того, как игрок расположил окно.
//
// Два режима поиска юнитов:
//
// * **по эталону** (веб-клиент, симулятор): спрайт юнита сопоставляется с
//   эталоном по всей площади;
// * **по круглому основанию** (мобильный клиент): у каждого юнита есть
//   круглое основание, и по нему находится и его место на поле, и его тип.
//   Основание — надёжная метка: у тёмных юнитов силуэт почти без контраста,
//   а совпадение с эталоном путается на 10-20 px. Режим включается сам в
//   analyze_board() (там доска всегда мобильная) или принудительно —
// VisionConfig::detect_by_base.
//
// Два режима определения стороны:
//   * по цвету «тела» юнита (как в Python-прототипе) — подходит симулятору;
//   * по положению на арене — в живом клиенте стороны по цвету не различимы.
//
// Как подключить к боту (cpp/src/main.cpp):
//
//     rr::VisionConfig vcfg;
//     vcfg.templates_dir = (root / "assets" / "real").string();
//     vcfg.template_map = {{"knight", "unit_00.png"}, {"archer", "unit_01.png"},
//                          {"golem", "unit_02.png"}, {"samurai", "unit_03.png"},
//                          {"warden", "unit_04.png"}};
//     vcfg.units = rr::default_live_specs();
//     vcfg.team_by_position = true;   // в живом клиенте цветов сторон нет
//     rr::VisionEngine vision(vcfg);
//     vision.load_templates();
//     rr::Planner planner(rr::PlannerConfig{}, rr::default_live_specs());
//
//     // в цикле боя, рядом с чтением энергии:
//     const rr::ArenaLayout arena = rr::web_arena();      // или rr::emulator_arena()
//     rr::GameState state = vision.analyze(canvas, arena, mana, confidence);
//     rr::add_deploy_lanes(&state);
//     const rr::Action action = planner.plan(state);        // кого звать
//
// Решение планировщика в живом клиенте — подсказка: виджет призыва сам
// выбирает юнита, поэтому клик по кнопке призыва не меняется.
#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "models.hpp"

namespace rr {

// Диапазоны оттенков (шкала OpenCV 0..180) для определения стороны.
struct FactionConfig {
    // Свои юниты.
    std::vector<std::pair<double, double>> ally_hue_ranges;
    // Чужие юниты.
    std::vector<std::pair<double, double>> enemy_hue_ranges;
    // Минимальная насыщенность и яркость, иначе сторона не определяется.
    int min_saturation = 60;
    int min_value = 40;
};

// Параметры детектора.
struct VisionConfig {
    // Каталог с эталонами юнитов.
    std::string templates_dir = "assets/real";
    // Явное сопоставление «тип юнита -> файл эталона».
    std::map<std::string, std::string> template_map;
    // Характеристики типов юнитов (нужны планировщику).
    std::vector<UnitSpec> units;
    // Порог совпадения с эталоном.
    double match_threshold = 0.60;
    // Порог подавления перекрывающихся находок (IoU).
    double nms_iou = 0.25;
    // Сколько находок оставлять от одного эталона.
    int max_per_template = 12;
    // Масштаб эталона под конкретный клиент: в мобильном клиенте юниты
    // крупнее, чем в веб-клиенте, из которого сняты эталоны.
    double template_scale = 1.0;
    // Сравнивать по оттенкам серого (как в Python-прототипе).
    bool grayscale = true;
    // Искать по всем эталонам каталога, а не только по описанным типам.
    bool detect_all_templates = false;
    // Параметры определения стороны по цвету.
    FactionConfig faction;
    // Определять сторону по положению на арене (в живом клиенте).
    bool team_by_position = false;
    // Доля высоты поля, выше которой юнит считается чужим.
    double team_split = 0.5;

    // ---- поиск юнитов по круглому основанию (мобильный клиент) ----
    //
    // В мобильном клиенте каждый юнит стоит на круглом основании. Основание
    // — надёжная метка (у тёмных юнитов силуэт почти без контраста) и
    // вместе с тем точное место юнита на поле: совпадение по эталону
    // центрируется по спрайту и путается на 10-20 px, а основание стоит
    // ровно там, где юнит стоит на земле.
    //
    // В веб-клиенте и в симуляторе оснований нет, поэтому режим выключен
    // по умолчанию, а автоматический режим включается только когда
    // оснований на доске достаточно много.
    bool detect_by_base = false;
    // Включать поиск по основаниям автоматически, если их нашлось много.
    bool auto_detect_by_base = true;
    // Сколько оснований нужно, чтобы доска считалась мобильной.
    int base_min_count = 4;
    // Радиус основания в канонических координатах, px.
    double base_radius_min = 24.0;
    double base_radius_max = 44.0;
    // Минимальное расстояние между центрами оснований, px.
    double base_min_distance = 20.0;
    // Минимальная доля окружности, на которой нашёлся край: у настоящего
    // основания она заметно больше, чем у мусора.
    double base_min_score = 0.045;
    // Минимальная доля углов, на которых граница настоящая (0 — не проверять).
    double base_min_support = 0.0;
    // Порог градиента при поиске оснований: процент краёв, которые голосуют.
    double base_gradient_percentile = 40.0;
    // Убирать круги, под которыми нашлось другое основание: круг выше
    // основания — это контур спрайта юнита, а не само основание.
    bool base_drop_sprite_circles = true;
    // Голосовать только нижней половиной окружности: спрайт закрывает
    // основание сверху, а снизу дуга видна целиком.
    bool base_lower_half_only = true;
    // На каком расстоянии от основания ещё искать совпадение с эталоном, px.
    double base_match_radius = 22.0;
    // Насколько искать положение эталона относительно основания, px.
    int classify_radius = 3;
    // Порог для находок по эталону, у которых основания рядом нет
    // (юнит без круглого основания). 1.0 — не искать вовсе.
    double loose_template_threshold = 1.0;
};

// Один эталон типа юнита.
struct Template {
    // Имя типа (для «прочих» эталонов — имя файла без расширения).
    std::string name;
    // Файл, из которого эталон загружен: по нему ищется карточка в колоде.
    std::string file;
    // Эталон в оттенках серого.
    cv::Mat image;
    // Эталон в цвете (нужен, если сравнение идёт без перевода в серый).
    cv::Mat color;
    // Размер эталона в пикселях.
    int size = 0;
    // Маска «содержимого» тайла: пиксели, которые принадлежат самому
    // юниту (тело, контур, глиф). Чёрные углы тайла — это фон: у соседних
    // юнитов они перекрывают друг друга, и из-за этого обычный отклик
    // падает. Оценка «по видимой части» их не учитывает.
    cv::Mat content;
    // Где внутри эталона находится центр круглого основания юнита.
    //
    // Эталон снимается вырезом вокруг юнита, и основание лежит не в
    // середине выреза: спрайт выше основания, а снизу поля. Поиск по
    // основанию совмещается с эталоном по этой точке.
    cv::Point2d base_offset{-1.0, -1.0};
};

// Хранилище эталонов: загрузка, поиск по имени.
class TemplateBank {
public:
    // Загрузить эталоны из каталога.
    //
    // Для каждого имени из ``names`` берётся файл из ``mapping`` (если есть)
    // или ``<имя>.png``. Остальные картинки каталога грузятся как
    // дополнительные эталоны: в живой игре типов юнитов больше, чем описано
    // в конфиге, и лишние эталоны помогают находить юниты вообще.
    bool load(const std::string& directory, const std::map<std::string, std::string>& mapping,
              const std::vector<std::string>& names, double scale = 1.0);
    // Эталон по имени (nullptr, если такого нет).
    const Template* get(const std::string& name) const;
    // Имена всех загруженных эталонов.
    std::vector<std::string> names() const;
    // Сколько эталонов загружено.
    size_t size() const { return templates_.size(); }

private:
    std::map<std::string, Template> templates_;
    std::vector<std::string> order_;
};

// Круглое основание юнита: в мобильном клиенте юнит стоит на круглом
// основании, и по нему точнее всего находится место юнита на поле.
struct UnitBase {
    // Центр основания в координатах поля.
    cv::Point center;
    // Радиус основания, px.
    int radius = 0;
    // Насколько круг похож на настоящее основание (0..1): у настоящего
    // основания граница круглая почти по всей окружности.
    double quality = 0.0;
};

// Найти круглые основания юнитов на канонической доске.
//
// Основание ищется преобразованием Хафа по светлоте, в диапазоне радиусов
// из конфига. Круги вне арены (пустые места доски) отбрасываются, близкие
// друг к другу — объединяются.
std::vector<UnitBase> find_unit_bases(const cv::Mat& board, const VisionConfig& config);

// Геометрия поля: углы на холсте и канонический размер.
struct ArenaLayout {
    // Четыре точки поля на холсте (по часовой стрелке, с левого верхнего).
    std::vector<cv::Point2f> quad;
    // Канонический размер поля.
    cv::Size board_size{334, 612};
    // Доля высоты поля, выше которой юнит считается чужим.
    double split = 0.5;
};

// Детектор юнитов.
class VisionEngine {
public:
    explicit VisionEngine(VisionConfig config);

    // Загрузить эталоны (без этого поиск ничего не найдёт).
    bool load_templates();
    // Загрузить соответствие «эталон юнита -> номер карточки в колоде».
    //
    // Читается из types.tsv рядом с эталонами (assets/real_emulator):
    // в колоде у каждого типа своя карточка, и чтобы поставить юнита
    // (или слить его с таким же) нужно знать, какую карточку тянуть.
    bool load_deck_map(const std::string& path);
    // Номер карточки в колоде для эталона (0 — неизвестно).
    int deck_position(const std::string& template_name) const;
    // Эталон по имени (nullptr, если такого нет) — для диагностики.
    const Template* template_by_name(const std::string& name) const;
    // Параметры детектора.
    const VisionConfig& config() const { return config_; }
    // Характеристики типов юнитов.
    const std::vector<UnitSpec>& specs() const { return config_.units; }

    // Найти все вхождения одного эталона в каноническом поле.
    std::vector<Detection> match(const cv::Mat& board, const std::string& name) const;
    // То же, но вместе с картой отклика: она нужна для разделения
    // слипшихся спрайтов (внутри одной находки могут прятаться ещё юниты,
    // и их видно по второму максимуму отклика).
    std::vector<Detection> match_map(const cv::Mat& board, const std::string& name,
                                     cv::Mat* response) const;
    // Найти юниты, спрятанные под другими юнитами.
    std::vector<Detection> split_merged(const cv::Mat& board,
                                         const std::vector<Detection>& detections,
                                         const std::map<std::string, cv::Mat>& responses) const;
    // Найти все юниты на каноническом поле и определить их сторону.
    std::vector<Detection> detect(const cv::Mat& board) const;
    // Найти юниты по круглым основаниям (мобильный клиент).
    //
    // Положение юнита — это центр его основания, тип определяется
    // сопоставлением с эталоном в окрестности основания.
    std::vector<Detection> detect_by_bases(const cv::Mat& board) const;
    // То же, но с уже найденными основаниями.
    std::vector<Detection> detect_by_bases(const cv::Mat& board,
                                           const std::vector<UnitBase>& bases) const;
    // Найти юниты только по эталонам (веб-клиент, симулятор).
    std::vector<Detection> detect_by_template(const cv::Mat& board) const;
    // Полный разбор кадра холста: вырезать поле, найти юниты, собрать состояние.
    GameState analyze(const cv::Mat& canvas, const ArenaLayout& arena, double mana,
                       double mana_confidence = 0.0,
                       const std::vector<std::string>& available = {}) const;
    // Разбор уже готового канонического поля.
    //
    // Нужен для мобильного клиента: там канонический кадр собирается
    // особым образом (половины арены по отдельности, общий масштаб по
    // основаниям юнитов — см. `rr::canonical_board`), и обычная
    // перспективная коррекция даёт другую геометрию.
    GameState analyze_board(const cv::Mat& board, double mana, double mana_confidence = 0.0,
                            const std::vector<std::string>& available = {}) const;
    // Собрать состояние поля из находок.
    GameState build_state(const std::vector<Detection>& detections, double mana,
                           double mana_confidence = 0.0,
                           const std::vector<std::string>& available = {}) const;

private:
    // Цвет «тела» юнита: кольцо вокруг центра, без белого глифа типа.
    static void sample_color(const cv::Mat& hsv, const Detection& det, double* hue,
                              int* saturation, int* value);
    // Сторона по цвету.
    Team classify_team(double hue, int saturation, int value) const;
    // Какие эталоны искать: описанные в конфиге типы или все файлы каталога.
    std::vector<std::string> template_names() const;
    // Определить сторону находки: по цвету тела или по положению на арене.
    void assign_teams(const cv::Mat& board, std::vector<Detection>& detections) const;

    VisionConfig config_;
    TemplateBank bank_;
    // Эталон юнита -> номер карточки в колоде.
    std::map<std::string, int> deck_positions_;
    bool loaded_ = false;
};

// Подавление перекрывающихся находок (общее для всех типов, чтобы один юнит
// не давал находку сразу от нескольких шаблонов).
std::vector<Detection> non_max_suppression(std::vector<Detection> detections, double iou_threshold);

// Поле веб-клиента: арена — прямоугольник (66,52)-(400,664) холста 492x804,
// канонический размер 334x612 (в пиксель к пикселю веб-клиента).
ArenaLayout web_arena();
// Поле мобильного клиента: нижняя половина арены холста 950x1034.
//
// Камера в бою ходит (арена то приближается, то отдаляется), поэтому прямоугольник
// задан приблизительно — для точного разбора нужен поиск арены по кадру.
ArenaLayout emulator_arena();

// Вырезать каноническое поле из холста (перспективная коррекция).
cv::Mat extract_field(const cv::Mat& canvas, const ArenaLayout& arena);

// Добавить в состояние точки высадки: левая и правая линия внизу поля.
//
// Доли взяты из симулятора: свои юниты появляются внизу, уходят к вражеской
// базе через мост своей линии.
void add_deploy_lanes(GameState* state);

// Характеристики типов юнитов из config/live.yaml.
//
// Конфиг на Python читается через yaml, в C++ парсера yaml нет, поэтому
// значения продублированы здесь. При изменении config/live.yaml их нужно
// поправить и здесь (в измерениях используется конфиг, а не эти значения).
std::vector<UnitSpec> default_live_specs();

// Сопоставление «тип юнита -> файл эталона» из config/live.yaml.
//
// Эталоны сняты с настоящего поля; соответствие между именем в конфиге и
// спрайтом в игре пока приблизительное — порядок карточек в колоде совпадает.
std::map<std::string, std::string> default_live_template_map();
// Эталоны мобильного клиента (assets/real_emulator).
std::map<std::string, std::string> default_emulator_template_map();

}  // namespace rr
