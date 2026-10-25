// Автобой: запуск на живой игре.
//
// Бот сам проходит по меню, анализирует поле и призывает юнитов.
// Кликает настоящей мышью (виртуальное устройство uinput).
//
// Запуск:
//     ./build/rrbot --steps 40
//     ./build/rrbot --steps 40 --dry-run      # только считать, не кликать
//     ./build/rrbot --debug                  # показывать, что бот видит

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "arena.hpp"
#include "board.hpp"
#include "capture.hpp"
#include "common.hpp"
#include <memory>

#include "game.hpp"
#include "hud.hpp"
#include "merge.hpp"
#include "models.hpp"
#include "mouse.hpp"
#include "planner.hpp"
#include "vision.hpp"

namespace {

void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// Показ кликов: в окне видно, куда бот нажимает, — можно поправить.
// Если окно открыть не удалось (нет графики), кадры с отметками
// складываются в каталог.
bool g_show = false;
bool g_window_failed = false;
int g_click_index = 0;
std::string g_out_dir = "/tmp/opencode/rrbot-cpp";

void show_click(const cv::Mat& canvas, cv::Point point, const std::string& kind, int index) {
    if (canvas.empty() || !g_show) return;
    cv::Mat view = canvas.clone();
    cv::circle(view, point, 24, cv::Scalar(0, 0, 255), 3);
    cv::putText(view, kind, point + cv::Point(-16, -30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 255), 2);
    if (!g_window_failed) {
        try {
            cv::imshow("rrbot — клики", view);
            cv::waitKey(1);
            return;
        } catch (const cv::Exception&) {
            g_window_failed = true;
        }
    }
    cv::imwrite(g_out_dir + "/click_" + std::to_string(index) + ".png", view);
}

// Перетащить пешку на пешку, ведя курсор за целью.
//
// Камера в бою едет: пока палец идёт к цели (перетаскивание занимает около
// секунды), цель смещается на десятки пикселей — и слияние попадает мимо.
// Поэтому после зажатия берём свежий кадр, определяем сдвиг поля по пешкам
// и поправляем точку отпускания.
//
// Сдвиг считаем по пешкам (rr::estimate_pawn_shift), а не по положению
// арены: детектор арены гуляет на десятки пикселей.
bool drag_merge(rr::Game& game, const std::vector<rr::Pawn>& pawns, cv::Point from, cv::Point to) {
    if (!game.drag_start_canvas(from)) {
        // Без мыши (или без окна) — обычный свайп через adb.
        return game.drag_canvas(from, to);
    }
    cv::Point cursor = from;
    cv::Point target = to;
    std::vector<rr::Pawn> before = pawns;
    for (int round = 0; round < 3; ++round) {
        // Ведём курсор к цели плавно: игра ждёт, пока палец «проведёт» по
        // карточке, иначе перетаскивание не засчитывается.
        for (int s = 1; s <= 5; ++s) {
            const double t = static_cast<double>(s) / 5.0;
            game.drag_move_canvas(
                cv::Point(static_cast<int>(std::lround(cursor.x + (target.x - cursor.x) * t)),
                          static_cast<int>(std::lround(cursor.y + (target.y - cursor.y) * t))));
            sleep_ms(40);
        }
        cursor = target;
        // Свежий кадр: за это время камера могла увести цель.
        const cv::Mat fresh = game.grab_canvas();
        if (fresh.empty()) continue;
        const rr::ArenaFrame arena = rr::find_arena(fresh);
        if (!arena.found) continue;
        const std::vector<rr::Pawn> after = rr::find_pawns(fresh, arena);
        const cv::Point shift = rr::estimate_pawn_shift(before, after, 60);
        if (shift.x == 0 && shift.y == 0) continue;
        before = after;
        target += shift;
        std::cout << "    веду за целью: сдвиг поля (" << shift.x << "," << shift.y << ")" << std::endl;
    }
    game.drag_end_canvas(target);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int steps = 40;
    bool dry_run = false;
    bool debug = false;
    bool probe = false;
    std::string backend_name = "desktop";
    std::string out_dir = "/tmp/opencode/rrbot-cpp";
    // Разбор сохранённого кадра вместо живого: удобно проверять чтение
    // цифр на записанных боях, не запуская игру.
    std::string frame_path;
    // Сбор эталона цифр: значение, написанное на экране («60»).
    std::string learn;
    // Куда складывать новые эталоны (по умолчанию — в каталог текущих).
    std::string learn_dir;
    // Больше этого числа образцов на цифру не сохраняем.
    int learn_max = 1000;
    // Сколько раундов (бой — выход в меню — снова бой) играть подряд.
    int rounds = 1;
    // Каталог для записи кадров боя: из них потом собираются эталоны юнитов.
    std::string record_dir;
    // Разбор поля: поиск юнитов и рекомендация планировщика. По умолчанию
    // выключено: в мобильном клиенте эталоны юнитов пока не подходят (они
    // сняты с веб-клиента), и разбор давал бы шум. Для веб-клиента работает.
    bool analyze_field = false;
    // Слияние и призыв можно отключить по отдельности: так проверяется жест
    // слияния вживую (бот только зовёт юнитов — а слияние делаем руками).
    bool no_merge = false;
    bool no_summon = false;
    // Умение героя (круглая кнопка справа внизу) можно отключить: так
    // проверяется жест нажатия вживую.
    bool no_ability = false;
    // Режим боя: по умолчанию Co-Op (соперника в PvP на этом сервере нет),
    // с ключом --pvp — PvP.
    bool coop_mode = true;
    bool help = false;
    // Каталог эталонов цифр: по умолчанию — из раскладки бэкенда.
    std::string digits_dir;
    // Картинка с самим числом (вырез области энергии): читаем цифры
    // прямо из неё, без раскладки.
    std::string digits_image;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "--dry-run") dry_run = true;
        else if (arg == "--debug") debug = true;
        else if (arg == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (arg == "--probe") probe = true;
        else if (arg == "--show") g_show = true;
        else if (arg == "--frame" && i + 1 < argc) frame_path = argv[++i];
        else if (arg == "--learn" && i + 1 < argc) learn = argv[++i];
        else if (arg == "--learn-dir" && i + 1 < argc) learn_dir = argv[++i];
        else if (arg == "--learn-max" && i + 1 < argc) learn_max = std::stoi(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds = std::stoi(argv[++i]);
        else if (arg == "--record" && i + 1 < argc) record_dir = argv[++i];
        else if (arg == "--vision") analyze_field = true;
        else if (arg == "--no-merge") no_merge = true;
        else if (arg == "--no-summon") no_summon = true;
        else if (arg == "--no-ability") no_ability = true;
        else if (arg == "--pvp") coop_mode = false;
        else if (arg == "--help" || arg == "-h") help = true;
        else if (arg == "--digits-dir" && i + 1 < argc) digits_dir = argv[++i];
        else if (arg == "--digits-image" && i + 1 < argc) digits_image = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }
    std::filesystem::create_directories(out_dir);
    g_out_dir = out_dir;

    if (help) {
        std::cout <<
            "Бот Rush Royale (C++).\n\n"
            "  --backend desktop|adb|emulator   источник кадра и клика\n"
            "  --steps N                        шагов в бою\n"
            "  --rounds N                       раундов подряд: бой → меню → бой\n"
            "  --probe                           показать, что бот видит в кадре\n"
            "  --frame ФАЙЛ                      разобрать сохранённый кадр\n"
            "  --digits-image ФАЙЛ              прочитать число из выреза энергии\n"
            "  --digits-dir КАТАЛог              эталоны цифр из другого каталога\n"
            "  --learn ЗНАЧЕНИЕ                  сохранить глифы числа как эталоны\n"
            "  --learn-dir КАТАЛог               куда складывать эталоны\n"
            "  --learn-max N                     максимум образцов на цифру\n"
            "  --record КАТАЛог                  записывать кадры боя\n"
            "  --vision                          разбор поля и подсказка планировщика\n"
            "  --dry-run                         считать, не кликая (веб-клиент)\n"
            "  --debug                           сохранять кадры боя в --out\n"
            "  --out КАТАЛог                     куда складывать кадры\n";
        return 0;
    }

    // Корень проекта: рядом с build/ лежит каталог assets.
    std::filesystem::path root = std::filesystem::path(argv[0]).parent_path().parent_path();
    if (!std::filesystem::exists(root / "assets")) {
        root = std::filesystem::path("/home/lennistyle/rushroyale-lab");
    }
    std::unique_ptr<rr::Backend> backend;
    if (backend_name == "adb") {
        backend = rr::make_adb_backend(false);
    } else if (backend_name == "emulator") {
        // Кадр из устройства, а клик — настоящей мышью по окну эмулятора:
        // клик видно на экране.
        backend = rr::make_adb_backend(true);
    } else {
        backend = rr::make_desktop_backend(dry_run);
    }
    // Раскладка приводится к настоящему размеру кадра: иначе кадр
    // растягивается на процент, и цифры энергии после этого не читаются.
    // Размер кадра узнаётся только из первого кадра (устройство может
    // повернуться), поэтому раскладку уточняем по ходу дела.
    const auto layout_for_size = [&](const cv::Size& size) {
        if (backend_name == "desktop") return rr::web_layout();
        return rr::emulator_layout().scaled_to(size);
    };
    rr::Layout layout = layout_for_size(backend->layout().canvas_size);
    rr::Hud hud((root / "assets").string(), layout);
    if (!digits_dir.empty()) {
        const std::string dir =
            digits_dir.front() == '/' ? digits_dir : (root / digits_dir).string();
        hud.use_digits(dir);
    }
    // Пересчитать раскладку под настоящий размер кадра.
    const auto update_layout = [&](const cv::Size& size) {
        if (size.width <= 0 || size.height <= 0) return;
        const rr::Layout next = layout_for_size(size);
        if (next.canvas_size == layout.canvas_size) return;
        layout = next;
        hud.set_layout(layout);
        std::cout << "    раскладка пересчитана под кадр " << size.width << "x" << size.height
                  << std::endl;
    };
    std::cout << "бэкенд: " << backend_name << " (" << layout.name << "), кадр "
              << layout.canvas_size.width << "x" << layout.canvas_size.height << std::endl;

    rr::Game game(backend.get());

    // Зрение и планировщик нужны всегда: по находкам бот сливает одинаковых
    // юнитов (тянет карточку на такого же). Ключ --vision включает печать
    // разбора поля — для замеров и отладки.
    std::unique_ptr<rr::VisionEngine> vision;
    std::unique_ptr<rr::Planner> planner;
    {
        rr::VisionConfig vcfg;
        // Эталоны юнитов: у мобильного клиента свой спрайт и свой масштаб,
        // поэтому для него — свой каталог. Пока его нет, берём веб-клиентские
        // (на мобильном клиенте они не подходят — см. HANDOFF).
        std::filesystem::path templates = root / "assets" / "real";
        std::map<std::string, std::string> template_map = rr::default_live_template_map();
        if (backend_name != "desktop") {
            const std::filesystem::path emulator_templates = root / "assets" / "real_emulator";
            if (std::filesystem::exists(emulator_templates)) {
                templates = emulator_templates;
                template_map = rr::default_emulator_template_map();
            }
        }
        vcfg.templates_dir = templates.string();
        vcfg.template_map = template_map;
        vcfg.units = rr::default_live_specs();
        // В живом клиенте стороны по цвету не различимы: своих определяем
        // по половине арены.
        vcfg.team_by_position = true;
        vision = std::make_unique<rr::VisionEngine>(vcfg);
        if (!vision->load_templates()) {
            std::cerr << "эталоны юнитов не загрузились" << std::endl;
            return 1;
        }
        // Соответствие «эталон юнита -> карточка в колоде»: нужно, чтобы
        // тянуть карточку на такого же юнита (слияние). Читается рядом с
        // эталонами, если файл есть.
        const std::string types_path =
            (root / "assets" / "real_emulator" / "types.tsv").string();
        if (vision->load_deck_map(types_path)) {
            std::cout << "карточки колоды: " << types_path << std::endl;
        }
        // Планировщику нужна высота канонического поля: по ней решается,
        // где середина (враг перешёл середину — защищаем его линию).
        // Высота известна из раскладки арены, поэтому ставим её перед
        // каждым разбором поля.
        rr::PlannerConfig pcfg;
        pcfg.defend_line_height = 640.0;
        planner = std::make_unique<rr::Planner>(pcfg, rr::default_live_specs());
        std::cout << "разбор поля: " << vision->config().templates_dir << std::endl;
    }

    // Режим разведки: показать, что бот видит в кадре.
    const auto report = [&](const cv::Mat& canvas) {
        double confidence = 0.0;
        std::cout << "кадр: " << canvas.cols << "x" << canvas.rows << std::endl;
        const rr::DigitReader::Reading digits = hud.read_mana_digits(canvas);
        std::cout << "цифры энергии: \"" << digits.text << "\" (уверенность "
                  << digits.confidence << ")" << std::endl;
        // По каждой цифре отдельно: видно, какая именно не узналась.
        std::cout << "по цифрам:";
        for (const auto& [name, score] : digits.glyphs) {
            std::cout << " " << name << "(" << score << ")";
        }
        std::cout << std::endl;
        std::cout << "энергия: " << hud.read_mana(canvas, &confidence)
                  << " (уверенность " << confidence << ")" << std::endl;
        std::cout << "флакон: " << (hud.has_energy_flask(canvas) ? "есть" : "нет") << std::endl;
        std::cout << "призыв доступен: " << (hud.summon_ready(canvas) ? "да" : "нет") << std::endl;
        const cv::Point ability = hud.ability_button(canvas);
        std::cout << "умение: кнопка (" << ability.x << "," << ability.y << "), готово "
                  << (hud.ability_ready(canvas) ? "да" : "нет") << std::endl;
        std::cout << "в бою: " << (hud.in_battle(canvas) ? "да" : "нет") << std::endl;
        cv::Rect text_rect;
        std::cout << "надпись PvP: "
                  << (hud.find_text(canvas, "pvp", &text_rect)
                          ? "найдена в " + std::to_string(text_rect.x) + "," +
                                std::to_string(text_rect.y)
                          : std::string("не найдена"))
                  << std::endl;
        // Надпись «Играть» на экране этажей: на платных этажах кнопка
        // золотая, и шаблон (снятый с синей кнопки) совпадает с ней хуже.
        std::cout << "надпись «Играть»: "
                  << (hud.find_text(canvas, "play", &text_rect, 0.40)
                          ? "найдена в " + std::to_string(text_rect.x) + "," +
                                std::to_string(text_rect.y)
                          : std::string("не найдена"))
                  << std::endl;
        std::cout << "кнопки:";
        for (const rr::UiElement& button : hud.find_buttons(canvas)) {
            std::cout << " " << button.kind << "(" << button.x << "," << button.y << "," << button.w
                      << "x" << button.h << ")";
        }
        std::cout << std::endl;
        std::cout << "кандидаты навигации:";
        for (const rr::UiElement& button : hud.navigation_candidates(canvas)) {
            std::cout << " " << button.kind << "(" << button.x << "," << button.y << ")";
        }
        std::cout << std::endl;
        // Экран Co-Op: какой из них и куда жать.
        cv::Rect coop_rect;
        const rr::Hud::CoopScreen coop_screen = hud.coop_screen(canvas, &coop_rect);
        const char* coop_name = "нет";
        switch (coop_screen) {
            case rr::Hud::CoopScreen::kModifier: coop_name = "модификатор"; break;
            case rr::Hud::CoopScreen::kAlly: coop_name = "союзник"; break;
            case rr::Hud::CoopScreen::kFloors: coop_name = "этажи"; break;
            default: break;
        }
        std::cout << "экран коопа: " << coop_name << " (" << coop_rect.x << "," << coop_rect.y << ","
                  << coop_rect.width << "x" << coop_rect.height << ")" << std::endl;
    };

    if (!digits_image.empty()) {
        // Картинка с числом: читаем цифры прямо из неё.
        cv::Mat image = cv::imread(digits_image, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "картинка не прочитана: " << digits_image << std::endl;
            return 1;
        }
        const rr::DigitReader::Reading reading = hud.digits().read(image);
        std::cout << "цифры: \"" << reading.text << "\" (уверенность " << reading.confidence << ")"
                  << std::endl;
        std::cout << "по цифрам:";
        for (const auto& [name, score] : reading.glyphs) {
            std::cout << " " << name << "(" << score << ")";
        }
        std::cout << std::endl;
        return 0;
    }

    if (!frame_path.empty()) {
        // Разбор сохранённого кадра: тот же расклад, что и у бэкенда.
        cv::Mat canvas = cv::imread(frame_path, cv::IMREAD_COLOR);
        if (!canvas.empty()) update_layout(cv::Size(canvas.cols, canvas.rows));
        if (canvas.empty()) {
            std::cerr << "кадр не прочитан: " << frame_path << std::endl;
            return 1;
        }
        // Разбор всегда идёт в каноническом размере холста: правила
        // поиска кнопок и шаблоны надписей настроены под него, а найденные
        // элементы пересчитываются обратно в координаты кадра.
        rr::resize_to_canvas(&canvas, rr::canonical_canvas_size(layout));
        report(canvas);
        if (vision && planner) {
            // Разбор поля по сохранённому кадру: арена ищется в кадре,
            // дальше — поиск юнитов и рекомендация планировщика. Так весь
            // разбор проверяется без запуска игры.
            const rr::ArenaFrame found = rr::find_arena(canvas);
            const cv::Mat board = rr::canonical_board(canvas);
            const rr::GameState state = board.empty()
                ? rr::GameState()
                : vision->analyze_board(board, 0.0, 0.0);
            rr::GameState with_lanes = state;
            rr::add_deploy_lanes(&with_lanes);
            // Высота канонического поля — по ней планировщик решает, где
            // середина поля.
            planner->set_field_height(board.empty() ? 640.0 : board.rows);
            const rr::Action action = planner->plan(with_lanes);
            std::cout << "арена: " << (found.found ? "найдена" : "не найдена")
                      << (found.found ? " " + std::to_string(found.full.x) + "," +
                                            std::to_string(found.full.y) + " " +
                                            std::to_string(found.full.width) + "x" +
                                            std::to_string(found.full.height)
                                      : "")
                      << std::endl;
            std::cout << "юнитов: своих " << with_lanes.ally_units.size() << ", чужих "
                      << with_lanes.enemy_units.size() << "; рекомендую: " << action.reason
                      << std::endl;
            // Найденные юниты: видно, какие из них можно слить.
            for (const rr::Detection& det : state.detections) {
                std::cout << "  " << (det.team == rr::Team::Ally ? "свой " : "чужой ") << det.name
                          << " карточка=" << det.deck_position << " ("
                          << static_cast<int>(det.center.x) << ","
                          << static_cast<int>(det.center.y) << ")" << std::endl;
            }
        }
        if (!learn.empty()) {
            // Сбор эталона: глифы идут слева направо, как цифры значения.
            const std::vector<cv::Mat> glyphs = hud.mana_glyphs(canvas);
            if (glyphs.size() != learn.size()) {
                std::cerr << "глифов " << glyphs.size() << ", а цифр в значении " << learn.size()
                          << std::endl;
                return 1;
            }
            for (size_t i = 0; i < glyphs.size(); ++i) {
                const std::string digit(1, learn[i]);
                // Образцов на цифру может быть много — но не бесконечно:
                // лишние не добавляем, чтобы каталог не разрастался.
                const std::string dir = learn_dir.empty() ? std::string() : learn_dir;
                const std::string target =
                    dir.empty() ? (root / "assets" / backend->layout().digits_dir).string() : dir;
                // Каталога может не быть — создаём, иначе перебор файлов
                // упадёт с исключением.
                std::error_code ec;
                std::filesystem::create_directories(target, ec);
                int have = 0;
                for (const auto& entry : std::filesystem::directory_iterator(target, ec)) {
                    const std::string stem = entry.path().stem().string();
                    if (stem == digit || stem.rfind(digit + "_", 0) == 0) ++have;
                }
                if (have >= learn_max) {
                    std::cout << "пропущено (" << digit << "): образцов уже " << have << std::endl;
                    continue;
                }
                const std::string path = hud.digits().save_sample(digit, glyphs[i], dir);
                std::cout << "сохранено: " << path << std::endl;
            }
        }
        return 0;
    }

    if (probe) {
        const cv::Mat canvas = game.grab_canvas();
        if (canvas.empty()) {
            std::cerr << "кадр не получен" << std::endl;
            return 1;
        }
        update_layout(cv::Size(canvas.cols, canvas.rows));
        report(canvas);
        return 0;
    }
    auto click_canvas = [&](cv::Point canvas_point) { game.click_canvas(canvas_point); };

    // 1) Дойти до боя.
    //
    // Два пути: PvP и Co-Op. По умолчанию бот идёт в Co-Op: на этом сервере
    // соперник в PvP находится через раз, а в кооп бой начинается сразу.
    //
    // Путь в Co-Op (Подземелья) состоит из четырёх экранов:
    //   главное меню -> «Co-Op» -> «Играть» на первом этаже ->
    //   «Случайный» (союзник) -> выбор модификатора -> бой.
    // Каждый экран узнаётся по тому, что на нём есть: надпись «Играть»,
    // надпись «Случайный», серо-синие карточки модификаторов.
    //
    // Экраны выбора союзника и модификатора обычная кнопочная логика не
    // берёт: там кнопок много, и первая попавшаяся — не та («Случайный»
    // рядом с иконкой медали, а её обводка даёт ту же синюю заливку).
    const auto go_to_battle = [&]() -> bool {
    int pvp_fails = 0;
    int pvp_waits = 0;
    // Сколько подряд кликов не дали эффекта: по этому числу понимаем,
    // что бот упёрся в экран, из которого сам не выйдет.
    int stuck = 0;
    bool awaiting_pvp = false;
    std::map<std::string, int> dead;
    std::map<std::string, int> tries;
    bool in_battle = false;
    // Идём по экранам Co-Op (этаж -> союзник -> модификатор -> бой).
    // Пока идём, «Назад» жать нельзя: он возвращает на предыдущий экран, и
    // бот топчется по кругу. Кроме того, не жмём ничего лишнего (рекламу
    // в том числе) — на этих экранах есть только свои кнопки.
    bool coop_flow = false;
    // Только что нажали кнопку в окне («Остаться», «Продолжить»): сразу
    // после этого «Назад» жать нельзя — окно «выйти из игры?» появится
    // снова, и бот будет ходить по кругу.
    int dialog_clicked = 0;
    // Сколько раз за проход бот отходил назад. Больше трёх — опасно: игра
    // закрывается, и бот остаётся на пустом экране лаунчера.
    int back_presses = 0;
    // Вернуть игру на экран (эмулятор): если бот всё-таки её закрыл.
    const auto relaunch_game = [&]() {
        std::system("waydroid app launch com.my.defense > /dev/null 2>&1");
        sleep_ms(8000);
    };
    for (int step = 0; step < 60 && !in_battle; ++step) {
        const cv::Mat canvas = game.grab_canvas();
        if (canvas.empty()) {
            sleep_ms(1000);
            continue;
        }
        // Раскладка — под настоящий размер кадра (игра могла повернуться).
        update_layout(cv::Size(canvas.cols, canvas.rows));
        // Игра ещё переворачивается: кадр не того пропорций, что обычно
        // (у мобильного клиента 941x1030). Разбирать такой кадр бессмысленно
        // — области раскладки едут, ждём.
        const double aspect = static_cast<double>(canvas.cols) / std::max(1, canvas.rows);
        if (std::abs(aspect - 941.0 / 1030.0) > 0.08) {
            sleep_ms(1500);
            continue;
        }

        // Экраны Co-Op разбираем первыми: выбор модификатора приходит уже во
        // время боя (окно висит над полем), и обычная проверка «идёт ли бой»
        // на нём срабатывает — бот решил бы, что бой начался, и потерял бы
        // первые волны.
        if (coop_mode) {
            cv::Rect coop_target;
            const rr::Hud::CoopScreen coop_screen = hud.coop_screen(canvas, &coop_target);
            const cv::Point coop_point(coop_target.x + coop_target.width / 2,
                                       coop_target.y + coop_target.height / 2);
            if (!record_dir.empty()) {
                // Кадры экранов меню: по ним видно, на каком экране бот
                // остановился (и почему не пошёл дальше).
                std::error_code ec;
                std::filesystem::create_directories(record_dir, ec);
                cv::imwrite((std::filesystem::path(record_dir) /
                             ("m" + std::to_string(step) + "_" +
                              std::to_string(static_cast<int>(coop_screen)) + ".png"))
                                .string(),
                            canvas);
            }
            if (coop_screen == rr::Hud::CoopScreen::kModifier) {
                std::cout << "[кооп " << step << "] выбираю модификатор" << std::endl;
                // Значок модификатора реагирует только на касание: клик
                // мышью по нему ничего не делает.
                game.touch_canvas(coop_point);
                coop_flow = true;
                sleep_ms(1500);
                continue;
            }
            if (coop_screen == rr::Hud::CoopScreen::kAlly) {
                std::cout << "[кооп " << step << "] выбираю союзника «Случайный»" << std::endl;
                click_canvas(coop_point);
                coop_flow = true;
                sleep_ms(2500);
                continue;
            }
            if (coop_screen == rr::Hud::CoopScreen::kFloors) {
                std::cout << "[кооп " << step << "] жму «Играть»" << std::endl;
                click_canvas(coop_point);
                coop_flow = true;
                sleep_ms(2500);
                continue;
            }
        }

        if (hud.in_battle(canvas)) {
            // Проверяем ещё раз: на экране итогов бывают кадры, где поле
            // ещё видно, и бот принял бы их за начало боя.
            sleep_ms(600);
            const cv::Mat again = game.grab_canvas();
            if (!again.empty() && hud.in_battle(again) && rr::find_arena(again).found) {
                std::cout << "== матч начался (шаг " << step << ") ==" << std::endl;
                in_battle = true;
                coop_flow = false;
                break;
            }
            continue;
        }
        std::vector<rr::UiElement> candidates = hud.navigation_candidates(canvas);
        if (pvp_fails > 0) {
            // PvP на этом сервере соперников не находит — берём Co-Op.
            std::vector<rr::UiElement> coops;
            for (const rr::UiElement& b : candidates) {
                // Кнопка «Co-Op» найдена по надписи; запасной вариант —
                // синяя кнопка справа от «PvP».
                if (b.kind == "coop") {
                    coops.push_back(b);
                } else if (b.kind == "continue" && b.center().x > 300 && b.y > 560 &&
                           b.y < 670) {
                    coops.push_back(b);
                }
            }
            if (!coops.empty()) {
                std::vector<rr::UiElement> merged = coops;
                for (const rr::UiElement& b : candidates) {
                    bool already = false;
                    for (const rr::UiElement& c : coops) {
                        if (c.x == b.x && c.y == b.y && c.w == b.w) {
                            already = true;
                            break;
                        }
                    }
                    if (!already) merged.push_back(b);
                }
                candidates = merged;
            }
        }
        if (coop_mode) {
            // В режиме Co-Op PvP не жмём никогда: соперника на сервере нет,
            // а поиск матча занимает полминуты. Кнопка «Co-Op» — первая.
            std::vector<rr::UiElement> coops;
            for (const rr::UiElement& b : candidates) {
                if (b.kind == "pvp" || b.kind == "orange") continue;
                if (b.kind == "coop") coops.push_back(b);
            }
            // «Отмена» в Co-Op — это отмена выбора модификатора: жать её
            // нельзя, иначе бот начинает экран заново по кругу.
            std::vector<rr::UiElement> without_cancel;
            for (const rr::UiElement& b : candidates) {
                if (b.kind == "pvp" || b.kind == "orange") continue;
                if (b.kind == "cancel") continue;
                without_cancel.push_back(b);
            }
            candidates = without_cancel;
            if (!coops.empty()) {
                std::vector<rr::UiElement> merged = coops;
                for (const rr::UiElement& b : candidates) {
                    bool already = false;
                    for (const rr::UiElement& c : coops) {
                        if (c.x == b.x && c.y == b.y && c.w == b.w) {
                            already = true;
                            break;
                        }
                    }
                    if (!already) merged.push_back(b);
                }
                candidates = merged;
            }
        }
        bool clicked = false;
        bool any_effect = false;
        for (const rr::UiElement& button : candidates) {
            // Фиктивный клик в центр экрана: в меню он бесполезен, а на
            // экране итогов попадает по панели награды. Лучше подождать.
            if (button.kind == "tap") continue;
            if (dead.count(button.kind) && dead[button.kind] > step) continue;
            // Экран с анимацией «меняется» после любого клика, поэтому одну
            // и ту же кнопку не жмём подряд больше трёх раз.
            if (tries[button.kind] >= 3) {
                dead[button.kind] = step + 12;
                tries[button.kind] = 0;
                continue;
            }
            tries[button.kind] += 1;
            const cv::Mat before = canvas.clone();
            show_click(canvas, button.center(), button.kind, g_click_index++);
            click_canvas(button.center());
            if (button.kind == "pvp" || button.kind == "coop") {
                // Ждём матч: подбор соперника идёт по сети и может занять
                // полминуты. Пока идёт — экран анимируется, поэтому по
                // кадру «изменилось» судить бесполезно: смотрим, начался
                // ли бой.
                bool started = false;
                for (int wait = 0; wait < 30; ++wait) {
                    sleep_ms(1000);
                    const cv::Mat frame = game.grab_canvas();
                    if (frame.empty()) continue;
                    // Настоящий бой отличается от экранов меню наличием поля:
                    // в меню (этажи Co-Op, главное меню) проверка «идёт ли
                    // бой» срабатывает — там тоже есть числа и флакон.
                    if (hud.in_battle(frame) && rr::find_arena(frame).found) {
                        started = true;
                        break;
                    }
                }
                if (started) {
                    std::cout << "== матч начался после «" << button.kind << "» ==" << std::endl;
                    in_battle = true;
                    coop_flow = false;
                }
                // Ждали и не дождались: сервер не находит соперника.
                ++pvp_waits;
                if (pvp_waits >= 2) {
                    pvp_fails += 1;
                    pvp_waits = 0;
                    std::cout << "    соперник не нашёлся, переключаюсь на Co-Op" << std::endl;
                }
            } else {
                sleep_ms(2500);
            }
            const cv::Mat after = game.grab_canvas();
            bool changed = after.empty();
            if (!changed) {
                cv::Mat diff;
                cv::absdiff(before, after, diff);
                const cv::Scalar mean = cv::mean(diff);
                changed = mean[0] > 1.5;
            }
            if (button.kind == "cancel" || (awaiting_pvp && button.kind == "ok")) {
                if (!coop_mode) {
                    // В Co-Op соперника нет, и «Отмена» ничего не отменяет:
                    // это кнопка «отмена выбора модификатора». Считать её
                    // неудачей PvP бессмысленно — так бот уходил в PvP.
                    pvp_fails += 1;
                    awaiting_pvp = false;
                    std::cout << "    PvP не работает, переключаюсь на Co-Op" << std::endl;
                }
            }
            std::cout << "[меню " << step << "] «" << button.kind << "» "
                      << (changed ? "ок" : "без эффекта") << std::endl;
            if (changed) {
                if (button.kind == "pvp") awaiting_pvp = true;
                // Нажали «Co-Op» — дальше идём по экранам Co-Op, «Назад» там
                // жать нельзя.
                if (button.kind == "coop") coop_flow = true;
                // Кнопка окна: после неё «Назад» жать нельзя (окно «выйти из
                // игры?» выскочит снова).
                if (button.kind == "ok") dialog_clicked = 8;
                clicked = true;
                any_effect = true;
                break;
            }
            dead[button.kind] = step + 4;
            ++stuck;
        }
        if (!any_effect && !clicked && !coop_flow) {
            // Экран не изменился: скорее всего, поверх висит поп-ап.
            // Снимаем его клавишей «Назад».
            if (back_presses < 3) {
                ++back_presses;
                game.press_back();
                sleep_ms(1200);
            }
        } else if (stuck >= 3 && !coop_flow) {
            // Клики не помогают: значит, мы попали в экран, из которого
            // бота вывести нечем («Подземелья», экран героя). Отходим назад.
            if (back_presses < 3) {
                ++back_presses;
                std::cout << "    экран не поддаётся, отходим назад" << std::endl;
                game.press_back();
                stuck = 0;
                sleep_ms(1500);
            } else {
                // Назад уже не помогает — возможно, игра закрылась. Вернём её.
                std::cout << "    экран не поддаётся, возвращаю игру" << std::endl;
                relaunch_game();
                back_presses = 0;
                stuck = 0;
            }
        }
        // Экраны, из которых нет кнопки выхода («Подземелья», карточка
        // героя): каждые несколько шагов отходим назад. На главном меню
        // «Назад» не жмём — там он закрыл бы игру (игра спрашивает «выйти
        // из игры?», и бот застревает в этом вопросе). И на экранах Co-Op
        // тоже: «Назад» там возвращает на предыдущий экран (этаж, союзник),
        // и бот топчется по кругу.
        bool main_menu = false;
        for (const rr::UiElement& button : candidates) {
            if (button.kind == "coop" || button.kind == "pvp" || button.kind == "orange") {
                main_menu = true;
            }
        }
        if (!coop_flow && !main_menu && !dialog_clicked && back_presses < 3 && step > 0 &&
            step % 5 == 4) {
            cv::Rect pvp_rect;
            if (!hud.find_text(canvas, "pvp", &pvp_rect)) {
                std::cout << "    отходим назад с экрана без кнопок" << std::endl;
                ++back_presses;
                game.press_back();
                sleep_ms(1500);
            }
        }
        if (!clicked) sleep_ms(1500);
        if (dialog_clicked > 0) --dialog_clicked;
    }
    if (!in_battle) {
        std::cerr << "не удалось дойти до боя" << std::endl;
        return false;
    }
    return true;
    };

    // 2) Основной цикл боя: зовём юнита, пока хватает энергии.
    const auto play_battle = [&](int round) {
    double last_summon = -10.0;
    // Слияние: ждём результата прошлой попытки и помним, с кем сливали.
    bool merge_pending = false;
    // С какого шага можно снова жать умение героя.
    int next_ability_step = 0;
    rr::Pawn merge_reference;
    int merge_same_before = 0;
    // Три неудачных слияния подряд — признак, что слияние сейчас не
    // работает: тогда бот какое-то время только зовёт юнитов.
    int merge_fail_streak = 0;
    int merge_pause_steps = 0;
    // Пары, которые не слились (обычно это юнит ранга 2 и юнит ранга 1 —
    // игра такие не принимает). Повторять их бессмысленно.
    //
    // Помним по клеткам сетки: координаты на кадре устаревают (камера едет),
    // а клетки — нет.
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> failed_pairs;
    // Клетки, с которых слияние уже не выходило: если с клетки не слилось
    // дважды, юнит на ней скорее всего уже ранга 2, и его больше не трогаем.
    // Ключ — клетка сетки, значение — сколько раз не вышло и на каком шаге
    // (неудача «стареет»: юнит мог погибнуть, и клетка снова годится).
    std::map<std::pair<int, int>, std::pair<int, int>> failed_cells;
    // Уже слитые юниты (ранг 2): их не сливают ни с кем — слияние второго
    // ранга с первым игра не принимает, а 2+2 теряет силу.
    //
    // Положение храним по сетке клеток: камера ходит, и абсолютные
    // координаты быстро устаревают.
    struct MergedCell {
        std::pair<int, int> cell;   // клетка сетки
        int life = 30;              // сколько шагов ещё помнить
    };
    std::vector<MergedCell> merged_cells;
    // Пара, которую сливали последний раз: её результат проверяем на
    // следующем шаге.
    cv::Point merge_source;
    cv::Point merge_target;
    // Положение своей половины арены в момент слияния: камера в бою ходит,
    // Пешки на момент слияния (без источника и цели): по ним оценивается
    // сдвиг камеры в следующем кадре (rr::estimate_pawn_shift).
    std::vector<rr::Pawn> merge_pawns;
    // Пара считается неудачной, если её клетки совпадают с одной из
    // неудачных (юниты стоят на месте, значит и пара та же).
    const auto is_failed_pair = [&](std::pair<int, int> from, std::pair<int, int> to) {
        for (const auto& failed : failed_pairs) {
            if (failed.first == from && failed.second == to) return true;
        }
        return false;
    };
    for (int step = 0; step < steps; ++step) {
        const cv::Mat canvas = game.grab_canvas();
        if (canvas.empty()) {
            std::cout << "кадр не получен" << std::endl;
            continue;
        }
        // Игра могла повернуться (портрет -> ландшафт): тогда размер кадра
        // меняется, и раскладку HUD надо пересчитать под него.
        update_layout(cv::Size(canvas.cols, canvas.rows));
        // Сначала проверяем, что бой ещё идёт: на экране итогов в области
        // энергии лежит совсем другое, и бот считал бы оттуда случайные
        // числа — и кликал бы в пустое место.
        if (!hud.in_battle(canvas)) {
            // Проверка «идёт ли бой» может сработать ложно: в начале боя по
            // полю идёт анимация (облака), HUD закрыт, и бот решает, что бой
            // кончился — и уходит в меню прямо во время боя. Поэтому
            // требуем несколько неудачных проверок подряд.
            bool really_over = true;
            for (int check = 0; check < 3; ++check) {
                sleep_ms(700);
                const cv::Mat again = game.grab_canvas();
                if (!again.empty() && hud.in_battle(again)) {
                    really_over = false;
                    break;
                }
            }
            if (!really_over) continue;
            std::cout << "== матч закончился ==" << std::endl;
            break;
        }
        double confidence = 0.0;
        const double mana = hud.read_mana(canvas, &confidence);
        const bool affordable = hud.summon_ready(canvas);
        // Состояние поля: оно нужно и для проверки прошлого слияния, и для
        // самого слияния (ниже разбор поля заполняет его).
        rr::GameState state;
        rr::BoardMapping mapping;
        // Пешки на поле: по ним ищем пары для слияния. Основания находятся
        // надёжнее, чем совпадения с эталонами юнитов, и не зависят от того,
        // какие карты в колоде.
        const rr::ArenaFrame arena = rr::find_arena(canvas);
        const std::vector<rr::Pawn> pawns = rr::find_pawns(canvas, arena);
        // Сетка клеток поля: по ней мы помним «клетку слитого юнита» и
        // «клетку, с которой слияние не вышло». Индексы клеток не зависят от
        // положения камеры, в отличие от координат относительно арены
        // (детектор арены гуляет на десятки пикселей).
        const rr::CellGrid grid = rr::estimate_grid(pawns);
        // Клетка пешки (если сетка оценена).
        const auto cell_of_pawn = [&](const cv::Point& pawn) {
            if (!grid.valid) return std::pair<int, int>{-1, -1};
            return grid.cell_of(pawn);
        };
        // Юнит уже слит (ранг 2)? Положение — по сетке клеток.
        //
        // Слитый юнит уходит вперёд, поэтому отслеживаем его: ищем пешку
        // рядом с запомненной клеткой и переносим пометку на неё. Если пешки
        // рядом нет — юнит погиб, пометку снимаем.
        for (MergedCell& merged : merged_cells) {
            const cv::Point absolute = grid.point_of(merged.cell);
            const rr::Pawn* nearest = nullptr;
            double best = 1e9;
            for (const rr::Pawn& pawn : pawns) {
                if (!pawn.ours) continue;
                const double distance = std::hypot(pawn.center.x - absolute.x,
                                                  pawn.center.y - absolute.y);
                if (distance < 50 && distance < best) {
                    best = distance;
                    nearest = &pawn;
                }
            }
            if (nearest != nullptr) {
                // Слитый юнит виден — пометка едет за ним.
                merged.cell = cell_of_pawn(nearest->center);
                merged.life = 30;
            } else if (--merged.life <= 0) {
                // Юнита рядом нет: он погиб или ушёл (а может, просто не
                // виден детектору — у слитых юнитов форма не круглая).
                merged.life = 0;
            }
        }
        merged_cells.erase(std::remove_if(merged_cells.begin(), merged_cells.end(),
                                           [](const MergedCell& merged) {
                                               return merged.life <= 0;
                                           }),
                            merged_cells.end());
        const auto is_merged = [&](const cv::Point& pawn) {
            const std::pair<int, int> cell = cell_of_pawn(pawn);
            for (const MergedCell& merged : merged_cells) {
                if (merged.cell == cell) return true;
            }
            return false;
        };
        // С клетки уже не слилось дважды — юнит там, скорее всего, ранга 2.
        const auto is_bad_cell = [&](const cv::Point& pawn) {
            const std::pair<int, int> cell = cell_of_pawn(pawn);
            const auto it = failed_cells.find(cell);
            if (it == failed_cells.end()) return false;
            // Давняя неудача ничего не значит: юнит мог погибнуть или уйти.
            if (step - it->second.second > 25) return false;
            return it->second.first >= 2;
        };
        int our_pawns = 0;
        for (const rr::Pawn& pawn : pawns) {
            if (pawn.ours) ++our_pawns;
        }
        // Занятость доски: сколько клеток высадки ещё свободно.
        //
        // Поле — сетка 5x3 (15 клеток). Если свободных клеток нет, призыв
        // всё равно не создаст юнита (виджет гаснет), и место надо освободить
        // слиянием — оно бесплатное.
        //
        // Занятость считается по клеткам (на клетке что-то лежит — она
        // занята), а не по найденным юнитам: слитые юниты детектор круглых
        // оснований не видит, и по ним доска выглядела бы свободной.
        const rr::BoardState board = rr::analyze_board(canvas, arena, pawns);
        std::cout << "[" << step << "] энергия=" << mana << " призыв=" << (affordable ? "да" : "нет")
                  << " пешек=" << our_pawns;
        if (board.valid) {
            std::cout << ", занято клеток " << board.occupied << "/" << board.capacity
                      << " (свободно " << board.free_cells << ")";
        }
        std::cout << std::endl;
        // Умение героя: круглая кнопка справа внизу. Жмём, когда откат
        // кончился (вокруг кнопки появляется жёлтое свечение).
        //
        // После нажатия умение уходит на откат, и кнопка гаснет сама. Но
        // между нажатием и этим бот успевает сделать ещё шаг, поэтому
        // жмём не чаще, чем раз в несколько шагов.
        if (!no_ability && step >= next_ability_step) {
            const cv::Point ability = hud.ability_button(canvas);
            if (ability.x >= 0 && hud.ability_ready(canvas)) {
                show_click(canvas, ability, "умение", g_click_index++);
                click_canvas(ability);
                // Пауза после нажатия: пока умение не ушло на откат, жать
                // снова нельзя (а у части героев нажатие открывает выбор
                // цели, и кнопка остаётся готовой).
                next_ability_step = step + 25;
                std::cout << "    жму умение героя" << std::endl;
            }
        }
        // Проверяем прошлое слияние: после слияния пешек такого вида должно
        // стать на одну меньше (слитый юнит выглядит иначе — с цифрой ранга).
        if (merge_pending) {
            // Результат слияния: пешка-источник исчезает, а на месте цели
            // остаётся юнит (уже другого ранга). Считаем и «похожих»
            // пешек: после слияния слитый юнит выглядит иначе, и в этот счёт
            // он не входит.
            //
            // Сдвиг камеры: за время перетаскивания поле уезжает (до 70 px),
            // значит, сместились и пешки. Сдвиг считаем ПО ПЕШКАМ, а не по
            // положению арены: детектор арены гуляет на десятки пикселей, и
            // из-за этого слияние, которое на самом деле вышло, считалось
            // неудачным (а «неудачная» пара попадала в чёрный список).
            const cv::Point shift = rr::estimate_pawn_shift(merge_pawns, pawns, 60);
            const int shift_x = shift.x;
            const int shift_y = shift.y;
            // Положения пешек с поправкой на ход камеры.
            const cv::Point merge_source_abs(merge_source.x + shift_x, merge_source.y + shift_y);
            const cv::Point merge_target_abs(merge_target.x + shift_x, merge_target.y + shift_y);
            // Эталон для подсчёта «похожих» берём с текущего кадра: камера
            // могла сдвинуть поле, и пешка-цель уже не на прежнем месте.
            rr::Pawn reference = merge_reference;
            for (const rr::Pawn& pawn : pawns) {
                if (!pawn.ours) continue;
                if (std::abs(pawn.center.x - merge_target_abs.x) < 30 &&
                    std::abs(pawn.center.y - merge_target_abs.y) < 30) {
                    reference = pawn;
                    break;
                }
            }
            const int same_now = rr::count_same_pawns(canvas, pawns, reference);
            bool source_gone = true;
            bool target_alive = false;
            for (const rr::Pawn& pawn : pawns) {
                if (!pawn.ours) continue;
                if (std::abs(pawn.center.x - merge_source_abs.x) < 30 &&
                    std::abs(pawn.center.y - merge_source_abs.y) < 30) {
                    source_gone = false;
                }
                if (std::abs(pawn.center.x - merge_target_abs.x) < 30 &&
                    std::abs(pawn.center.y - merge_target_abs.y) < 30) {
                    target_alive = true;
                }
            }
            // Слияние вышло, если пешка-источник исчезла: её больше нет на
            // поле (юнит поглощён). Проверять, что на месте цели стоит юнит,
            // нельзя: результат слияния — случайная карта из колоды, а у
            // некоторых юнитов форма не круглая, и детектор их не находит.
            if (source_gone) {
                std::cout << "    слияние вышло: пешек стало " << our_pawns << ", похожих было "
                          << merge_same_before << ", стало " << same_now
                          << (target_alive ? "" : " (на месте цели новый юнит — другая форма)")
                          << std::endl;
                merge_fail_streak = 0;
                failed_pairs.clear();
                failed_cells.clear();
                // Слитый юнит (ранг 2) — запоминаем его клетку: его больше
                // не сливают.
                merged_cells.push_back({cell_of_pawn(merge_target_abs), 30});
                if (merged_cells.size() > 12) merged_cells.erase(merged_cells.begin());
            } else {
                std::cout << "    слияние не вышло: пешек " << our_pawns << ", похожих было "
                          << merge_same_before << ", стало " << same_now << std::endl;
                ++merge_fail_streak;
                failed_pairs.emplace_back(cell_of_pawn(merge_source_abs),
                                          cell_of_pawn(merge_target_abs));
                if (failed_pairs.size() > 8) failed_pairs.erase(failed_pairs.begin());
                // Клетки-кандидаты: с них слияние уже не выходило.
                for (const cv::Point& cell : {merge_source_abs, merge_target_abs}) {
                    auto& entry = failed_cells[cell_of_pawn(cell)];
                    entry.first += 1;
                    entry.second = step;
                }
                if (failed_cells.size() > 24) failed_cells.erase(failed_cells.begin());
                if (merge_fail_streak >= 3) {
                    merge_pause_steps = 10;
                    merge_fail_streak = 0;
                    std::cout << "    слияний подряд не вышло — какое-то время только зовём юнитов"
                              << std::endl;
                }
            }
            merge_pending = false;
        }
        if (merge_pause_steps > 0) --merge_pause_steps;
        if (!record_dir.empty()) {
            // Кадры боя: из них собираются эталоны юнитов для мобильного
            // клиента (веб-клиентские не подходят).
            std::error_code ec;
            std::filesystem::create_directories(record_dir, ec);
            const std::string name = "b" + std::to_string(round) + "_" + std::to_string(step) + ".png";
            cv::imwrite((std::filesystem::path(record_dir) / name).string(), canvas);
        }
        if (vision && planner) {
            // Разбор поля: своих, чужих и кого выгоднее звать. Клик от
            // этого не зависит — виджет призыва сам выбирает юнита.
            double field_height = 640.0;
            if (backend_name == "desktop") {
                state = vision->analyze(canvas, rr::web_arena(), mana, confidence);
                field_height = rr::web_arena().board_size.height;
            } else {
                // Мобильный клиент: камера ходит, и зашитые углы поля быстро
                // устаревают — арену ищем по кадру. Канонический кадр
                // собирается по половинам арены и общему масштабу (по
                // основаниям юнитов): именно в такой геометрии собраны
                // эталоны юнитов.
                const cv::Mat board = rr::canonical_board(canvas, &mapping);
                if (!board.empty()) {
                    state = vision->analyze_board(board, mana, confidence);
                    field_height = board.rows;
                } else {
                    std::cout << "    арена не найдена" << std::endl;
                }
            }
            rr::GameState with_lanes = state;
            rr::add_deploy_lanes(&with_lanes);
            // Высота канонического поля — по ней планировщик решает, где
            // середина поля.
            planner->set_field_height(field_height);
            const rr::Action action = planner->plan(with_lanes);
            if (analyze_field) {
                std::cout << "    юнитов: своих " << with_lanes.ally_units.size() << ", чужих "
                          << with_lanes.enemy_units.size() << "; рекомендую: " << action.reason
                          << std::endl;
                // Найденные юниты: видно, какие из них можно слить.
                for (const rr::Detection& det : state.detections) {
                    std::cout << "      " << (det.team == rr::Team::Ally ? "свой " : "чужой ")
                              << det.name << " карточка=" << det.deck_position << " ("
                              << static_cast<int>(det.center.x) << ","
                              << static_cast<int>(det.center.y) << ")" << std::endl;
                }
            }
        }
        if (affordable || merge_pause_steps <= 0) {
            // Призываем часто: цена призыва растёт, но энергия восстанавливается
            // быстро, и промедление — это потерянные юниты на поле.
            // Слияние бесплатно, поэтому оно не зависит от энергии.
            static const auto start = std::chrono::steady_clock::now();
            const double now =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (now - last_summon >= 0.35) {
                bool merged = false;
                // Слияние — главное: два одинаковых юнита, поставленных рядом,
                // дают одного сильного. Ищем пару соседних одинаковых пешек
                // на своей половине и тянем одну на другую.
                if (!no_merge && merge_pause_steps <= 0) {
                    const std::vector<rr::MergePair> pairs = rr::find_merge_pairs(canvas, pawns);
                    // Сначала соседние пешки: слияние через всю доску может
                    // не сработать (или это просто неудобно). Если соседних
                    // слияний не выходит — пробуем и через всю доску.
                    //
                    // Пара, которая не сработала, в blacklist: скорее всего,
                    // это юнит ранга 2 и юнит ранга 1, а игра такие не
                    // сливает. Повторять одно и то же бессмысленно.
                    //
                    // Пороги подобраны замером живых боёв
                    // (tools/merge_failure_analysis.py): у неудачных пар
                    // пешки чаще разного размера (разный ранг) и портреты
                    // похожи слабее.
                    const auto good = [&](const rr::MergePair& pair, double min_score,
                                          double max_size_diff) {
                        if (pair.score < min_score) return false;
                        if (pair.size_diff > max_size_diff) return false;
                        // Разные ранги игра не сливает («ранг 1 + ранг 2» —
                        // отказ), такие пары не трогаем вовсе.
                        if (!pair.same_rank) return false;
                        // Слитый юнит (ранг 2 и выше) не сливаем: слияние
                        // второго ранга с первым не принимается, а «2 + 2»
                        // теряет силу.
                        if (pair.rank_source > 1) return false;
                        if (is_merged(pawns[pair.source].center)) return false;
                        if (is_merged(pawns[pair.target].center)) return false;
                        if (is_bad_cell(pawns[pair.source].center)) return false;
                        if (is_bad_cell(pawns[pair.target].center)) return false;
                        return !is_failed_pair(cell_of_pawn(pawns[pair.source].center),
                                               cell_of_pawn(pawns[pair.target].center));
                    };
                    // Выбор пары: берём ту, у которой выше оценка успеха
                    // (rr::merge_success_estimate — таблица, собранная замерами
                    // живых боёв). Оценка учитывает и похожесть портретов, и
                    // ранг (по форме основания).
                    //
                    // Про ранг: ранг на поле виден — это форма подставки
                    // (круг — ранг 1, капля — ранг 2, щит — ранг 3;
                    // notes/ЗАДАЧА1_РАНГ.md, исправление 24.10). Игра сливает
                    // только юнитов одного ранга, поэтому пары разных рангов
                    // бот не трогает, а слитые юниты (ранг 2 и выше) не
                    // сливает вовсе: с первым рангом игра не примет, а «2 + 2»
                    // теряет силу.
                    //
                    // Пороги (похожесть и размер) — ослабляются постепенно,
                    // чтобы бот не встал на забитой доске.
                    const auto pick = [&](double min_score, double max_size_diff,
                                          bool adjacent_only) {
                        const rr::MergePair* best = nullptr;
                        double best_estimate = -1.0;
                        for (const rr::MergePair& pair : pairs) {
                            if (!good(pair, min_score, max_size_diff)) continue;
                            if (adjacent_only && !pair.adjacent) continue;
                            const double estimate = rr::merge_success_estimate(pair);
                            if (estimate <= best_estimate) continue;
                            best_estimate = estimate;
                            best = &pair;
                        }
                        return best;
                    };
                    auto best = pick(rr::kMergeTryStrict, rr::kMergeSizeMax, true);
                    // Если строгих пар нет — ослабляем условия: сначала
                    // похожесть, потом размер пешек.
                    if (best == nullptr) {
                        best = pick(rr::kMergeTryMin, rr::kMergeSizeMax, true);
                    }
                    if (best == nullptr) {
                        best = pick(rr::kMergeTryMin - 0.06, rr::kMergeSizeMax, true);
                    }
                    // Доска почти забита: без слияния новый юнит не встанет
                    // (виджет призыва гаснет), поэтому берём любую пару,
                    // которую вообще можно слить.
                    const bool board_tight = board.valid ? board.free_cells <= 2
                                                          : our_pawns >= 10;
                    if (best == nullptr && board_tight) {
                        best = pick(rr::kMergeTryFloor, rr::kMergeSizeRelaxed, true);
                    }
                    if (best == nullptr && board_tight) {
                        // Совсем ничего нет — ослабляем сходство портретов
                        // (тип карты может отличаться), но ранг оставляем:
                        // разные ранги игра не сливает.
                        best = pick(rr::kMergeTryFloor, rr::kMergeSizeRelaxed, false);
                    }
                    // Если соседних пар нет (или все не вышли) — пробуем через
                    // всю доску, но только с очень похожими портретами.
                    if (best == nullptr && merge_fail_streak >= 2) {
                        // Соседних пар не нашлось — пробуем через всю доску, но
                        // только с очень похожими портретами.
                        best = pick(0.9, rr::kMergeSizeRelaxed, false);
                    }
                    if (best != nullptr) {
                        const rr::MergePair& pair = *best;
                        const cv::Point from = pawns[pair.source].center;
                        const cv::Point to = pawns[pair.target].center;
                        show_click(canvas, from, "слияние", g_click_index++);
                        // Перетаскиваем, ведя курсор за целью: камера едет,
                        // и по старым координатам слияние попадает мимо.
                        drag_merge(game, pawns, from, to);
                        // Ранг пешек — по форме подставки: ранг 1 = круг,
                        // ранг 2 = капля, ранг 3 = щит. Игра сливает только
                        // юнитов одного ранга.
                        const double fill_src = rr::pedestal_fill(canvas, pawns[pair.source]);
                        const double fill_dst = rr::pedestal_fill(canvas, pawns[pair.target]);
                        std::cout << "    сливаю: (" << from.x << "," << from.y << ") на (" << to.x
                                  << "," << to.y << "), похожесть " << pair.score
                                  << ", разница размеров " << static_cast<int>(pair.size_diff * 100)
                                  << " %, основания " << static_cast<int>(fill_src * 100) << "/"
                                  << static_cast<int>(fill_dst * 100)
                                  << " % (ранг " << pair.rank_source << "/" << pair.rank_target
                                  << ")"
                                  << ", оценка успеха "
                                  << static_cast<int>(rr::merge_success_estimate(pair) * 100) << " %"
                                  << ", сдвиг (" << to.x - from.x << "," << to.y - from.y << ")"
                                  << std::endl;
                        merge_pending = true;
                        merge_source = from;
                        merge_target = to;
                        merge_reference = pawns[pair.target];
                        merge_same_before = rr::count_same_pawns(canvas, pawns, merge_reference);
                        // Пешки для оценки сдвига камеры: источник и цель
                        // исключаем — они двигаются сами (источник едет за
                        // курсором), и в сдвиг камеры попадать не должны.
                        merge_pawns.clear();
                        for (int i = 0; i < static_cast<int>(pawns.size()); ++i) {
                            if (i == pair.source || i == pair.target) continue;
                            merge_pawns.push_back(pawns[i]);
                        }
                        merged = true;
                    }
                }
                // Звать юнита имеет смысл, только если есть свободная клетка:
                // на забитой доске призыв всё равно не создаст юнита (виджет
                // гаснет), а место надо освободить слиянием.
                const bool has_room = !board.valid || board.free_cells > 0;
                if (!merged && affordable && !no_summon && has_room) {
                    const cv::Point button = hud.summon_button(canvas);
                    show_click(canvas, button, "призыв", g_click_index++);
                    click_canvas(button);
                }
                last_summon = now;
                if (debug) {
                    cv::imwrite(out_dir + "/step_" + std::to_string(step) + ".png", canvas);
                }
            }
        }
        sleep_ms(250);
    }
    };

    // Раунды: бой, выход в меню, снова бой. Так бот играет без остановки.
    for (int round = 0; round < rounds; ++round) {
        std::cout << "== раунд " << (round + 1) << "/" << rounds << " ==" << std::endl;
        try {
            if (!go_to_battle()) {
                return round == 0 ? 1 : 0;
            }
            play_battle(round);
        } catch (const cv::Exception& error) {
            // Редкий случай: OpenCV спотыкается о кадр (например, о пустой
            // или не того размера маску). Бот не должен из-за этого падать:
            // пропускаем раунд и идём дальше.
            std::cerr << "исключение в бою: " << error.what() << std::endl;
            sleep_ms(1000);
        }
        if (round + 1 < rounds) {
            // Выходим с экрана итогов: «Забрать», «Продолжить» и прочее
            // разбирает go_to_battle() в следующем раунде.
            sleep_ms(2000);
        }
    }

    return 0;
}
