# Исследовательский бот Rush Royale — версия на C++.
#
# Сборка:  make
# Запуск:  ./build/rrbot --steps 40
#
# Зависимости: OpenCV (opencv5), libevdev.

CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter
PKG_CFLAGS := $(shell pkg-config --cflags opencv5)
# Только нужные модули OpenCV: полный набор тянет за собой viz, а ему нужен
# VTK, которого в системе нет. geometry нужен для перспективной коррекции
# поля (getPerspectiveTransform).
OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_geometry
LDLIBS     := $(OPENCV_LIBS) -levdev
# Зависимости от заголовков: без них смена common.hpp оставляет часть
# объектных файлов со старой структурой — и бот падает с порчей памяти.
DEPFLAGS := -MMD -MP

SRC := $(wildcard src/*.cpp)
OBJ := $(patsubst src/%.cpp,build/%.o,$(SRC))
DEP := $(OBJ:.o=.d)
BIN := build/rrbot

# Утилита замера точности детекции — отдельный бинарь, чтобы не мешать боту.
EVAL_SRC := eval/evaluate_vision.cpp
EVAL_OBJ := build/eval_evaluate_vision.o build/vision.o build/planner.o
EVAL_BIN := build/rrvision

# Демонстрация планировщика: печатает решение по сценариям (для тестов).
PLAN_OBJ := build/eval_plan_demo.o build/vision.o build/planner.o
PLAN_BIN := build/rrplan

# Проверочные программы: пешки и пары для слияния, поиск арены.
MERGE_OBJ := build/eval_merge_probe.o build/merge.o build/arena.o
MERGE_BIN := build/merge_probe
# Занятость доски: сколько клеток высадки занято.
BOARD_OBJ := build/eval_board_probe.o build/board.o build/merge.o build/arena.o
BOARD_BIN := build/board_probe
# Оценка вероятности успеха слияния (таблица, собранная замерами).
ESTIMATE_OBJ := build/eval_merge_estimate.o build/merge.o build/arena.o
ESTIMATE_BIN := build/merge_estimate
ARENA_OBJ := build/eval_arena_probe.o build/arena.o
ARENA_BIN := build/arena_probe
# Проба жеста колоды: тянем карточку из колоды на клетку и смотрим, что вышло.
# Линкуется со всеми объектами бота, кроме main.o (своей точки входа у неё нет).
DECK_OBJ := build/eval_deck_probe.o $(filter-out build/main.o,$(OBJ))
DECK_BIN := build/deck_probe

# Зависимости от заголовков подключаются для всех объектов, включая
# проверочные программы: без этого они собираются под старые заголовки и
# падают (уже было: бинарь планировщика печатал пустые оценки).
ALL_OBJ := $(OBJ) $(EVAL_OBJ) $(PLAN_OBJ) $(MERGE_OBJ) $(ARENA_OBJ) $(ESTIMATE_OBJ) $(BOARD_OBJ) build/eval_deck_probe.o
DEP := $(ALL_OBJ:.o=.d)

all: $(BIN) $(EVAL_BIN) $(PLAN_BIN) $(MERGE_BIN) $(ARENA_BIN) $(ESTIMATE_BIN) $(BOARD_BIN) $(DECK_BIN)

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(PKG_CFLAGS) -Isrc -c $< -o $@

build/eval_%.o: eval/%.cpp | build
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(PKG_CFLAGS) -Isrc -c $< -o $@

build:
	mkdir -p build

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(EVAL_BIN): $(EVAL_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(PLAN_BIN): $(PLAN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(MERGE_BIN): $(MERGE_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(ARENA_BIN): $(ARENA_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(ESTIMATE_BIN): $(ESTIMATE_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BOARD_BIN): $(BOARD_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(DECK_BIN): $(DECK_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -rf build

-include $(DEP)

.PHONY: all clean
