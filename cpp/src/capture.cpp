#include "capture.hpp"

#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace rr {

bool grab_screen(int x, int y, int w, int h, cv::Mat* out) {
    if (w <= 0 || h <= 0) return false;
    static int counter = 0;
    const std::string path = "/tmp/rrbot-grim-" + std::to_string(::getpid()) + "-" +
                             std::to_string(counter++) + ".png";
    const std::string cmd = "grim -g \"" + std::to_string(x) + "," + std::to_string(y) + " " +
                            std::to_string(w) + "x" + std::to_string(h) + "\" " + path +
                            " 2>/dev/null";
    const int code = std::system(cmd.c_str());
    if (code != 0) return false;
    cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
    ::remove(path.c_str());
    if (frame.empty()) return false;
    *out = std::move(frame);
    return true;
}

bool find_canvas_rect(const cv::Mat& frame, cv::Rect* out) {
    if (frame.empty()) return false;
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    // Порог по яркости низкий: на затемнённых экранах рамка почти
    // чернеет, но оттенок и насыщенность сохраняются.
    cv::inRange(hsv, cv::Scalar(98, 150, 10), cv::Scalar(116, 255, 255), mask);

    const int height = mask.rows;
    const int width = mask.cols;
    std::vector<double> rows(height, 0.0);
    std::vector<double> cols(width, 0.0);
    for (int y = 0; y < height; ++y) {
        double sum = 0.0;
        const auto* row = mask.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            if (row[x]) {
                sum += 1.0;
                cols[x] += 1.0;
            }
        }
        rows[y] = sum;
    }
    double row_max = 0.0;
    double col_max = 0.0;
    for (double v : rows) row_max = std::max(row_max, v);
    for (double v : cols) col_max = std::max(col_max, v);
    if (row_max < width * 0.4 || col_max < height * 0.15) return false;

    const double row_threshold = row_max * 0.8;
    const double col_threshold = col_max * 0.3;
    int top = -1;
    int bottom = -1;
    int left = -1;
    int right = -1;
    for (int y = 0; y < height; ++y) {
        if (rows[y] >= row_threshold) {
            if (top < 0) top = y;
            bottom = y;
        }
    }
    for (int x = 0; x < width; ++x) {
        if (cols[x] >= col_threshold) {
            if (left < 0) left = x;
            right = x;
        }
    }
    if (top < 0 || bottom < 0 || left < 0 || right < 0) return false;
    if (bottom - top < height * 0.5 || right - left < width * 0.3) return false;
    *out = cv::Rect(left, top, right - left + 1, bottom - top + 1);
    return true;
}

}  // namespace rr
