
#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <random>
#include <iostream>

#define PI 3.1415926535897932384626433f
#define NO_SAMPLE -1

namespace fpds {

    struct vec2 {
        vec2() : x(0.0f), y(0.0f) { }
        vec2(float x, float y) : x(x), y(y) { }

        float x;
        float y;
    };

    struct vec3 {
        vec3() : x(0.0f), y(0.0f), z(0.0f) { }
        vec3(float x, float y, float z) : x(x), y(y), z(z) { }

        float x;
        float y;
        float z;
    };

    [[nodiscard]] int uniform_int_distribution(int min, int max) {
        static std::random_device device;
        static std::default_random_engine generator(device());
        std::uniform_int_distribution<int> distribution(min, max);

        return distribution(generator);
    }

    [[nodiscard]] float uniform_real_distribution(float min, float max) {
        static std::random_device device;
        static std::default_random_engine generator(device());

        std::uniform_real_distribution<float> distribution(min, max);

        return distribution(generator);
    }

    struct grid {
        grid(vec2 dimensions, float r) : cell_size_(r / sqrtf(2.0f)),
                                         grid_width_(static_cast<int>(std::ceil(dimensions.x / cell_size_))),
                                         grid_height_(static_cast<int>(std::ceil(dimensions.y / cell_size_))),
                                         grid_size_(grid_width_ * grid_height_),
                                         grid_(grid_width_ * grid_height_),
                                         sample_(0) {
            for (int i = 0; i < grid_size_; ++i) {
                grid_[i] = NO_SAMPLE;
            }
        }

        void register_point(vec2 world_coordinates) {
            vec2 grid_coordinates = convert_to_grid_coordinates(world_coordinates);
            grid_[index(grid_coordinates)] = sample_;
        }

        [[nodiscard]] vec2 convert_to_grid_coordinates(const vec2& coordinate) const {
            return { std::floor(coordinate.x / cell_size_), std::floor(coordinate.y / cell_size_) };
        }

        [[nodiscard]] int index(vec2 grid_coordinates) const {
            return static_cast<int>(grid_coordinates.x) + grid_width_ * static_cast<int>(grid_coordinates.y);
        }

        float cell_size_;
        int grid_width_;
        int grid_height_;
        int grid_size_;

        std::vector<int> grid_;
        int sample_;
    };



    [[nodiscard]] vec2 convert_to_grid_space(const vec2& coordinate, float cell_size) {
        return { std::floor(coordinate.x / cell_size), std::floor(coordinate.y / cell_size) };
    }

    // 'r' - minimum distance to be maintained between final samples.
    // 'k' - limit of samples to try before sample rejection (defaulted at 30, provided by the paper).
    [[nodiscard]] std::vector<vec2> fast_poisson_disk_2d(vec2 dimensions, float r, int k = 30) {
        float cell_size = r / sqrtf(2.0f);
        int grid_width = static_cast<int>(std::ceil(dimensions.x / cell_size));
        int grid_height = static_cast<int>(std::ceil(dimensions.y / cell_size));
        int grid_size = grid_width * grid_height;

        // Construct grid (n-dimensional array of integers).
        std::vector<int> grid(grid_size);
        for (int i = 0; i < grid_size; ++i) {
            grid[i] = NO_SAMPLE;
        }

        std::vector<int> active_list;
        std::vector<vec2> point_list;

        // Generate initial sample, randomly chosen uniformly from the given domain.
        // In world coordinate space.
        vec2 sample = vec2(uniform_real_distribution(0.0f, dimensions.x), uniform_real_distribution(0.0f, dimensions.y));
        point_list.emplace_back(sample);

        // Insert sample into background grid.
        // Initial sample has index 0.
        int counter = 0;
        vec2 grid_sample = convert_to_grid_space(sample, cell_size);
        grid[static_cast<int>(grid_sample.x) + grid_width * static_cast<int>(grid_sample.y)] = counter;
        active_list.emplace_back(counter);

        ++counter;

        while (!active_list.empty()) {
            // Choose random index from active list.
            int index = uniform_int_distribution(0, (int)active_list.size() - 1);
            sample = point_list[index];

            bool found = false;

            // Generate up to k points between 'r' and '2r' distance away from the randomly chosen point.
            for (int i = 0; i < k; ++i) {
                float radians = uniform_real_distribution(0.0f, 2.0f * PI);
                float radius = uniform_real_distribution(r, 2.0f * r);

                vec2 test_point (sample.x + radius * cosf(radians), sample.y + radius * sinf(radius));

                if (test_point.x < 0.0f || test_point.x >= dimensions.x) {
                    continue;
                }

                if (test_point.y < 0.0f || test_point.y >= dimensions.y) {
                    continue;
                }

                vec2 grid_test_point = convert_to_grid_space(test_point, cell_size);

                // Check to make sure point doesn't exist in this cell.
                int current_sample_index = grid[static_cast<int>(grid_test_point.x) + grid_width * static_cast<int>(grid_test_point.y)];
                if (current_sample_index != NO_SAMPLE) {
                    continue;
                }

                bool valid = true;

                // Check grid cells directly adjacent to the test cell.
                for (int y = -1; y < 2; ++y) {
                    bool break_out = false;

                    for (int x = -1; x < 2; ++x) {
                        if (x == 0 && y == 0) {
                            continue;
                        }

                        int row = static_cast<int>(grid_test_point.x) + x;
                        int col = static_cast<int>(grid_test_point.y) + y;

                        // Ensure valid bounds.
                        if (row < 0 || row >= grid_width) {
                            continue;
                        }

                        if (col < 0 || col >= grid_height) {
                            continue;
                        }

                        // Offset from current point.
                        current_sample_index = grid[row + grid_width * col];

                        if (current_sample_index != NO_SAMPLE) {
                            // Grid stores indices into points_list.
                            vec2 t = point_list[current_sample_index];

                            // Ensure separation between current and test point is at least 'r'.
                            float distance = sqrtf((t.x - test_point.x) * (t.x - test_point.x) + (t.y - test_point.y) * (t.y - test_point.y));
                            if (distance < r) {
                                valid = false;
                                break_out = true;
                                break;
                            }
                        }
                    }

                    if (break_out) {
                        break;
                    }
                }

                if (valid) {
                    // Found valid sample, append to list.
                    point_list.emplace_back(test_point);

                    // Mark point in grid.
                    grid[static_cast<int>(grid_test_point.x) + grid_width * static_cast<int>(grid_test_point.y)] = counter;
                    active_list.emplace_back(counter);

                    ++counter;

                    found = true;
                    break;
                }
            }

            if (!found) {
                // 'k' attempts passed and no point found, remove index from sample list.
                active_list.erase(active_list.begin() + index);
            }
        }

        return point_list;
    }

}