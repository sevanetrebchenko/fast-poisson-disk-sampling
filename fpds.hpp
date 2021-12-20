
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
        vec2() : x(0.0f), y(0.0f) {
        }

        vec2(float x, float y) : x(x), y(y) {
        }

        [[nodiscard]] float operator[](int index) const {
            if (index < 0 || index >= 2) {
                throw std::out_of_range("index " + std::to_string(index) + " is out of range.");
            }
            return index == 0 ? x : y;
        }

        float x;
        float y;
    };

    struct vec3 {
        [[nodiscard]] float operator[](int index) const {
            if (index < 0 || index >= 3) {
                throw std::out_of_range("index " + std::to_string(index) + " is out of range.");
            }
            return index == 0 ? x : index == 1 ? y : z;
        }

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
        int sample_index = 0;
        vec2 grid_sample = convert_to_grid_space(sample, cell_size);
        grid[static_cast<int>(grid_sample.x) + grid_width * static_cast<int>(grid_sample.y)] = sample_index;
        active_list.emplace_back(sample_index);

        ++sample_index;

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
                        int current_sample_index = grid[row + grid_width * col];

                        if (current_sample_index != NO_SAMPLE) {
                            // Grid stores indices into points_list.
                            sample = point_list[current_sample_index];

                            // Ensure separation between current and test point is at least 'r'.
                            float distance = (sample.x - test_point.x) * (sample.x - test_point.x) + (sample.y - test_point.y) * (sample.y - test_point.y);
                            if (distance < r * r) {
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
                    grid[static_cast<int>(grid_test_point.x) + grid_width * static_cast<int>(grid_test_point.y)] = sample_index;
                    active_list.emplace_back(sample_index);

                    ++sample_index;

                    found = true;
                    break;
                }
            }

            if (!found && !active_list.empty()) {
                // 'k' attempts passed and no point found, remove index from sample list.
                active_list.erase(active_list.begin() + index);
            }
        }

        return point_list;
    }

}