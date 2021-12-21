
#pragma once

#include <vector>
#include <cmath>
#include <random>

#define PI 3.1415926535897932384626433f
#define NO_SAMPLE -1

namespace fpds {

    // Utility functionality + helper classes.

    struct vec2 {
        vec2() : x(0.0f), y(0.0f) {}
        vec2(float x, float y) : x(x), y(y) {}

        float x;
        float y;
    };

    struct ivec2 {
        ivec2() : x(0), y(0) {}
        ivec2(int x, int y) : x(x), y(y) {}

        int x;
        int y;
    };

    struct vec3 {
        vec3() : x(0.0f), y(0.0f), z(0.0f) {}
        vec3(float x, float y, float z) : x(x), y(y), z(z) {}

        float x;
        float y;
        float z;
    };

    struct ivec3 {
        ivec3() : x(0), y(0), z(0) {}
        ivec3(int x, int y, int z) : x(x), y(y), z(z) {}

        int x;
        int y;
        int z;
    };

    [[nodiscard]] float distance2(const vec2& a, const vec2& b) {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    }

    [[nodiscard]] float distance2(const vec3& a, const vec3& b) {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    }

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
        grid(const vec2& dimensions, float separation_distance)
                : cell_size(separation_distance / sqrtf(2.0f)),
                  grid_width(static_cast<int>(std::ceil(dimensions.x / cell_size))),
                  grid_height(static_cast<int>(std::ceil(dimensions.y / cell_size))),
                  grid_depth(-1), // Unused for a 2-dimensional grid.
                  grid_size(grid_width * grid_height),
                  grid_data(grid_size) {
            for (int i = 0; i < grid_size; ++i) {
                grid_data[i] = NO_SAMPLE;
            }
        }

        grid(const vec3& dimensions, float separation_distance)
                : cell_size(separation_distance / sqrtf(2.0f)),
                  grid_width(static_cast<int>(std::ceil(dimensions.x / cell_size))),
                  grid_height(static_cast<int>(std::ceil(dimensions.y / cell_size))),
                  grid_depth(static_cast<int>(std::ceil(dimensions.z / cell_size))),
                  grid_size(grid_width * grid_height * grid_depth),
                  grid_data(grid_size) {
            for (int i = 0; i < grid_size; ++i) {
                grid_data[i] = NO_SAMPLE;
            }
        }

        // 2D index into flattened array.
        [[nodiscard]] int get(int x, int y) const {
            return grid_data[x + grid_width * y];
        }

        void set(int value, int x, int y) {
            grid_data[x + grid_width * y] = value;
        }

        // 3D index into flattened array.
        [[nodiscard]] int get(int x, int y, int z) const {
            return grid_data[x + grid_width * z + grid_width * grid_depth * y];
        }

        void set(int value, int x, int y, int z) {
            grid_data[x + grid_width * z + grid_width * grid_depth * y] = value;
        }

        [[nodiscard]] ivec2 convert_to_grid_coordinates(const vec2& world_coordinates) const {
            return { static_cast<int>(std::floor(world_coordinates.x / cell_size)),
                     static_cast<int>(std::floor(world_coordinates.y / cell_size)) };
        }

        [[nodiscard]] ivec3 convert_to_grid_coordinates(const vec3& world_coordinates) const {
            return { static_cast<int>(std::floor(world_coordinates.x / cell_size)),
                     static_cast<int>(std::floor(world_coordinates.y / cell_size)),
                     static_cast<int>(std::floor(world_coordinates.z / cell_size)) };
        }

        float cell_size;

        int grid_width;
        int grid_height;
        int grid_depth;
        int grid_size;

        std::vector<int> grid_data;
    };



    // Fast Poisson Disk Sampling algorithm, for 2D applications.
    // 'r' - minimum distance to be maintained between final point samples.
    // 'k' - limit of samples to try before sample rejection (defaulted at 30, provided by the paper).
    [[nodiscard]] std::vector<vec2> fast_poisson_disk_2d(vec2 dimensions, float r, int k = 30) {
        grid g { dimensions, r };

        std::vector<int> active_list;
        std::vector<vec2> point_list;

        // Generate initial sample, randomly chosen uniformly from the given domain.
        // Sample is in world coordinates.
        vec2 sample_world_coordinates = vec2(uniform_real_distribution(0.0f, dimensions.x),
                                             uniform_real_distribution(0.0f, dimensions.y));

        // Record sample in grid.
        int sample_index = 0;
        ivec2 sample_grid_coordinates = g.convert_to_grid_coordinates(sample_world_coordinates);
        g.set(sample_index, sample_grid_coordinates.x, sample_grid_coordinates.y);

        point_list.emplace_back(sample_world_coordinates);
        active_list.emplace_back(sample_index);

        ++sample_index;

        while (!active_list.empty()) {
            // Choose random index from active sample list.
            int index = uniform_int_distribution(0, (int) active_list.size() - 1);
            sample_world_coordinates = point_list[index];

            bool found_sample = false;

            // Try up to 'k' times to find a valid point.
            for (int i = 0; i < k; ++i) {
                // Uniformly generate test points between 'r' and '2r' distance away around the chosen point.
                float radians = uniform_real_distribution(0.0f, 2.0f * PI);
                float radius = uniform_real_distribution(r, 2.0f * r);

                vec2 test_sample_world_coordinates = vec2(sample_world_coordinates.x + radius * cosf(radians),
                                                          sample_world_coordinates.y + radius * sinf(radians));

                // Ensure offsetting point did not push it out of bounds.
                if (test_sample_world_coordinates.x < 0.0f || test_sample_world_coordinates.x >= dimensions.x) {
                    continue;
                }
                if (test_sample_world_coordinates.y < 0.0f || test_sample_world_coordinates.y >= dimensions.y) {
                    continue;
                }

                ivec2 test_sample_grid_coordinates = g.convert_to_grid_coordinates(test_sample_world_coordinates);

                // Don't override cells that already have samples in them.
                int test_sample_index = g.get(test_sample_grid_coordinates.x, test_sample_grid_coordinates.y);
                if (test_sample_index != NO_SAMPLE) {
                    continue;
                }

                bool valid_sample = true;

                // Check grid cells directly adjacent to the test cell to ensure the validity of the selected sample.
                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        if (x == 0 && y == 0) {
                            continue;
                        }

                        int x_offset = test_sample_grid_coordinates.x + x;
                        int y_offset = test_sample_grid_coordinates.y + y;

                        // Ensure desired offset into the grid is in range.
                        if (x_offset < 0 || x_offset >= g.grid_width) {
                            continue;
                        }
                        if (y_offset < 0 || y_offset >= g.grid_height) {
                            continue;
                        }

                        // Offset from current point.
                        test_sample_index = g.get(x_offset, y_offset);

                        if (test_sample_index != NO_SAMPLE) {
                            // Found existing sample in the checked grid cell.
                            // Selected sample may still be valid if the separation between the existing and selected
                            // samples is adequately far.
                            const vec2& existing_sample = point_list[test_sample_index];

                            // Ensure separation between current and test point is at least 'r'.
                            if (distance2(existing_sample, test_sample_world_coordinates) < r * r) {
                                valid_sample = false;
                                break;
                            }
                        }
                    }
                }

                if (valid_sample) {
                    // Record sample in grid.
                    g.set(sample_index, test_sample_grid_coordinates.x, test_sample_grid_coordinates.y);

                    point_list.emplace_back(test_sample_world_coordinates);
                    active_list.emplace_back(sample_index);

                    ++sample_index;

                    found_sample = true;
                    break;
                }
            }

            if (!found_sample) {
                // Failed to find a valid point position after 'k' attempts.
                // We can say, within a reasonable certainty, that no more points can fit around the chosen point.
                active_list.erase(active_list.begin() + index);
            }
        }

        return point_list;
    }



    // Fast Poisson Disk Sampling algorithm, for 3D applications.
    // 'r' - minimum distance to be maintained between final point samples.
    // 'k' - limit of samples to try before sample rejection (defaulted at 30, provided by the paper).
    [[nodiscard]] std::vector<vec3> fast_poisson_disk_3d(vec3 dimensions, float r, int k = 30) {
        grid g { dimensions, r };

        std::vector<int> active_list;
        std::vector<vec3> point_list;

        // Generate initial sample, randomly chosen uniformly from the given domain.
        // Sample is in world coordinates.
        vec3 sample_world_coordinates = vec3(uniform_real_distribution(0.0f, dimensions.x),
                                             uniform_real_distribution(0.0f, dimensions.y),
                                             uniform_real_distribution(0.0f, dimensions.z));

        // Record sample in grid.
        int sample_index = 0;
        ivec3 sample_grid_coordinates = g.convert_to_grid_coordinates(sample_world_coordinates);
        g.set(sample_index, sample_grid_coordinates.x, sample_grid_coordinates.y, sample_grid_coordinates.z);

        point_list.emplace_back(sample_world_coordinates);
        active_list.emplace_back(sample_index);

        ++sample_index;

        while (!active_list.empty()) {
            // Choose random index from active sample list.
            int index = uniform_int_distribution(0, (int) active_list.size() - 1);
            sample_world_coordinates = point_list[index];

            bool found_sample = false;

            // Try up to 'k' times to find a valid point.
            for (int i = 0; i < k; ++i) {
                // Uniformly generate test points between 'r' and '2r' distance away around the chosen point.
                float theta = uniform_real_distribution(0.0f, 2.0f * PI);
                float phi = uniform_real_distribution(0.0f, PI);
                float radius = uniform_real_distribution(r, 2.0f * r);

                vec3 test_sample_world_coordinates = vec3(sample_world_coordinates.x + radius * cosf(theta) * sinf(phi),
                                                          sample_world_coordinates.y + radius * sinf(theta) * sinf(phi),
                                                          sample_world_coordinates.z + radius * cosf(phi));

                // Ensure offsetting point did not push it out of bounds.
                if (test_sample_world_coordinates.x < 0.0f || test_sample_world_coordinates.x >= dimensions.x) {
                    continue;
                }
                if (test_sample_world_coordinates.y < 0.0f || test_sample_world_coordinates.y >= dimensions.y) {
                    continue;
                }
                if (test_sample_world_coordinates.z < 0.0f || test_sample_world_coordinates.z >= dimensions.z) {
                    continue;
                }

                ivec3 test_sample_grid_coordinates = g.convert_to_grid_coordinates(test_sample_world_coordinates);

                // Don't override cells that already have samples in them.
                int test_sample_index = g.get(test_sample_grid_coordinates.x,
                                              test_sample_grid_coordinates.y,
                                              test_sample_grid_coordinates.z);
                if (test_sample_index != NO_SAMPLE) {
                    continue;
                }

                bool valid_sample = true;

                // Check grid cells directly adjacent to the test cell to ensure the validity of the selected sample.
                for (int y = -1; y < 2; ++y) {
                    for (int x = -1; x < 2; ++x) {
                        for (int z = -1; z < 2; ++z) {
                            if (x == 0 && y == 0 && z == 0) {
                                continue;
                            }

                            int x_offset = test_sample_grid_coordinates.x + x;
                            int y_offset = test_sample_grid_coordinates.y + y;
                            int z_offset = test_sample_grid_coordinates.z + z;

                            // Ensure desired offset into the grid is in range.
                            if (x_offset < 0 || x_offset >= g.grid_width) {
                                continue;
                            }
                            if (y_offset < 0 || y_offset >= g.grid_height) {
                                continue;
                            }
                            if (z_offset < 0 || z_offset >= g.grid_depth) {
                                continue;
                            }

                            // Offset from current point.
                            test_sample_index = g.get(x_offset, y_offset, z_offset);

                            if (test_sample_index != NO_SAMPLE) {
                                // Found existing sample in the checked grid cell.
                                // Selected sample may still be valid if the separation between the existing and selected
                                // samples is adequately far.
                                const vec3& existing_sample = point_list[test_sample_index];

                                // Ensure separation between current and test point is at least 'r'.
                                if (distance2(existing_sample, test_sample_world_coordinates) < r * r) {
                                    valid_sample = false;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (valid_sample) {
                    // Record sample in grid.
                    g.set(sample_index, test_sample_grid_coordinates.x, test_sample_grid_coordinates.y, test_sample_grid_coordinates.z);

                    point_list.emplace_back(test_sample_world_coordinates);
                    active_list.emplace_back(sample_index);

                    ++sample_index;

                    found_sample = true;
                    break;
                }
            }

            if (!found_sample) {
                // Failed to find a valid point position after 'k' attempts.
                // We can say, within a reasonable certainty, that no more points can fit around the chosen point.
                active_list.erase(active_list.begin() + index);
            }
        }

        return point_list;
    }

}