
cmake_minimum_required(VERSION 3.0)

# Project information.
project(fast-poisson-disk-sampling
        VERSION 1.0
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

# Build project sources.
add_executable(fast-poisson-disk-sampling
        "${PROJECT_SOURCE_DIR}/main.cpp"
        )



