
#include "fpds.hpp"
#include <iostream>

int main() {
    std::vector<fpds::vec2> points = fpds::fast_poisson_disk_2d(fpds::vec2(500, 500), 50.0f);
    return 0;
}
