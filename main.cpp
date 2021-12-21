
#include "fpds.hpp"
#include <iostream>

int main() {
    std::vector<fpds::vec3> points = fpds::fast_poisson_disk_3d(fpds::vec3(500, 500, 500), 25.0f);
    std::cout << "generated " << points.size() << " samples." << std::endl;

    return 0;
}
