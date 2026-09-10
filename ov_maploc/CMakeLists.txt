cmake_minimum_required(VERSION 3.10)
project(ov_maploc)

# C++17: Eigen fixed-size members inside STL containers need aligned new
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
find_package(Threads REQUIRED)

# ROS-free core: map, rig, ORB, pose solver, localizer
add_library(ov_maploc_core STATIC
        src/map.cpp
        src/rig.cpp
        src/orb_extractor.cpp
        src/pose_solver.cpp
        src/localizer.cpp)
target_include_directories(ov_maploc_core PUBLIC src ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ov_maploc_core PUBLIC ${OpenCV_LIBS} Threads::Threads)
set_target_properties(ov_maploc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# offline benchmark (no ROS), for measuring the target board
add_executable(maploc_bench src/bench.cpp)
target_link_libraries(maploc_bench ov_maploc_core)

find_package(ament_cmake QUIET)
if (ament_cmake_FOUND)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(cv_bridge REQUIRED)

    add_executable(maploc_node src/ros_node.cpp)
    target_link_libraries(maploc_node ov_maploc_core)
    ament_target_dependencies(maploc_node rclcpp std_msgs geometry_msgs sensor_msgs tf2_ros cv_bridge)

    install(TARGETS maploc_node maploc_bench DESTINATION lib/${PROJECT_NAME})
    install(PROGRAMS scripts/map_npz_to_bin.py DESTINATION lib/${PROJECT_NAME})
    ament_package()
else ()
    message(STATUS "ament_cmake not found: building the core library and maploc_bench only")
    install(TARGETS maploc_bench DESTINATION bin)
endif ()
