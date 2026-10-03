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
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)
find_package(Threads REQUIRED)

# ROS 1 (catkin) or ROS 2 (ament), as the other OpenVINS packages; neither: the
# core library and maploc_bench only. catkin_package() must come before the
# targets so that they land in the devel space.
find_package(catkin QUIET COMPONENTS roscpp std_msgs geometry_msgs sensor_msgs tf2_ros cv_bridge)
if (catkin_FOUND)
    catkin_package()
else ()
    find_package(ament_cmake QUIET)
endif ()

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
set(MAPLOC_TOOLS maploc_bench)

# the map builder (no ROS): needs Ceres for the bundle adjustment
find_package(Ceres QUIET)
if (Ceres_FOUND)
    add_executable(maploc_build src/build_map.cpp)
    target_include_directories(maploc_build PRIVATE ${CERES_INCLUDE_DIRS})
    target_link_libraries(maploc_build ov_maploc_core ${CERES_LIBRARIES} ${OpenCV_LIBS})
    list(APPEND MAPLOC_TOOLS maploc_build)
else ()
    message(STATUS "Ceres not found: no maploc_build")
endif ()

if (catkin_FOUND)
    add_executable(maploc_node src/ros1_node.cpp)
    target_include_directories(maploc_node PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(maploc_node ov_maploc_core ${catkin_LIBRARIES})

    install(TARGETS maploc_node ${MAPLOC_TOOLS} RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
    install(PROGRAMS scripts/map_npz_to_bin.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
    install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
elseif (ament_cmake_FOUND)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(cv_bridge REQUIRED)

    add_executable(maploc_node src/ros_node.cpp)
    target_link_libraries(maploc_node ov_maploc_core)
    ament_target_dependencies(maploc_node rclcpp std_msgs geometry_msgs sensor_msgs tf2_ros cv_bridge)

    install(TARGETS maploc_node ${MAPLOC_TOOLS} DESTINATION lib/${PROJECT_NAME})
    install(PROGRAMS scripts/map_npz_to_bin.py DESTINATION lib/${PROJECT_NAME})
    ament_package()
else ()
    message(STATUS "neither catkin nor ament_cmake found: building the core library and maploc_bench only")
    install(TARGETS ${MAPLOC_TOOLS} DESTINATION bin)
endif ()
