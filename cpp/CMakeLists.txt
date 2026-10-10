cmake_minimum_required(VERSION 3.20)
project(elevator_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(elevator STATIC src/building.cpp)
target_include_directories(elevator PUBLIC include)
set_target_properties(elevator PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(elevator PRIVATE -Wall -Wextra)
  # No fused multiply-add: the kinematics must round exactly as Python does.
  target_compile_options(elevator PUBLIC -ffp-contract=off)
endif()

add_executable(elevator_replay tools/replay.cpp)
target_link_libraries(elevator_replay PRIVATE elevator)

add_executable(elevator_bench tools/bench.cpp)
target_link_libraries(elevator_bench PRIVATE elevator)

enable_testing()
add_executable(elevator_tests tests/test_building.cpp)
target_link_libraries(elevator_tests PRIVATE elevator)
add_test(NAME elevator_tests COMMAND elevator_tests)
