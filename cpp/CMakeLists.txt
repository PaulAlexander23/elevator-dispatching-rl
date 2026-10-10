cmake_minimum_required(VERSION 3.20)
project(elevator_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(elevator STATIC src/building.cpp src/vec_env.cpp)
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

# The Python module elevator_rl._cpp, written into src/elevator_rl/ so the
# package finds it. Needs the project's virtualenv Python (for nanobind):
#   cmake -S cpp -B cpp/build -DPython_EXECUTABLE=.venv/bin/python
option(ELEVATOR_PYTHON "Build the elevator_rl._cpp Python module" ON)
if(ELEVATOR_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_cpp NB_STATIC python/module.cpp)
    target_link_libraries(_cpp PRIVATE elevator)
    set_target_properties(_cpp PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../src/elevator_rl")
  else()
    message(WARNING "nanobind not found for ${Python_EXECUTABLE}: skipping the Python module. "
                    "Pass -DPython_EXECUTABLE=$PWD/.venv/bin/python after `uv sync`.")
  endif()
endif()
