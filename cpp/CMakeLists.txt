cmake_minimum_required(VERSION 3.20)
project(elevator_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The package's C++ is built by scikit-build-core during `uv sync` (or `pip
# install`), which sets SKBUILD: the Python module and the C library are
# installed into the wheel's elevator_rl/, and the tools and tests are skipped.
# A hand build (`cmake -S cpp -B cpp/build`) is for the tools and the C++
# tests; it leaves its modules in cpp/build/elevator_rl/, not in src/, so they
# can't shadow the installed ones.
set(ELEVATOR_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/elevator_rl")
if(SKBUILD)
  set(ELEVATOR_TOOLS_DEFAULT OFF)
else()
  set(ELEVATOR_TOOLS_DEFAULT ON)
endif()
option(ELEVATOR_TOOLS "Build the replay and benchmark tools and the C++ tests" ${ELEVATOR_TOOLS_DEFAULT})

add_library(elevator STATIC src/building.cpp src/vec_env.cpp)
target_include_directories(elevator PUBLIC include)
set_target_properties(elevator PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(elevator PRIVATE -Wall -Wextra)
  # No fused multiply-add: the kinematics must round exactly as Python does.
  target_compile_options(elevator PUBLIC -ffp-contract=off)
endif()

if(ELEVATOR_TOOLS)
  add_executable(elevator_replay tools/replay.cpp)
  target_link_libraries(elevator_replay PRIVATE elevator)

  add_executable(elevator_bench tools/bench.cpp)
  target_link_libraries(elevator_bench PRIVATE elevator)

  enable_testing()
  add_executable(elevator_tests tests/test_building.cpp)
  target_link_libraries(elevator_tests PRIVATE elevator)
  add_test(NAME elevator_tests COMMAND elevator_tests)
endif()

# The flat C API (capi/elevator_c.h) as a shared library, loaded from Python
# with ctypes by elevator_rl/ctypes_env.py. Written next to the nanobind
# module (ELEVATOR_OUTPUT_DIR); needs no Python at build time.
option(ELEVATOR_C_API "Build the libelevator_c shared library" ON)
if(ELEVATOR_C_API)
  add_library(elevator_c SHARED capi/elevator_c.cpp)
  target_link_libraries(elevator_c PRIVATE elevator)
  target_include_directories(elevator_c PUBLIC capi)
  # dllexport on Windows; on Linux and macOS, hide everything but ELEVATOR_API.
  target_compile_definitions(elevator_c PRIVATE ELEVATOR_C_BUILDING)
  set_target_properties(elevator_c PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # The .so (LIBRARY) on Linux, the .dll (RUNTIME) on Windows; the same
    # folder for every configuration of a multi-config generator (MSVC).
    LIBRARY_OUTPUT_DIRECTORY "${ELEVATOR_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${ELEVATOR_OUTPUT_DIR}")
  if(UNIX AND NOT APPLE)
    # The hidden preset covers only this target's own sources: without this,
    # every symbol of the static elevator library is exported too.
    target_link_options(elevator_c PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} config)
    set_target_properties(elevator_c PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${config} "${ELEVATOR_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_${config} "${ELEVATOR_OUTPUT_DIR}")
  endforeach()
  if(MSVC)
    # Link the C runtime statically, so the DLL doesn't depend on a
    # particular vcruntime being installed (see the pitfalls in the plan).
    set_target_properties(elevator_c elevator PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  if(SKBUILD)
    install(TARGETS elevator_c LIBRARY DESTINATION elevator_rl RUNTIME DESTINATION elevator_rl)
  endif()
endif()

# The Python module elevator_rl._cpp. scikit-build-core passes its Python and
# installs nanobind in the build environment; a hand build needs the project's
# virtualenv Python (for nanobind), or -DELEVATOR_PYTHON=OFF:
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
      LIBRARY_OUTPUT_DIRECTORY "${ELEVATOR_OUTPUT_DIR}")
    if(SKBUILD)
      install(TARGETS _cpp LIBRARY DESTINATION elevator_rl)
    endif()
  else()
    message(WARNING "nanobind not found for ${Python_EXECUTABLE}: skipping the Python module. "
                    "Pass -DPython_EXECUTABLE=$PWD/.venv/bin/python after `uv sync`.")
  endif()
endif()
