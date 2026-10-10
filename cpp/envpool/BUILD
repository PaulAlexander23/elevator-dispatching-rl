# Bazel targets for the elevator env inside an EnvPool checkout. Used by
# scripts/build_envpool.sh, which copies this directory to envpool/elevator/
# and the C++ port (cpp/include, cpp/src) to envpool/elevator/cpp/.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "elevator_core",
    srcs = ["cpp/src/building.cpp"],
    hdrs = ["cpp/include/elevator/building.hpp"],
    # No fused multiply-add: the kinematics must round exactly as Python does.
    copts = ["-ffp-contract=off"],
    includes = ["cpp/include"],
)

cc_library(
    name = "elevator_envpool_h",
    hdrs = ["elevator_envpool.h"],
    deps = [
        ":elevator_core",
        "//envpool/core:async_envpool",
        "//envpool/core:env",
        "//envpool/core:env_spec",
    ],
)

pybind_extension(
    name = "elevator_envpool",
    srcs = ["elevator_envpool.cc"],
    linkopts = ["-ldl"],
    deps = [
        ":elevator_envpool_h",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "elevator",
    srcs = ["__init__.py"],
    data = [":elevator_envpool"],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "elevator_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)
