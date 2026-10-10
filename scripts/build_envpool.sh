#!/usr/bin/env bash
# Build an EnvPool wheel that contains the elevator env (cpp/envpool/).
#
# EnvPool only takes new envs inside its own Bazel tree, so this:
#   1. checks out sail-sg/envpool at a pinned commit (in cpp/build/envpool);
#   2. copies cpp/envpool/ to envpool/elevator/ and the C++ port next to it;
#   3. trims the package to EnvPool's core plus the elevator env, so the build
#      skips Atari, MuJoCo, Procgen and the rest (and their Qt/SWIG/Go needs);
#   4. builds the wheel with Bazel for Python 3.12 into cpp/build/envpool-dist/.
#
# Needs git, a C++17 compiler, Java and bazelisk (as `bazel`) on PATH.
# The first build downloads Bazel's dependencies and takes a few minutes.
#
#   scripts/build_envpool.sh
#   uv pip install --no-deps cpp/build/envpool-dist/envpool-*.whl
set -euo pipefail

ENVPOOL_REPO=https://github.com/sail-sg/envpool.git
ENVPOOL_COMMIT=9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3  # 2026-09-15, v1.2.7
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${ENVPOOL_DIR:-$ROOT/cpp/build/envpool}
DIST=$ROOT/cpp/build/envpool-dist
BAZEL=${BAZEL:-bazel}

if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --filter=blob:none "$ENVPOOL_REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin "$ENVPOOL_COMMIT"
# Discard the previous run's edits so the trims below apply to a clean tree.
git -C "$SRC" checkout --quiet --force "$ENVPOOL_COMMIT"
git -C "$SRC" clean --quiet -fd envpool

# 2. The elevator env, as EnvPool's other envs are laid out.
ENV_DIR=$SRC/envpool/elevator
mkdir -p "$ENV_DIR/cpp/src"
cp "$ROOT"/cpp/envpool/{BUILD,__init__.py,registration.py,elevator_envpool.h,elevator_envpool.cc} "$ENV_DIR"/
cp -r "$ROOT/cpp/include" "$ENV_DIR/cpp/"
cp "$ROOT/cpp/src/building.cpp" "$ENV_DIR/cpp/src/"

# 3. Trim the package to the core and the elevator env.
python3 - "$SRC" <<'EOF'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])


def edit(path, old, new, count=1):
    path = src / path
    text = path.read_text()
    found = len(re.findall(old, text, flags=re.S))
    if found != count:
        sys.exit(f"{path}: expected {count} match(es) of {old!r}, found {found}")
    path.write_text(re.sub(old, new, text, flags=re.S))


# Register only the elevator env.
(src / "envpool/entry.py").write_text(
    '"""Entry point for all envs\' registration."""\n\n'
    "import envpool.elevator.registration  # noqa: F401\n"
)
edit("envpool/BUILD", r'(name = "entry",.*?deps = \[).*?(\],)',
     r'\1"//envpool/elevator:elevator_registration"\2')
edit("envpool/BUILD", r'(":registration",\n).*?("//envpool/python",)',
     r'\1        "//envpool/elevator",\n        \2')
edit("envpool/BUILD", r'(        "//envpool/python",\n).*?(    \],\n\))',
     r"\1\2")
# Drop the data of envs that are no longer built.
edit("BUILD", r'    "//third_party/gfootball:setup_py_data",\n', "")
edit("BUILD", r'    # Materialize.*?"//third_party/mjlab:testdata",\n', "")
# Runtime dependencies of the core and Python API only, and a local version.
edit("setup.cfg", r"version = ([0-9.]+)", r"version = \1+elevator")
edit("setup.cfg", r"install_requires =\n.*?\n\n",
     "install_requires =\n    dm-env>=1.4\n    glfw>=2.10.0\n    gymnasium>=0.26\n"
     "    numpy>=1.19\n    packaging\n    optree>=0.6.0\n\n")
EOF
cp "$SRC/third_party/pip_requirements/requirements-release-lock.txt" \
   "$SRC/third_party/pip_requirements/requirements.txt"

# 4. Build the wheel (the release config: -O3, no asserts).
cd "$SRC"
"$BAZEL" run --config=release //:setup_py312 -- bdist_wheel
mkdir -p "$DIST"
rm -f "$DIST"/envpool-*.whl
cp bazel-bin/setup_py312.runfiles/_main/dist/envpool-*.whl "$DIST"/
ls "$DIST"/envpool-*.whl
