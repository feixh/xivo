cmake_minimum_required(VERSION 3.5)
project(feh)

# Set operating system variables
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(IS_LINUX TRUE)
endif ()
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(IS_MAC TRUE)
endif ()
message(STATUS "IS_LINUX ${IS_LINUX}")
message(STATUS "IS_MAC ${IS_MAC}")

option(BUILD_G2O "build with g2o support" OFF)
option(USE_GPERFTOOLS "use gperf for performance profiling" OFF)

# EKF state size. These were compile-time only, set by hand-editing the
# commented-out add_definitions() in src/CMakeLists.txt; core.h falls back to 30
# features / 15 groups, which is what every number in RESULTS.md was produced
# with. Exposing them as cache variables makes a larger-state variant a build
# argument instead of a source edit -- and, paired with XIVO_OUTPUT_SUFFIX, lets
# it live beside the default build rather than overwriting bin/ and lib/.
#
# The defaults are 90/45 rather than core.h's 30/15 because that is what
# cfg/tumvi_stereo.json needs: at 30 features the filter discards most of what a
# stereo pair gives it, and the six-room mean ATE goes from 0.048 m to 0.076 m.
# Raising the *state* alone does nothing -- tracker_cfg.num_features_max caps
# what reaches it -- so the shipped stereo config raises both together. Build
# with -DEKF_MAX_FEATURES=30 -DEKF_MAX_GROUPS=15 to reproduce RESULTS.md.
#
# The out-of-state (MSCKF) update wants more still: its pose window occupies
# group slots on top of the reference groups of the in-state features, and
# cfg/oos.json was tuned at 200/60 with a 100-130 feature tracker (see
# notes-n-prompts/notes-oos/). Note the group count is not result-neutral even
# with the pose window off -- the number of free group slots decides which
# branch of SelectAndAddNewFeatures runs and how many groups AddGroupOfFeatures
# admits -- so a configuration has to be compared against a baseline built at
# the same state size. Pair a non-default size with XIVO_OUTPUT_SUFFIX to keep
# both builds on disk.
set(EKF_MAX_FEATURES "90" CACHE STRING
    "features in the EKF state; blank uses core.h's default of 30")
set(EKF_MAX_GROUPS "45" CACHE STRING
    "groups in the EKF state; blank uses core.h's default of 15")
if (EKF_MAX_FEATURES)
  add_definitions(-DEKF_MAX_FEATURES=${EKF_MAX_FEATURES})
  message(STATUS "EKF_MAX_FEATURES ${EKF_MAX_FEATURES}")
endif ()
if (EKF_MAX_GROUPS)
  add_definitions(-DEKF_MAX_GROUPS=${EKF_MAX_GROUPS})
  message(STATUS "EKF_MAX_GROUPS ${EKF_MAX_GROUPS}")
endif ()

# Appended to the bin/ and lib/ output directories, so two builds with different
# state sizes can coexist. `scripts/pyxivo.py` takes -lib to match.
set(XIVO_OUTPUT_SUFFIX "" CACHE STRING
    "suffix on the bin/ and lib/ output directories")

if (USE_GPERFTOOLS)
  add_definitions(-DUSE_GPERFTOOLS)
endif (USE_GPERFTOOLS)

if (BUILD_G2O)
  add_definitions(-DUSE_G2O)
endif (BUILD_G2O)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++17 -Wno-narrowing -Wno-register -fPIC -g")

# Overridable because valgrind cannot execute the AVX-512 that -march=native
# emits on this machine; a heap-profiling build sets e.g. -march=x86-64-v3.
set(XIVO_ARCH_FLAGS "-mtune=native -march=native" CACHE STRING "architecture flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XIVO_ARCH_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
#add_definitions(-DNDEBUG)

# Sanitizer instrumentation, e.g. -DXIVO_SANITIZE=address or
# -DXIVO_SANITIZE=address,undefined. Off by default. See scripts/mem/README.md.
set(XIVO_SANITIZE "" CACHE STRING "sanitizers to enable (address, undefined, ...)")
if (XIVO_SANITIZE)
  message(STATUS "XIVO_SANITIZE ${XIVO_SANITIZE}")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fsanitize=${XIVO_SANITIZE} -fno-omit-frame-pointer -fno-optimize-sibling-calls")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=${XIVO_SANITIZE}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=${XIVO_SANITIZE}")
endif (XIVO_SANITIZE)

# Where bin/ and lib/ go. Two independent knobs, both of which exist so a
# variant build can sit beside the plain one rather than overwrite it:
# XIVO_OUTPUT_DIR moves the whole prefix (scripts/mem/build.sh, for a sanitized
# or valgrind-friendly tree) and XIVO_OUTPUT_SUFFIX renames bin/lib in place
# (build_arch.sh, for a differently-sized EKF state -- see EKF_MAX_FEATURES).
set(XIVO_OUTPUT_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "prefix for the bin/ and lib/ output directories")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${XIVO_OUTPUT_DIR}/bin${XIVO_OUTPUT_SUFFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${XIVO_OUTPUT_DIR}/lib${XIVO_OUTPUT_SUFFIX})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${XIVO_OUTPUT_DIR}/lib${XIVO_OUTPUT_SUFFIX})

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)

# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

# Sophus formats Eigen expressions through libfmt in its assertion messages.
# fmt >= 9 dropped the implicit ostream fallback, so those messages no longer
# compile. Basic logging keeps the assertions but prints the format string
# verbatim instead of formatting Eigen types; behaviour is otherwise unchanged.
add_definitions(-DSOPHUS_USE_BASIC_LOGGING)

# Online temporal calibration: estimate the IMU-camera time offset td as part of
# the state. Set here rather than in src/CMakeLists.txt (where it used to live)
# for the same reason as the EKF_MAX_* cache variables above: `add_definitions`
# only reaches the current directory and the subdirectories added after it, and
# pyxivo is built from this top-level list, so a define under src/ gives the
# Python bindings a different state layout than the library they link against.
add_definitions(-DUSE_ONLINE_TEMPORAL_CALIB)

find_package(OpenCV REQUIRED)
find_package(Python3 REQUIRED Interpreter Development)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/gflags/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/glog/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/gperftools/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/pnp/build
        ${PROJECT_SOURCE_DIR}/thirdparty/ceres-solver/lib
        /usr/local/lib
)
if (IS_LINUX)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif(IS_LINUX)


include_directories(
        ${PROJECT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/thirdparty/gflags/include
        ${PROJECT_SOURCE_DIR}/thirdparty/glog/include
        ${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/include
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/include
        ${PROJECT_SOURCE_DIR}/thirdparty/eigen/include/eigen3
        ${PROJECT_SOURCE_DIR}/thirdparty/gperftools/include
        ${PROJECT_SOURCE_DIR}/thirdparty/pybind11/include
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/include
        ${PROJECT_SOURCE_DIR}/thirdparty/pnp
        ${PROJECT_SOURCE_DIR}/thirdparty/sophus
        ${JSONCPP_INCLUDE_DIRS}
        /usr/include/suitesparse
        /usr/include
        ${OpenCV_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)



enable_testing()
add_subdirectory(thirdparty/googletest)
add_subdirectory(thirdparty/gflags)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/googletest/include)
# add_subdirectory(thirdparty/abseil-cpp)
if (BUILD_G2O)
  link_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o/release/lib)
  include_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o/release/include)
  list(APPEND deps
    g2o_core
    g2o_solver_dense
    g2o_solver_cholmod
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_types_slam3d
    g2o_types_sba
    g2o_stuff
    cholmod
    cxsparse
    )
endif(BUILD_G2O)

# feh
add_subdirectory(common)
add_subdirectory(src)

########################################
# PYTHON BINDING
########################################
# NOTE: to build with a specific python version
# cmake -DPYTHON_EXECUTABLE=path/to/python ..
# By default, the python binding generated is only compatible with
# your default python interpreter, which you can check by typing
# "which python" in your terminal.
# If you see an error saying "Python.h: No such file or directory", you probably
# need to "sudo apt-get install python3-dev" assuming you are binding to python3.
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(thirdparty/pybind11)
pybind11_add_module(pyxivo MODULE pybind11/pyxivo.cpp)
set(libxivo common xest xapp)
if (BUILD_G2O)
  list(APPEND libxivo xopt)
endif(BUILD_G2O)
target_link_libraries(pyxivo PRIVATE ${libxivo})
