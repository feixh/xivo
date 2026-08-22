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

if (USE_GPERFTOOLS)
  add_definitions(-DUSE_GPERFTOOLS)
endif (USE_GPERFTOOLS)

if (BUILD_G2O)
  add_definitions(-DUSE_G2O)
endif (BUILD_G2O)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++17 -Wno-narrowing -Wno-register -fPIC -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
#add_definitions(-DNDEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# disable logging whose severity level is below the given integer
add_definitions(-DGOOGLE_STRIP_LOG=1)

# add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

# Sophus formats Eigen expressions through libfmt in its assertion messages.
# fmt >= 9 dropped the implicit ostream fallback, so those messages no longer
# compile. Basic logging keeps the assertions but prints the format string
# verbatim instead of formatting Eigen types; behaviour is otherwise unchanged.
add_definitions(-DSOPHUS_USE_BASIC_LOGGING)

# EKF state-size overrides for the defaults in src/core.h. These MUST live here
# rather than in src/CMakeLists.txt: `add_definitions` only reaches the current
# directory and its subdirectories, and the pyxivo module is built from this
# top-level list, so a define set under src/ would give the Python bindings a
# different kMaxGroup/kFullSize than the library they link against -- an ODR
# violation that silently changes the trajectories the evaluation scripts see.
#
# The out-of-state update needs a sliding window of past poses in the state, on
# top of the reference groups of the in-state features, so the default 15 group
# slots is not enough (see notes-n-prompts/notes-oos/m3-pose-window.md). This is
# not result-neutral even with the pose window off: the number of free group
# slots decides which branch of SelectAndAddNewFeatures runs and how many groups
# AddGroupOfFeatures admits. Both the baseline and the OOS numbers in the report
# use this same value.
add_definitions(-DEKF_MAX_GROUPS=60)
add_definitions(-DEKF_MAX_FEATURES=200)

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
