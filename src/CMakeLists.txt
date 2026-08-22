cmake_minimum_required(VERSION 3.5)

project(estimator)

# # overwrite binary output directory
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# if set, use inverse-depth parametrization
# add_definitions(-DUSE_INVDEPTH)

# Online calibration of the IMU intrinsics (Cg, Ca). Measured inert on TUM-VI:
# with P.Cg = P.Ca = 1e-5 (prior sigma 0.32%) the estimates move ~1e-6 over a
# whole sequence, i.e. 0.03% of one sigma, because the dataset's IMU is already
# factory-calibrated. Enabling it leaves mean ATE and RPE unchanged (0.1090 vs
# 0.1102, well inside the +/-0.005 ensemble noise) while adding 15 weakly
# observable states, which raises the ATE spread back to sd 0.0047 from the
# 0.0020 temporal calibration alone achieves. The Jacobian chain it needs was
# broken until M6 (bug #35) and its unit tests are ifdef'd behind this flag, so
# turn it on when touching rodrigues.h.
#add_definitions(-DUSE_ONLINE_IMU_CALIB)

# Online temporal calibration (camera/IMU time offset td). ON: it estimates a
# small nonzero offset (-0.077 ms on room1) and halves the run-to-run ATE spread
# (sd 0.0047 -> 0.0020) at a slightly better mean. Requires the td plumbing
# fixed in M6 (bug #46: State::td was read uninitialised).
add_definitions(-DUSE_ONLINE_TEMPORAL_CALIB)
#add_definitions(-DUSE_ONLINE_CAMERA_CALIB)

# if set, enables the mapper object (and loop closure)
#add_definitions(-DUSE_MAPPER)

# if set, approximate the initial correlation between feature state and
# group state with Hessian from depth refinement optimization.
# WARNING: this feature not work yet.
# add_definitions(-DAPPROXIMATE_INIT_COVARIANCE)

# Sets the maximum number of features and groups in the EKF. Otherwise
# use the default values in core.h
#add_definitions(-DEKF_MAX_FEATURES=125)
#add_definitions(-DEKF_MAX_GROUPS=75)

# Sets the frequency at which to project rotation matrices to SO(3) to get
# rid of accumulated numeric error
#add_definitions(-DENFORCE_SO3_FREQ=50)

# Sets the size of the message buffer heap; this is the number of messages stored
# before they are processed. A bigger size can accommodate a more-negative value
# of td, but will be slower.
#add_definitions(-DMESSAGE_BUFFER_SIZE=100)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/common)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib)

list(APPEND deps
        opencv_highgui
        opencv_features2d
        opencv_core
        opencv_video
        opencv_imgproc
        opencv_imgcodecs
        opencv_xfeatures2d
        glog
        GLEW
        pangolin
        pthread
        jsoncpp
        DBoW2
        libpnp
        common
        ceres
        cxsparse
        cholmod
        lapack
        blas
        fmt
        )
if (IS_LINUX)
  list(APPEND deps GL)
endif (IS_LINUX)

if (USE_GPERFTOOLS)
  list(APPEND deps profiler)
endif (USE_GPERFTOOLS)

add_library(xapp STATIC
        estimator_process.cpp
        loader.cpp
        geometry.cpp
        metrics.cpp
        publisher.cpp
        graphwriter.cpp
        viewer.cpp)
target_link_libraries(xapp ${deps})

add_library(xest STATIC
        factory.cpp
        estimator.cpp
        estimator_accessors.cpp
        princedormand.cpp
        rk4.cpp
        visualize.cpp
        tracker.cpp
        manager.cpp
        update.cpp
        graphbase.cpp
        graph.cpp
        feature.cpp
        fastbrief.cpp
        oos.cpp
        group.cpp
        helpers.cpp
        options.cpp
        param.cpp
        mm.cpp
        mapper.cpp
        camera_manager.cpp
        imu.cpp)
if (IS_MAC)
  target_link_libraries(xest INTERFACE ${deps} "-Wl, -framework OpenGL")
elseif (IS_LINUX)
  target_link_libraries(xest INTERFACE ${deps})
endif()

set(libxivo xest xapp)

if (BUILD_G2O)
  add_library(xopt STATIC
    optimizer.cpp
    optimizer_adapters.cpp
  )
  target_link_libraries(xopt xest ${deps})
  list(APPEND libxivo xopt)
endif(BUILD_G2O)

add_executable(vio app/vio.cpp)
target_link_libraries(vio ${libxivo} gflags::gflags)

add_executable(feature_tracker_only app/feature_tracker_only.cpp)
target_link_libraries(feature_tracker_only ${libxivo} gflags::gflags)

################################################################################
# TOOLING
################################################################################
# add_executable(eval app/evaluate.cpp)
# target_link_libraries(eval estimator application gflags::gflags)

################################################################################
# TESTS
################################################################################
add_executable(unitTests_NumericalAlgorithms
               test/unittest_givens.cpp
               test/unittest_rodrigues.cpp
               test/unittest_helpers.cpp)
target_link_libraries(unitTests_NumericalAlgorithms xest ${deps} gtest gtest_main)
add_test(NAME NumericalAlgorithms COMMAND unitTests_NumericalAlgorithms)

add_executable(unitTests_Jacobians
               test/unittest_jacobians_instate.cpp
               test/unittest_jacobians_oos.cpp
               test/unittest_reanchor.cpp
               test/unittest_helpers.cpp)
target_link_libraries(unitTests_Jacobians ${libxivo} ${deps} gtest gtest_main)
add_test(NAME Jacobians COMMAND unitTests_Jacobians)

add_executable(unitTests_pinhole
               test/unittest_camera_pinhole.cpp)
target_link_libraries(unitTests_pinhole xest ${deps} gtest gtest_main)
add_test(NAME CamerasPinhole COMMAND unitTests_pinhole)

add_executable(unitTests_radtan
               test/unittest_camera_radtan.cpp)
target_link_libraries(unitTests_radtan xest ${deps} gtest gtest_main)
add_test(NAME CamerasRadtan COMMAND unitTests_radtan)

add_executable(unitTests_equi
               test/unittest_camera_equi.cpp)
target_link_libraries(unitTests_equi xest ${deps} gtest gtest_main)
add_test(NAME CamerasEqui COMMAND unitTests_equi)

add_executable(unitTests_atan
               test/unittest_camera_atan.cpp)
target_link_libraries(unitTests_atan xest ${deps} gtest gtest_main)
add_test(NAME CamerasAtan COMMAND unitTests_atan)

add_executable(unitTests_triangulation
               test/unittest_triangulation.cpp)
target_link_libraries(unitTests_triangulation xest ${deps} gtest gtest_main)
add_test(NAME Triangulation COMMAND unitTests_triangulation)

add_executable(unitTests_geometry
               test/unittest_geometry.cpp)
target_link_libraries(unitTests_geometry xapp xest ${deps} gtest gtest_main)
add_test(NAME Geometry COMMAND unitTests_geometry)

if (BUILD_G2O)
  message(INFO ${libxivo})
  add_executable(test_optimizer test/test_optimizer.cpp)
  target_link_libraries(test_optimizer ${libxivo})
endif(BUILD_G2O)

# add_executable(test_estimator test/test_estimator.cpp )
# target_link_libraries(test_estimator estimator gtest gtest_main)
#
# add_executable(test_simulator test/test_simulator.cpp )
# target_link_libraries(test_simulator estimator gtest gtest_main)

# add_executable(test_tumvi test/test_tumvi.cpp )
# target_link_libraries(test_tumvi estimator gtest gtest_main)

# add_executable(test_viewer test/test_viewer.cpp)
# target_link_libraries(test_viewer estimator gtest gtest_main)

# add_executable(test_tracker test/test_tracker.cpp )
# target_link_libraries(test_tracker estimator gtest gtest_main)
#
# add_executable(test_matcher test/test_matcher.cpp )
# target_link_libraries(test_matcher estimator gtest gtest_main)
