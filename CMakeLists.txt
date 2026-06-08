cmake_minimum_required(VERSION 3.16)
project(fpddp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# blasfeo + hpipm are built once via scripts/build_deps.sh into external/install.
set(FPDDP_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/external/install)
if(NOT EXISTS ${FPDDP_DEPS}/lib/libhpipm.a)
  message(FATAL_ERROR "blasfeo/hpipm not found in ${FPDDP_DEPS}. "
                      "Run: git submodule update --init && scripts/build_deps.sh")
endif()

add_library(blasfeo STATIC IMPORTED)
set_target_properties(blasfeo PROPERTIES IMPORTED_LOCATION ${FPDDP_DEPS}/lib/libblasfeo.a)
add_library(hpipm STATIC IMPORTED)
set_target_properties(hpipm PROPERTIES IMPORTED_LOCATION ${FPDDP_DEPS}/lib/libhpipm.a)

# header-only fp-ddp + vendored irk + anyany
add_library(fpddp INTERFACE)
target_include_directories(fpddp INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/anyany/include
  ${FPDDP_DEPS}/include)
target_link_libraries(fpddp INTERFACE hpipm blasfeo m)   # link order: hpipm before blasfeo
target_compile_options(fpddp INTERFACE -Wall -Wextra)

enable_testing()
foreach(t irk_demo sens_test model_smoke lqr_test ddp_test globalization_test cmp_pendulum cmp_cartpole cmp_unicycle)
  add_executable(${t} test/${t}.cpp)
  target_link_libraries(${t} PRIVATE fpddp)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
