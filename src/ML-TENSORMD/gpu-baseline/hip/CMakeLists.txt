project(hipkernel LANGUAGES CXX HIP)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.hip)

if (LINK_SHARED)
    add_library(hipkernel SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
else()
    add_library(hipkernel STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
endif()
target_include_directories(hipkernel PRIVATE ../include)

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
target_compile_options(hipkernel PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2 -g>)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hipkernel PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O0 -g>)
else()
    target_compile_options(hipkernel PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
endif()

set_target_properties(hipkernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hipkernel PROPERTIES HIP_SEPARABLE_COMPILATION ON)

find_package(hipblas)
find_package(OpenMP)
target_link_libraries(hipkernel PRIVATE roc::hipblas OpenMP::OpenMP_CXX)