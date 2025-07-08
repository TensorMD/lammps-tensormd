project(cudakernel LANGUAGES CXX CUDA)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

if (LINK_SHARED)
    add_library(cudakernel SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
else()
    add_library(cudakernel STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
endif()

target_include_directories(cudakernel PRIVATE ../include)

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(cudakernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O2 -g -fopenmp>)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cudakernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O0 -g -fopenmp>)
    target_compile_definitions(cudakernel PRIVATE GPU_DEBUG)
else()
    target_compile_options(cudakernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -fopenmp>)
endif()

set_target_properties(cudakernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cudakernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cudakernel PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cudakernel PROPERTIES CUDA_ARCHITECTURES "70;72;75;80")

target_link_libraries(cudakernel PRIVATE "-lcublas")