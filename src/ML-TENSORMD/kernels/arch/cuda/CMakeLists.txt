cmake_minimum_required(VERSION 3.24)
project(tensormd_cuda LANGUAGES CXX CUDA)

# The library needs to be compiled with C++17 standard to use modern features.
set(CMAKE_CXX_STANDARD 17)

# Define Sources
# IMPORTANT: Only compile the entry point `kernels_gpu.cu`. 
# The other files (kernels_gpu_forces.cu, etc.) are included inside kernels_gpu.cu
# and should NOT be added to add_library directly.
set(CUDA_SOURCES 
    cuda_common.cu
    kernels_cuda_core_setup.cu
    kernels_cuda_core_setup_sync.cu
    kernels_cuda_core_setup_neigh.cu
    kernels_cuda_core_forces.cu
    kernels_cuda_core_P.cu
    kernels_cuda_core_G.cu
    kernels_cuda_core_W.cu
    kernels_cuda_core_Fa.cu
    kernels_cuda_utils.cu
    kernels_cuda_utils_timer.cu
    kernels_cuda_utils_memory.cu
    kernels_cuda_encoder.cu
    kernels_cuda_encoder_cutoff.cu
    kernels_cuda_encoder_Fr.cu
    kernels_cuda_spline_cubic.cu
    kernels_cuda_fusion_forward.cu
    kernels_cuda_fusion_backward.cu
    kernels_cuda_fusion.cu
    kernels_cuda_head.cu
    kernels_cuda_head_fusion.cu
    kernels_cuda_head_hybrid.cu
    kernels_cuda_v1_ops.cu
    kernels_cuda_v2_ops.cu
)

# Create Library
# We use OBJECT library or STATIC. Static is safer for linking into LAMMPS.
add_library(tensormd_cuda STATIC ${CUDA_SOURCES})

# Compile Options
target_compile_definitions(tensormd_cuda PRIVATE TENSORMD_GPU)

# Optimization flags
if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(tensormd_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 -g -lineinfo -fopenmp>)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tensormd_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O0 -g -G -fopenmp>)
    target_compile_definitions(tensormd_cuda PRIVATE TENSORMD_DEBUG)
else()
    target_compile_options(tensormd_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fopenmp>)
endif()

set_source_files_properties(
  # kernels_cuda_fusion_backward_speed.cu
  kernels_cuda_head_hybrid.cu
  PROPERTIES
    COMPILE_OPTIONS "-Xptxas=-v,-warn-spills,-warn-lmem-usage"
)

# Include Directories
# Needs to see ../../../tensormd_types.h and internal headers
target_include_directories(tensormd_cuda PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../.. 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ 
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

# Architecture Flags
set_target_properties(tensormd_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tensormd_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tensormd_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if (ENABLE_CUDA_WMMA)
    set_target_properties(tensormd_cuda PROPERTIES CUDA_ARCHITECTURES "80")
    target_compile_definitions(tensormd_cuda PRIVATE ENABLE_CUDA_WMMA)
else()
    set_target_properties(tensormd_cuda PROPERTIES CUDA_ARCHITECTURES "80")
endif()

# Link cuBLAS
target_link_libraries(tensormd_cuda PRIVATE "-lcublas -lcusolver")
