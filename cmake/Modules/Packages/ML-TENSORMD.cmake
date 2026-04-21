# ------------------------------
# ML-TENSORMD package for LAMMPS
# ------------------------------

# 1. Define the root directory for includes
set(TENSORMD_ROOT ${LAMMPS_SOURCE_DIR}/ML-TENSORMD)
target_include_directories(lammps PRIVATE ${TENSORMD_ROOT})

# 2. Collect C++ Sources
# Logic Core
file(GLOB TENSORMD_SRCS_CORE 
    ${TENSORMD_ROOT}/tensormd.cpp
    ${TENSORMD_ROOT}/tensormd_v1.cpp
    ${TENSORMD_ROOT}/tensormd_v2.cpp
    ${TENSORMD_ROOT}/tensormd_timer.cpp
    ${TENSORMD_ROOT}/tensormd_strategy.cpp
    ${TENSORMD_ROOT}/encoders/*.cpp
    ${TENSORMD_ROOT}/heads/*.cpp
    ${TENSORMD_ROOT}/utils/*.cpp
)

# LAMMPS Interface (Pair, Fixes, Computes)
file(GLOB TENSORMD_SRCS_LMP 
    ${TENSORMD_ROOT}/pair_tensormd.cpp
    # ${TENSORMD_ROOT}/lammps/*.cpp 
)

# CPU Kernels
file(GLOB TENSORMD_SRCS_KERNELS_CPU 
    ${TENSORMD_ROOT}/kernels/kernels.cpp
    ${TENSORMD_ROOT}/kernels/kernels_*.cpp
    ${TENSORMD_ROOT}/kernels/arch/cpu/*.cpp
)

# Add to LAMMPS target
target_sources(lammps PRIVATE 
    ${TENSORMD_SRCS_CORE} 
    ${TENSORMD_SRCS_LMP}
    ${TENSORMD_SRCS_KERNELS_CPU}
)

# Find the MKL library for optimized linear algebra operations
find_package(MKL)

# CPU BLAS/LAPACK support
if(MKL_FOUND)
    message(STATUS "TensorMD: MKL found at $ENV{MKLROOT}")
    target_compile_definitions(lammps PRIVATE EIGEN_USE_MKL_ALL)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -I${MKLROOT}/include")
    endif()
    if(BUILD_OMP)
        set(MKL_LIBRARY "-L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(MKL_LIBRARY "${MKL_LIBRARY} -lmkl_intel_thread -liomp5")
        else()
            set(MKL_LIBRARY "${MKL_LIBRARY} -lmkl_gnu_thread -lgomp")
        endif()
        target_link_libraries(lammps PRIVATE "${MKL_LIBRARY}")
        target_compile_definitions(lammps PRIVATE MULTI_THREAD_BLAS)
    else()
        target_link_libraries(lammps PRIVATE "${MKL_LIBRARY}")
    endif()
else()
    if("${BLA_VENDOR}" STREQUAL "OpenBLAS")
        set(BLAS_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib")
        set(BLAS_INCLUDE_DIRS "/opt/homebrew/opt/openblas/include")
        target_link_libraries(lammps PRIVATE "-ld_classic")
    else()
        find_package(BLAS REQUIRED)
    endif()
    if(MULTI_THREAD_BLAS)
        target_compile_definitions(lammps PRIVATE MULTI_THREAD_BLAS)
    endif()
    target_compile_definitions(lammps PRIVATE EIGEN_USE_BLAS)
    target_link_libraries(lammps PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(lammps PRIVATE ${BLAS_INCLUDE_DIRS})
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(lammps PRIVATE TENSORMD_DEBUG)
endif()

find_package(ZLIB REQUIRED)
target_link_libraries(lammps PRIVATE ZLIB::ZLIB)

find_package(libzip REQUIRED)
target_link_libraries(lammps PRIVATE libzip::zip)

find_package(Boost COMPONENTS filesystem iostreams REQUIRED)
target_include_directories(lammps PUBLIC ${Boost_INCLUDE_DIR})

# 4. Third-Party Libraries (FetchContent)
include(FetchContent)

# Set CMake policies for compatibility with dependencies
set(CMAKE_POLICY_VERSION_MINIMUM 3.15)
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
set(CNPYPP_SPAN_IMPL "BOOST" CACHE STRING "BOOST" FORCE)

# The third-party library yaml-cpp for parsing YAML string
if("${YAMLCPP_REPO}" STREQUAL "")
    set(YAMLCPP_REPO "https://github.com/jbeder/yaml-cpp.git")
endif()
message("Fetch yaml-cpp from ${YAMLCPP_REPO}")
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY "${YAMLCPP_REPO}"
        GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)
target_link_libraries(lammps PUBLIC yaml-cpp::yaml-cpp)

# The third-party library cnpy++ for reading and writing NumPy .npz files
if("${CNPYPP_REPO}" STREQUAL "")
    set(CNPYPP_REPO "https://github.com/mreininghaus/cnpypp.git")
endif()
message("Fetch cnpy++ from ${CNPYPP_REPO}")
FetchContent_Declare(cnpy++
        GIT_REPOSITORY "${CNPYPP_REPO}"
        GIT_SHALLOW True
)
FetchContent_MakeAvailable(cnpy++)
target_link_libraries(lammps PUBLIC cnpy++)

# GPU support
if(BUILD_GPU)
    message("TensorMD GPU Support Enabled")
    if(BUILD_CUDA)
        message("Backend: CUDA")
        enable_language(CUDA)
        # Define TENSORMD_GPU for the C++ Host code so it knows to call GPU kernels
        target_compile_definitions(lammps PRIVATE TENSORMD_GPU)

        # Find CUDA Toolkit and Link Runtime
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(lammps PRIVATE CUDA::cudart)

        # Add the CUDA subdirectory
        add_subdirectory(${TENSORMD_ROOT}/kernels/arch/cuda tensormd_cuda_build)
        
        # Link the resulting library
        target_link_libraries(lammps PRIVATE tensormd_cuda)
    endif()
endif()
