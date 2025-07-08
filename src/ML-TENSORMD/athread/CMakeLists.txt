project(athread-tensoralloy C)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ATHREAD_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/slaves ATHREAD_SLAVE_SRC)

add_library(athread-tensoralloy STATIC ${ATHREAD_SRC})
add_library(athread-slave-tensoralloy STATIC ${ATHREAD_SLAVE_SRC})

if(BUILD_BASELINE)
    target_compile_definitions(athread-slave-tensoralloy PRIVATE BUILD_BASELINE)
endif()

target_compile_options(athread-slave-tensoralloy PRIVATE -mslave -msimd -mfma -Ofast -faddress_align=64)

target_link_libraries(athread-tensoralloy PRIVATE athread-slave-tensoralloy)