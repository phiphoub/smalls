set(SOURCES
    vector_device_test.cu
    )

cuda_add_executable(vector_device_test
	${SOURCES}
    )

target_link_libraries(vector_device_test ${GTEST_BOTH_LIBRARIES})

# Add tests to CTest - the tests are automatically picked up by CMake and registered with CTest
add_test(vector_device_test ${EXECUTABLE_OUTPUT_PATH}/types_test)
