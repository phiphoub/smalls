set(SOURCES
    vector_host_test.cpp
    )

add_executable(vector_host_test
	${SOURCES}
    )

target_link_libraries(vector_host_test gtest gtest_main)

# Add tests to CTest - the tests are automatically picked up by CMake and registered with CTest
add_test(vector_host_test ${EXECUTABLE_OUTPUT_PATH}/types_test)
