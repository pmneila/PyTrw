
cmake_minimum_required(VERSION 2.8)
project(trw)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.42.0 REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)

# Set the numpy headers dir.
execute_process(COMMAND python -c "import numpy; print numpy.get_include()"
                RESULT_VARIABLE NUMPY_RESULT
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
                ERROR_VARIABLE NUMPY_ERROR)
if(${NUMPY_RESULT} EQUAL 1)
    message(SEND_ERROR ${NUMPY_ERROR})
endif(${NUMPY_RESULT} EQUAL 1)

set(trw_SOURCES core/minimize.cpp
                core/MRFEnergy.cpp
                core/ordering.cpp
                core/treeProbabilities.cpp
                core/example.cpp
                python.cpp
                pyarray_index.cpp)

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
add_library(_trw MODULE ${trw_SOURCES})
set_target_properties(_trw PROPERTIES PREFIX "")
target_link_libraries(_trw ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(trw ${trw_SOURCES})
