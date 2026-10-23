# Python extension module `hce` (pybind11). Skipped with a message when Python or pybind11 is
# missing, and in sanitizer builds (the interpreter would need the sanitizer runtime preloaded).
if(HCE_SANITIZE)
  message(STATUS "hce: Python module skipped in sanitizer builds")
  return()
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module)
if(NOT Python_FOUND)
  message(STATUS "hce: Python development files not found; skipping the Python module")
  return()
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE pybind11_rc)
if(pybind11_rc EQUAL 0)
  list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
endif()
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "hce: pybind11 not found for ${Python_EXECUTABLE}; skipping the Python module "
                 "(pip install pybind11, then configure with -DPython_EXECUTABLE=<that python>)")
  return()
endif()

pybind11_add_module(hce_python hce_module.cpp)
set_target_properties(hce_python PROPERTIES
  OUTPUT_NAME hce
  LIBRARY_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/python>)
target_link_libraries(hce_python PRIVATE hce_format)

if(HCE_BUILD_TESTS)
  add_test(NAME python_bindings
    COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -v)
  set_tests_properties(python_bindings PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")
endif()
