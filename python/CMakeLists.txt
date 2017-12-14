# Defines the CMake commands/policies
cmake_minimum_required( VERSION 3.5.0 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

function(cythonate MODULE SOURCES)
  add_cython_target( ${MODULE} CXX )
  add_library( ${MODULE} MODULE ${${MODULE}} ${${SOURCES}} )
  python_extension_module(${MODULE})
  
  install(TARGETS ${MODULE} LIBRARY DESTINATION ${MODULE})
  
  target_link_libraries(${MODULE} ${ARB_LIBRARIES})
  target_link_libraries(${MODULE} ${EXTERNAL_LIBRARIES})
  
  #message("XXX: ${PROJECT_SOURCE_DIR}/python/${MODULE}")
  include_directories("${PROJECT_SOURCE_DIR}")
  target_link_libraries(${MODULE} ${PYTHON_LIBRARIES})
  
  if(ARB_WITH_MPI)
    target_link_libraries(${MODULE} ${MPI_C_LIBRARIES})
    set_property(TARGET ${MODULE} APPEND_STRING PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
  endif()

  set_target_properties(
    ${MODULE}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/miniapps/${MODULE}"
  )
endfunction()

#enable_testing()
#find_file( NOSETESTS_EXECUTABLE nosetests )
#add_test( nosetests "${NOSETESTS_EXECUTABLE}" -v --with-xunit )

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( miniapp )
add_subdirectory( pyarbor )
#add_subdirectory( bin )
