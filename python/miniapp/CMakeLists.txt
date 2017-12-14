project( pyminiapp )

set(PYMINIAPP_SOURCES
    miniapp-base.cpp
    miniapp_recipes.cpp
    morphology_pool.cpp
    trace.cpp
    io.cpp
)

#cythonate(pyminiapp PYMINIAPP_SOURCES )
set(MODULE pyminiapp)
set(SOURCES PYMINIAPP_SOURCES)

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


