project( pyarbor )

set(PYARBOR_SOURCES
    pyarbor-base.cpp
    miniapp_recipes.cpp
    morphology_pool.cpp
    trace.cpp
    io.cpp
)

cythonate(pyarbor PYARBOR_SOURCES)
