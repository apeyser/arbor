cythonate(
  pyarbor
  miniapp_recipes.cpp
  morphology_pool.cpp
  trace.cpp
  io.cpp
)

install(PROGRAMS cyminiapp.py DESTINATION bin)
