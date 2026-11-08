set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

# The clang driver links with -l and -L. Keep that per language: CUDA's platform module
# brings MSVC's link flags for every language.
foreach(lang C CXX)
    set( CMAKE_${lang}_LINK_LIBRARY_FLAG "-l" )
    set( CMAKE_${lang}_LIBRARY_PATH_FLAG "-L" )
endforeach()
