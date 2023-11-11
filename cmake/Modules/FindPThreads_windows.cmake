# FindPThreads_windows.cmake

# Check if pthreadVC2.lib is present
find_library(PTHREAD_LIBRARIES
    NAMES pthreadVC2.lib pthreadVC.lib pthreadVCE2.lib
    HINTS ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/3rdparty/pthreads/lib
)

# Check if pthread.h is present
find_path(PTHREAD_INCLUDE_DIR pthread.h
    HINTS ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/3rdparty/pthreads/include
)

# Check if we found the PThreads library
if (PTHREAD_LIBRARIES AND PTHREAD_INCLUDE_DIR)
    set(PTHREADS_FOUND TRUE)
endif ()

# Provide the results to the user
mark_as_advanced(PTHREAD_LIBRARIES PTHREAD_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PThreads DEFAULT_MSG PTHREAD_LIBRARIES PTHREAD_INCLUDE_DIR)

# Define variables for the user
if (PTHREADS_FOUND)
    set(PThreads_FOUND TRUE)
    set(PThreads_LIBRARIES ${PTHREAD_LIBRARIES})
    set(PThreads_INCLUDE_DIRS ${PTHREAD_INCLUDE_DIR})
endif ()

# Provide results to the user
if (PTHREADS_FOUND)
    if (NOT PThreads_FIND_QUIETLY)
        message(STATUS "Found PThreads: ${PTHREAD_LIBRARIES}")
    endif ()
else ()
    if (PThreads_FIND_REQUIRED)
        message(FATAL_ERROR "PThreads library not found")
    endif ()
endif ()
