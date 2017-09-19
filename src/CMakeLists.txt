
file( GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

add_library( ssd_wrapper SHARED ${srcs} )

target_link_libraries( ssd_wrapper PUBLIC ${Caffe_LIBRARIES} ${OPENCVLIBS})
target_include_directories( ssd_wrapper PUBLIC ${Caffe_INCLUDE_DIRS} )
target_compile_definitions( ssd_wrapper PUBLIC ${Caffe_DEFINITIONS} )

INSTALL( TARGETS ssd_wrapper
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION staticlib
        ARCHIVE DESTINATION lib )
