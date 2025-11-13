#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GDeflate::GDeflate" for configuration "Debug"
set_property(TARGET GDeflate::GDeflate APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GDeflate::GDeflate PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "GDeflate::libdeflate_static"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libGDeflate.a"
  )

list(APPEND _cmake_import_check_targets GDeflate::GDeflate )
list(APPEND _cmake_import_check_files_for_GDeflate::GDeflate "${_IMPORT_PREFIX}/lib/libGDeflate.a" )

# Import target "GDeflate::GDeflate_shared" for configuration "Debug"
set_property(TARGET GDeflate::GDeflate_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GDeflate::GDeflate_shared PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "GDeflate::libdeflate_static"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libGDeflate.so.1.0.0"
  IMPORTED_SONAME_DEBUG "libGDeflate.so.1"
  )

list(APPEND _cmake_import_check_targets GDeflate::GDeflate_shared )
list(APPEND _cmake_import_check_files_for_GDeflate::GDeflate_shared "${_IMPORT_PREFIX}/lib/libGDeflate.so.1.0.0" )

# Import target "GDeflate::libdeflate_static" for configuration "Debug"
set_property(TARGET GDeflate::libdeflate_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(GDeflate::libdeflate_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libdeflate.a"
  )

list(APPEND _cmake_import_check_targets GDeflate::libdeflate_static )
list(APPEND _cmake_import_check_files_for_GDeflate::libdeflate_static "${_IMPORT_PREFIX}/lib/libdeflate.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
