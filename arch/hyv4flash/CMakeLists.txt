# hyv4flash: the Hy4-flash architecture (docs/DESIGN.md).
rad_add_plugin(hyv4flash architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hyv4flash.cpp)
# The copy of rad_fp8.h with the rotated clamp goes ahead of the installed one (vendor/arch/rad_fp8.h).
target_include_directories(hyv4flash BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
