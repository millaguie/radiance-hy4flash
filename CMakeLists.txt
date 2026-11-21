# Hy4-flash plugins for radiance 2.2.0, out of tree (docs/PLUGIN.md §13). One tree for 2.2.0 only:
# the 2.1.0 one (ABI 40) is the state before the move, and keeping both would mean two vendored
# rad_fp8.h copies for one engine change.
cmake_minimum_required(VERSION 3.21)
project(radiance_hy4flash LANGUAGES CXX)
find_package(radiance REQUIRED)
# arch/hyv4flash/vendor/arch/rad_fp8.h is a copy of the radiance 2.2.0 header (ABI 43): another ABI
# needs it redone, so the build stops here.
if(NOT RADIANCE_ABI_VERSION EQUAL 43)
  message(FATAL_ERROR "radiance-hy4flash: the vendor/arch/rad_fp8.h copy is from radiance 2.2.0 (ABI 43); "
                      "this radiance is ABI ${RADIANCE_ABI_VERSION}. Redo it from its arch/common/rad_fp8.h.")
endif()
if(RAD_HIP_FOUND)
  enable_language(HIP)
endif()
rad_plugin_defaults()
add_subdirectory(kernels/hy4k)
add_subdirectory(arch/hyv4flash)
