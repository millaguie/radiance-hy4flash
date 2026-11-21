# hy4k: the new Hy4-flash ops (docs/DESIGN.md). Host kernels (oracle, reference) and, with a device
# build, the HIP rows of plan step 6.
set(HY4K_HOST
    ${CMAKE_CURRENT_SOURCE_DIR}/schemas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_hc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_mla.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_vit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_rope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_engram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_idx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_ring.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)
# The card rows (plan step 6) only where the installed radiance has a device build.
set(HY4K_HIP
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_hc.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_engram.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_idx.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_attn.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_vit.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_quant.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/dev_ring.hip)
if(RAD_HIP_FOUND)
  rad_add_plugin(hy4k kernels SOURCES ${HY4K_HOST} HIP_SOURCES ${HY4K_HIP})
  target_compile_definitions(hy4k PRIVATE HY4K_WITH_DEVICE=1)
else()
  rad_add_plugin(hy4k kernels SOURCES ${HY4K_HOST})
endif()
# No FMA and no reassociation: the reference contract (docs/PLUGIN.md §6).
target_compile_options(hy4k PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
                                    $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
# OpenMP as in libref: the host kernels split rows across threads (disjoint writes).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hy4k PRIVATE OpenMP::OpenMP_CXX)
endif()
