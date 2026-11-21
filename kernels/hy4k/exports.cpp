/* exports.cpp -- the exports block of the kernel plugin (docs/PLUGIN.md §1). */
#include "hy4k.h"
#include "rad_abi_layout.h"

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "hy4k", "0.1.0",
    "Hy4-flash ops: iHC, MLA K/V assembly, Engram hash and gate, indexer",
#ifdef HY4K_WITH_DEVICE
    "device",
#else
    "host",
#endif
};
}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
/* The host rows are FP32 oracles: reference implementations, which the engine does not serve
 * without `server.debug-accept-reference-kernels` (docs/PLUGIN.md §1). The card rows are
 * production ones. The mark is per row; marking the whole library would make the filter reject
 * the card rows too (as in plugin/spechub). */
extern "C" int rad_plugin_reference(void) { return 0; }
extern "C" int rad_kernel_reference(int i) { return hy4k_kernel_is_reference(i); }
extern "C" int rad_kernel_schema_count(void) { return hy4k_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return hy4k_schema_at(i); }
extern "C" int rad_kernel_count(void) { return hy4k_kernel_count(); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) { return hy4k_kernel_at(i); }
