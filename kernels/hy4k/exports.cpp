/* exports.cpp -- the exports block of the kernel plugin (docs/PLUGIN.md §1). */
#include "hy4k.h"
#include "rad_abi_layout.h"

#include <cstdlib>

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
/* Every card row reads and writes only its operands and the scratch the engine hands it (no
 * allocation it keeps, no counter kept between launches: the indexer tail, the Engram history and
 * the DFly2 ring are INOUT operands), so in a recorded pass the engine may start a launch beside
 * independent earlier ones, ordering it by the byte ranges it reads and writes (docs/PLUGIN.md §1,
 * abi/rad_abi.h RadKernelConcurrentFn). Unmarked, every hy4k launch was ordered on both sides, and so
 * were the stock launches next to it. A row's own launches (hc_pre's two passes, form 2g and its
 * merge, ngram ids and state) stay ordered: the engine orders every dispatch after an op's first.
 * As in the Inkling 2 plugin (radiance-inkling2-plugin, kernels/ink2k/exports.cpp, round 4).
 * HY4K_CONCURRENT=0 withdraws the promise (every launch ordered), for comparisons. */
extern "C" int rad_kernel_concurrent(int i) {
    static const bool off = [] {
        const char* v = std::getenv("HY4K_CONCURRENT");
        return v && v[0] == '0';
    }();
    return hy4k_kernel_is_reference(i) || off ? 0 : 1;
}
/* Every card row's kernels have chained forms (dev_common.h, chain_wait / chain_done), picked from
 * the `chain` parameter; their wait empties the scalar data cache behind its acquire, so the rows
 * answer 2 (abi/rad_abi.h, RadKernelChainFn). Whether the engine chains at all is the architecture's
 * probe (hyv4flash.cpp, chain_pays). Without this export every hy4k launch kept its barrier bit, and
 * so did the chained libr4d launch behind it. As in the Inkling 2 plugin (round 7).
 * HY4K_CHAIN=0 withdraws it (hy4k launches keep the bit), for comparisons. */
extern "C" int rad_kernel_chain(int i) {
    static const bool off = [] {
        const char* v = std::getenv("HY4K_CHAIN");
        return v && v[0] == '0';
    }();
    return hy4k_kernel_is_reference(i) || off ? 0 : 2;
}
