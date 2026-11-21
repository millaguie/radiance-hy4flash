/* hy4k.h -- internal contract of the Hy4-flash kernel plugin: schemas and rows. */
#ifndef HY4K_H
#define HY4K_H

#include "rad_abi.h"
#include "rad_plugin.h"

int                 hy4k_schema_count();
const RadOpSchema*  hy4k_schema_at(int i);
int                 hy4k_kernel_count();
const RadKernelInfo* hy4k_kernel_at(int i);
int                 hy4k_kernel_is_reference(int i);

#endif
