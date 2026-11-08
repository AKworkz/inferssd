// SPDX-License-Identifier: Apache-2.0
#pragma once
// Entry points of this backend's shared library that its own objects and the tests link by name.
#if defined(_WIN32)
#define SSD_CUDA_EXPORT __declspec(dllexport)
#else
#define SSD_CUDA_EXPORT __attribute__((visibility("default")))
#endif
