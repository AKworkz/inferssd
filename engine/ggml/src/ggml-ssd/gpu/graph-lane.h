// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "provider.h"

namespace ssd { namespace gpu {
// A GPU expert lane for any ggml backend whose device shares host memory and can read it in place. It serves grouped
// prefill only, through the ordinary backend interface: a graph of the chain (up, gate, the activation, down) over the
// layer's experts as their reads land, reading the weights from the read ring where they are. It keeps nothing across
// layers, so decode stays on the CPU and the host ledger is as without a GPU tier. The sink's cost table moves an expert
// here when the measured lines say so.
const provider_v1 * graph_lane_provider();
}} // namespace ssd::gpu
