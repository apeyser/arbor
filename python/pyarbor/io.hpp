#pragma once

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common_types.hpp>
#include <util/optional.hpp>
#include <util/path.hpp>

#include "pyarbor-base.hpp"

namespace arb {
namespace io {

/// Helper function for loading a vector of spike times from file
/// Spike times are expected to be in milli seconds floating points
/// On spike-time per line

std::vector<time_type>  get_parsed_spike_times_from_path(arb::util::path path);

} // namespace io
} // namespace arb
