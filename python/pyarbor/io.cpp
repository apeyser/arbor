#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <tclap/CmdLine.h>
#include <json/json.hpp>

#include <util/meta.hpp>
#include <util/optional.hpp>
#include <util/strprintf.hpp>

#include "io.hpp"

namespace arb {
namespace io {
/// Parse spike times from a stream
/// A single spike per line, trailing whitespace is ignore
/// Throws a usage error when parsing fails
///
/// Returns a vector of time_type

std::vector<time_type> parse_spike_times_from_stream(std::ifstream & fid) {
    std::vector<time_type> times;
    std::string line;
    while (std::getline(fid, line)) {
        std::stringstream s(line);

        time_type t;
        s >> t >> std::ws;

        if (!s || s.peek() != EOF) {
            throw std::runtime_error( util::strprintf(
                    "Unable to parse spike file on line %d: \"%s\"\n",
                    times.size(), line));
        }

        times.push_back(t);
    }

    return times;
}

/// Parse spike times from a file supplied in path
/// A single spike per line, trailing white space is ignored
/// Throws a usage error when opening file or parsing fails
///
/// Returns a vector of time_type

std::vector<time_type> get_parsed_spike_times_from_path(arb::util::path path) {
    std::ifstream fid(path);
    if (!fid) {
        throw std::runtime_error(util::strprintf(
            "Unable to parse spike file: \"%s\"\n", path.c_str()));
    }

    return parse_spike_times_from_stream(fid);
}

} // namespace io
} // namespace arb
