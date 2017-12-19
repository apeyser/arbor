#pragma once

/*
 * Store trace data from samplers with metadata.
 */

#include <string>
#include <vector>
#include <sstream>

#include <common_types.hpp>
#include <simple_sampler.hpp>
#include <profiling/meter_manager.hpp>

using trace_entry = arb::trace_entry<double>;
using trace_data = arb::trace_data<double>;
using simple_sampler = arb::simple_sampler<double>;

inline simple_sampler make_simple_sampler(trace_data& trace) {
    return arb::make_simple_sampler(trace);
}

struct sample_trace {
    arb::cell_member_type probe_id;
    std::string name;
    std::string units;
    trace_data samples;
};

void write_trace_csv(const sample_trace& trace, const std::string& prefix);
void write_trace_json(const sample_trace& trace, const std::string& prefix);

inline std::string to_string(const arb::util::meter_report& mr) {
    std::ostringstream s;
    s << mr;
    return s.str();
}
