#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <recipe.hpp>
#include <util/optional.hpp>
#include <model.hpp>
#include <io/exporter_spike_file.hpp>
#include <communication/global_policy.hpp>

#include "morphology_pool.hpp"

// miniapp-specific recipes

namespace arb {

struct probe_distribution {
    float proportion = 1.f; // what proportion of cells should get probes?
    bool all_segments = true;    // false => soma only
    bool membrane_voltage = true;
    bool membrane_current = true;
};

struct basic_recipe_param {
    // `num_compartments` is the number of compartments to place in each
    // unbranched section of the morphology, A value of zero indicates that
    // the number of compartments should equal the number of piecewise
    // linear segments in the morphology description of that branch.
    unsigned num_compartments = 1;

    // Total number of synapses on each cell.
    unsigned num_synapses = 1;

    std::string synapse_type = "expsyn";
    float min_connection_delay_ms = 20.0;
    float mean_connection_delay_ms = 20.75;
    float syn_weight_per_cell = 0.3;

    morphology_pool morphologies = default_morphology_pool;

    // If true, iterate through morphologies rather than select randomly.
    bool morphology_round_robin = false;

    // If set we are importing the spikes injected in the network from file
    // instead of a single spike at t==0
    util::optional<std::string> input_spike_path;  // Path to file with spikes
};

typedef util::optional<std::string> opt_string; //for cython

std::unique_ptr<recipe> make_basic_ring_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

std::unique_ptr<recipe> make_basic_kgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

std::unique_ptr<recipe> make_basic_rgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

using file_export_type = io::exporter_spike_file<global_policy>;
using spike_export_function = model::spike_export_function;

spike_export_function file_exporter(
    string file_name,
    string output_path,
    string file_extension,
    bool over_write)
{
    unique_ptr<file_export_type> file_exporter
        = util::make_unique<file_export_type>(
            file_name,
            output_path,
            file_extension,
            over_write);
    return [&](const std::vector<spike>& spikes) {
        file_exporter->output(spikes);
    };
}

} // namespace arb
