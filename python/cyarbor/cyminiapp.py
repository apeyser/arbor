#!/usr/bin/python3

import cyarbor
import sys

class Miniapp:
    
    ################## parameters ###########################
    #
    cells = 1000
    synapses_per_cell = 500
    syn_type = "expsyn"
    compartments_per_segment = 100
    morphologies = None # string
    morph_rr = False # False => pick morphologies randomly, true => pick morphologies round-robin.

    # Network type (default is rgraph):
    all_to_all = False
    ring = False

    # Simulation running parameters:
    tfinal = 100.0
    dt = 0.025
    bin_regular = False # False => use 'following' instead of 'regular'.
    bin_dt = 0.0025    # 0 => no binning.

    # Probe/sampling specification.
    sample_dt = 0.1
    probe_soma_only = False
    probe_ratio = 0  # Proportion of cells to probe.
    trace_prefix = "trace_"
    trace_max_gid = None # Only make traces up to this gid.
    trace_format = "json" # Support only 'json' and 'csv'.

    # Parameters for spike output.
    spike_file_output = False
    single_file_per_rank = False
    over_write = True
    output_path = "./"
    file_name = "spikes"
    file_extension = "gdf"

    # Parameters for spike input.
    spike_file_input = False
    input_spike_path = None # Path to file with spikes

    # Dry run parameters (pertinent only when built with 'dryrun' distrib model).
    dry_run_ranks = 1

    # Turn on/off profiling output for all ranks.
    profile_only_zero = False

    # Report (inefficiently) on number of cell compartments in sim.
    report_compartments = False

    # Be more verbose with informational messages.
    verbose = False
    
    #
    ####################### Run miniapp ##############
    #
    def run(self):
        try:
            gpg = cyarbor.GlobalPolicyGuard(["cyarbor"])
            self._run(gpg)
        except: raise # exception printing is automatic

    # internal to Run ##############
    def _run(self, gpg):
        meters = cyarbor.MeterManager()
        meters.start()

        # Some mask stream thing goes here

        if cyarbor.GlobalPolicy.kind() \
           == cyarbor.GlobalPolicyKind.dryrun:
            cells_per_rank = self.cells/self.dry_run_ranks
            if self.cells % self.dry_run_ranks:
                ++cells_per_rank
                self.cells = cells_per_rank*self.dry_run_ranks

            cyarbor.GlobalPolicy.set_sizes(self.dry_run_ranks,
                                           self.cells_per_rank)
    
        nd = cyarbor.HW.NodeInfo()
        nd.num_cpu_cores = cyarbor.Threading.num_threads()
        nd.num_gpus = cyarbor.HW.num_gpus() > 0
        self.banner(nd)

        meters.checkpoint("setup")

        pdist = cyarbor.ProbeDistribution()
        pdist.proportion = self.probe_ratio
        pdist.all_segments = not self.probe_soma_only

        recipe = self.make_recipe(pdist)
        
        if self.report_compartments:
            self.report_compartment_stats(recipe)

        exporter = cyarbor.FileExporter(
            self.file_name,
            self.output_path,
            self.file_extension,
            self.over_write)
        
        decomp = cyarbor.Decomp(recipe, nd)
        model = cyarbor.Model(recipe, decomp)

        sample_traces = []
        for g in decomp.groups:
            if g.kind != cyarbor.CellKind.cable1d_neuron:
                continue
            for gid in g.gids:
                if self.trace_max_gid and gid > self.trace_max_gid:
                    continue
                for j in range(0, recipe.num_probes(gid)):
                    cell_member = cyarbor.CellMemberType(gid, j)
                    probe_info = recipe.get_probe(cell_member)
                    trace = make_trace(probe_info)
                    sample_traces.append(trace)
        
        ssched = cyarbor.Schedule.regular_schedule(self.sample_dt)
        for trace in sample_traces:
            probe = cyarbor.Probe.one_probe(trace.probe_id)
            sampler = trace.make_simple_sampler()
            model.add_sampler(probe, ssched, sampler)

        if self.bin_dt == 0:
            binning_policy = cyarbor.BinningKind.none
        elif self.bin_regular:
            binning_policy = cyarbor.BinningKind.regular
        else:
            binning_policy = cyarbor.BinningKind.following
        model.set_binning_policy(binning_policy, self.bin_dt)

        if self.spike_file_output:
            if self.single_file_per_rank:
                model.set_local_spike_callback(exporter)
            elif cyarbor.GlobalPolicy.id() == 0:
                model.set_global_spike_callback(exporter)

        meters.checkpoint("model-init")

        model.run(self.tfinal, self.dt)

        meters.checkpoint("model-simulate")

        cyarbor.Util.profiler_output(0.001, self.profile_only_zero)
        self.write("there were {} spikes\n".format(model.num_spikes()))

        # check format of trace_format json, csv
        if self.trace_format == 'json':
            write_trace = cyarbor.Util.write_trace_json
        else:
            write_trace = cyarbor.Util.write_trace_csv
            
        for trace in sample_traces:
            write_trace(trace, self.trace_prefix)

        report = cyarbor.Util.make_meter_report(meters)
        self.write(cyarbor.Util.to_string(report))
        if cyarbor.GlobalPolicy.id() == 0:
            with open("meters.json", "w") as fid:
                fid.write(cyarbor.Util.to_json(report) + "\n")

    def write(self, string):
        if cyarbor.GlobalPolicy.id() == 0:
            sys.stdout.write(string)

    def banner(self, nd):
        self.write("""\
==========================================
  Arbor miniapp
  - distributed : {} ({})
  - threads     : {} ({})
  - gpus        : {}
==========================================
""".format(
    cyarbor.GlobalPolicy.size(),
    cyarbor.GlobalPolicy.kind(),
    nd.num_cpu_cores,
    cyarbor.Threading.description(),
    nd.num_gpus
 ))

    def make_recipe(self, pdist):
        p = cyarbor.BasicRecipeParam()
        
        if self.morphologies:
            self.write("loading morphologies...\n")
            p.morphologies.clear();
            p.morphologies.load_swc_morphology_glob(self.morphologies)
            self.write(
                "loading morphologies: {} loaded.\n"\
                .format(p.morphologies.size())
            )
            
        p.morphology_round_robin = self.morph_rr;
        p.num_compartments = self.compartments_per_segment;
        if self.all_to_all:
            p.num_synapses = self.cells-1
        else:
            p.num_synapses = self.synapses_per_cell
        p.synapse_type = self.syn_type

        if self.spike_file_input:
            p.input_spike_path = self.input_spike_path

        if self.all_to_all: make = p.make_basic_kgraph_recipe
        elif self.ring:     make = p.make_basic_ring_recipe
        else:               make = p.make_basic_rgraph_recipe
        return make(self.cells, pdist)
    
    def report_compartment_stats(self, recipe):
        ncell = recipe.num_cells()
        ncomp_total = 0
        ncomp_min = sys.maxsize
        ncomp_max = 0

        for i in range(ncell):
            ncomp = 0
            c = recipe.get_cell_description(i)
            if isinstance(c, cyarbor.Cell):
                ncomp = c.num_compartments()
                ncomp_total += ncomp;
                ncomp_min = min(ncomp_min, ncomp);
                ncomp_max = max(ncomp_max, ncomp);

        self.write("compartments/cell: min={}; max={}; mean={}\n"\
                   .format(ncomp_min, ncomp_max,
                           ncomp_total/ncell))

    def make_trace(self, probe):
        name = ""
        units = ""

        addr = probe.address
        if not isinstance(addr, CellProbeAddress):
            raise TypeError()
        
        if addr.kind == cyarbor.ProbeKind.membrane_voltage:
            name = "v"
            units = "mV"
        elif addr.kind == cyarbor.ProbeKind.membrane_current:
            name = "i"
            units = "mA/cm²"

        if addr.location.segment:
            name += "dend"
        else:
            name += "soma"

        return cyarbor.SampleTrace(probe.id, name, units)
