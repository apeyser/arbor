#!/usr/bin/python3

import pyarbor
import sys

class Miniapp:
    
    ################## parameters ###########################
    #
    cells = 1000
    dry_run_ranks = 1
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
    trace_max_gid = None // Only make traces up to this gid.
    trace_format = "json" # Support only 'json' and 'csv'.

    # Parameters for spike output.
    spike_file_output = false
    single_file_per_rank = false
    over_write = true
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
        gpg = pyarbor.GlobalPolicyGuard(["pyarbor"])
        try: self._run(gpg)
        except: raise # exception printing is automatic

    # internal to Run ##############
    def _run(self, gpg):
        meters = pyarbor.MeterManager()
        meters.start()

        # Some mask stream thing goes here

        if pyarbor.GlobalPolicy.kind() \
           == pyarbor.GlobalPolicyKind.dryrun:
            cells_per_rank = self.cells/self.dry_run_ranks
            if self.cells % self.dry_run_ranks:
                ++cells_per_rank
                self.cells = cells_per_rank*self.dry_run_ranks

            pyarbor.GlobalPolicy.set_sizes(self.dry_run_ranks,
                                           self.cells_per_rank)
    
        nd = pyarbor.HW.NodeInfo()
        nd.num_cpu_cores = pyarbor.Threading.num_threads()
        nd.num_gpus = pyarbor.HW.num_gpus() > 0
        self.banner(nd)

        meters.checkpoint("setup")

        pdist = pyarbor.ProbeDistribution()
        pdist.proportion = self.probe_ratio
        pdist.all_segments = not self.probe_soma_only

        recipe = self.make_recipe(pdist)
        
        if self.report_compartments:
            self.report_compartment_stats(recipe)

        exporter = pyarbor.FileExporter(
            self.file_name,
            self.output_path,
            self.file_extension,
            self.over_write)
        
        decomp = pyarbor.partition_load_balance(recipe, nd)
        model = pyarbor.Model(recipe, decomp)

        sample_traces = []
        for g in decomp.groups:
            if g.kind != pyarbor.CellKinds.cable1d_neuron:
                continue
            for gid in g.gids:
                if self.trace_max_gid and gid > self.trace_max_gid:
                    continue
                for j in range(0, recipe.num_probes(gid)):
                    cell_member = pyarbor.CellMemberType(gid, j)
                    probe_info = recipe.get_probe(cell_member)
                    trace = make_trace(probe_info)
                    sample_traces.append(trace)
        
        ssched = pyarbor.Schedule.regular_schedule(self.sample_dt)
        for trace in sample_traces:
            probe = pyarbor.Probe.one_probe(trace.probe_id)
            sampler = trace.make_simple_sampler()
            m.add_sampler(probe, ssched, sampler)

        if self.bin_dt == 0:
            binning_policy = pyarbor.BinningKind.none
        else if self.bin_regular:
            binning_policy = pyarbor.BinningKind.regular
        else:
            binning_policy = pyarbor.BinningKind.following
        m.set_binning_policy(binning_policy, self.bin_dt);

        if options.spike_file_output:
            if options.single_file_per_rank:
                model.set_local_spike_callback(exporter)
            elif pyarbor.GlobalPolicy.id() == 0:
                model.set_global_spike_callback(exporter)

        meters.checkpoint("model-init")

        m.run(self.tfinal, self.dt)

        meters.checkpoint("model-simulate")

        pyarbor.Util.profiler_output(0.001, self.profile_only_zero)
        self.write("there were {} spikes\n".format(m.num_spikes()))

        # check format of trace_format json, csv
        if self.trace_format == 'json':
            write_trace = pyarbor.Util.write_trace_json
        else:
            write_trace = pyarbor.Util.write_trace_csv
            
        for trace in sample_traces:
            write_trace(trace, self.trace_prefix)

        report = pyarbor.Util.make_meter_report(meters)
        self.write(pyarbor.Util.to_string(report))
        if pyarbor.GlobalPolicy.id() == 0:
            with open("meters.json", "w") as fid:
                fid.write(pyarbor.Util.to_json(report) + "\n")

    def write(self, string):
        if pyarbor.GlobalPolicy.id() == 0:
            sys.stdout.write(string)

    def banner(self, nd):
        self.write("""\
==========================================
  Arbor miniapp\n";
  - distributed : {} ({})
  - threads     : {} ({})
  - gpus        : {}
==========================================
""".format(
    pyarbor.GlobalPolicy.size(),
    pyarbor.GlobalPolicy.kind(),
    nd.num_cpu_cores,
    pyarbor.Threading.description(),
    nd.num_gpus
 ))

    def make_recipe(self, pdist):
        p = pyarbor.BasicRecipeParam()
        
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
        ncomp_min = sys.maxint
        ncomp_max = 0

        for i in range(ncell):
            ncomp = 0
            c = recipe.get_cell_description(i)
            if isinstance(c, pyarbor.Cell):
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
        
        if addr.kind == pyarbor.ProbeKind.membrane_voltage:
            name = "v"
            units = "mV"
        elif addr.kind == pyarbor.ProbeKind.membrane_current:
            name = "i"
            units = "mA/cm²"

        if addr.location.segment:
            name += "dend"
        else:
            name += "soma"

        return pyarbor.SampleTrace(probe.id, name, units)
