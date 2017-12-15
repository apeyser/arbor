#!python3

import pyarbor

class Miniapp:
    ################## parameters ###########################
    
    cells = 1000
    dry_run_ranks = 1
    syn_type = "expsyn"
    compartments_per_segment = 100
    #util::optional<std::string> morphologies;
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
    trace_prefix = "trace_";
    #util::optional<unsigned> trace_max_gid; // Only make traces up to this gid.
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

    ####################### Run miniapp ##############
    def run(self):
        gpg = pyarbor.GlobalPolicyGuard(["pyarbor"])
        try: self._run(gpg)
        except: raise # exception printing is automatic

    # internal to Run ##############
    def _run(self, gpg):
        meters = pyarbor.MeterManager()
        meters.start()

        # Some mask stream thing goes here

        if pyarbor.GlobalPolicy.kind() == pyarbor.GlobalPolicyKind.dryrun:
            cells_per_rank = self.cells/self.dry_run_ranks
            if self.cells % self.dry_run_ranks:
                ++cells_per_rank
                self.cells = cells_per_rank*self.dry_run_ranks

            pyarbor.GlobalPolicy.set_sizes(self.dry_run_ranks, self.cells_per_rank)
    
        nd = pyarbor.HW.NodeInfo()
        nd.num_cpu_cores = pyarbor.Threading.num_threads()
        nd.num_gpus = pyarbor.HW.num_gpus() > 0
        banner(nd)

        meters.checkpoint("setup")

        # ....
