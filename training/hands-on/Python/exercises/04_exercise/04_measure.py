import sys
import pyarb as arb
from pyarb import connection
from pyarb import cell_member as cmem

# A recipe, that describes the cells and network of a model, can be defined
# in python by implementing the pyarb.recipe interface.
class ring_recipe(arb.recipe):
    
    def __init__(self, n=4):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arb.recipe.__init__(self)
        self.ncells = n
    
    # The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells
    
    # The cell_description method returns a cell
    def cell_description(self, gid):
        
        cell = arb.make_soma_cell()
        loc = arb.segment_location(0, 0.5)
        cell.add_synapse(loc)
        cell.add_detector(loc, 20)
        
        if gid==0:
            cell.add_stimulus(loc, 0, 20, 0.01)
        return cell
    
    def num_targets(self, gid):
        return 1
    
    def num_sources(self, gid):
        return 1
    
    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def kind(self, gid):
        return arb.cell_kind.cable1d
    
    # Make a ring network
    def connections_on(self, gid):
        src = self.num_cells()-1 if gid==0 else gid-1
        return [connection(cmem(src,0), cmem(gid,0), 0.1, 10)]

# Make parallel execution context
arb.mpi_init();
comm = arb.mpi_comm()

print(comm, '\n')

resources = arb.proc_allocation()
context = arb.context(resources, comm)

print(context, '\n')

# OPTIONAL            # ================================ TASK 4 ================================ #
# OPTIONAL            # 4a) Initiate meter manager and start the measurement
meters = #TODO#
#TODO#

recipe = ring_recipe(100)

# OPTIONAL            # 4b) i) Set checkpoint 'recipe create'
#TODO#

decomp = arb.partition_load_balance(recipe, context)

# OPTIONAL            # 4b) ii) Set checkpoint 'load balance'
#TODO#

sim = arb.simulation(recipe, decomp, context)

# OPTIONAL            # 4b) iii) Set checkpoint 'simulation init'
#TODO#

# OPTIONAL            # 4c) Build the spike recorder
#TODO

sim.run(2000, 0.025)

# OPTIONAL            # 4b) iv) Set checkpoint 'simulation run'
#TODO#

# OPTIONAL            # 4d) Make and print meter report
#TODO#

print('SPIKES:')
# OPTIONAL            # 4e) Get the recorder`s spikes
spikes = #TODO#

# Print at most 10 spikes
n_spikes_out = min(len(spikes), 10)
for i in range(n_spikes_out):
    spike = spikes[i]
    print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))

if n_spikes_out<len(spikes):
    print('  ...')
    spike = spikes[-1]
    print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))

arb.mpi_finalize();
