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
# ================================ SOLUTION 1 ================================ #
        # 1a) Make a soma cell
        cell = arb.make_soma_cell()
        
        # 1b) Add synapse at segment 0 at location 0.5
        loc = arb.segment_location(0, 0.5)
        cell.add_synapse(loc)
        
        # 1c) Add detector with threshold 20 mV
        cell.add_detector(loc, 20)
        
        # 1d) Add stimulus to first cell with gid 0 at t0 = 0 ms for duration of 20 ms with weight 0.01 nA
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
# ================================ SOLUTION 1 ================================ #
        # 1e) Define the source of cell with gid as the previous cell with gid-1
        #    caution: close the ring at gid 0
        src = self.num_cells()-1 if gid==0 else gid-1
        return [connection(cmem(src,0), cmem(gid,0), 0.1, 10)]

# ================================ SOLUTION 2 ================================ #
# Make parallel execution context
arb.mpi_init();
comm = arb.mpi_comm()
print(comm, '\n')

# 2a) Get hardware resources
resources = arb.proc_allocation()

# 2b) Create a context which takes the resources and an MPI communicator as input
context = arb.context(resources, comm)

print(context, '\n')

# OPTIONAL            # ================================ SOLUTION 4 ================================ #
# OPTIONAL            # 4a) Initiate meter manager and start the measurement
meters = arb.meter_manager()
meters.start(context)

# 2c) Initiate the recipe with 100 cells
n_cells = 100
recipe = ring_recipe(n_cells)

# OPTIONAL            # 4b) i) Set checkpoint 'recipe create'
meters.checkpoint('recipe create', context)

# 2d) Partition the simulation over the distributed system which takes the recipe and the context as input
decomp = arb.partition_load_balance(recipe, context)

# OPTIONAL            # 4b) ii) Set checkpoint 'load balance'
meters.checkpoint('load balance', context)

# ================================ SOLUTION 3 ================================ #
# 3a) Initiate simulation with the recipe, the domain decomposition and the context
sim = arb.simulation(recipe, decomp, context)

# OPTIONAL            # 4b) iii) Set checkpoint 'simulation init'
meters.checkpoint('simulation init', context)

# OPTIONAL            # 4c) Build the spike recorder
recorder = arb.make_spike_recorder(sim)

# 3b) Run the simulation for 2000 ms with time stepping of 0.025 ms
tSim = 2000
dt = 0.025
sim.run(tSim, dt)

# OPTIONAL            # 4b) iv) Set checkpoint 'simulation run'
meters.checkpoint('simulation run', context)

# OPTIONAL            # 4d) Make and print meter report
print(arb.make_meter_report(meters, context))

print('SPIKES:')
# OPTIONAL            # 4e) Get the recorder`s spikes
spikes = recorder.spikes

# Print at most 10 spikes
n_spikes_out = min(len(spikes), 10)
for i in range(n_spikes_out):
    spike = spikes[i]
    print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))

if n_spikes_out<len(spikes):
    print('  ...')
    spike = spikes[-1]
    print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))

# Visualization of spiking activity
# Use a raster plot to visualize spiking activity.

n_spikes = len(spikes)
tVec = np.arange(0,tSim,dt)
SpikeMat_rows = n_cells # number of cells
SpikeMat_cols = math.floor(tSim/dt)
SpikeMat = np.zeros((SpikeMat_rows, SpikeMat_cols))

# save spike trains in matrix:
# (if spike in cell n at time step k, then SpikeMat[n,k]=1, else 0)
for i in range(n_spikes):
    spike = spikes[i]
    tCur = math.floor(spike.time/dt)
    SpikeMat[spike.source.gid][tCur] = 1

for i in range(SpikeMat_rows):
    for j in range(SpikeMat_cols):
        if(SpikeMat[i,j] == 1):
            x1 = [i,i+0.5]
            x2 = [j,j]
            plot.plot(x2,x1,color = 'black')

plot.title('Spike raster plot')
plot.xlabel('Spike time (ms)')
tick = range(0,SpikeMat_cols+10000,10000)
label = range(0,tSim+250,250)
plot.xticks(tick, label)
plot.ylabel('Neuron (gid)')
plot.show()

arb.mpi_finalize();
