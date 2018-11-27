import sys
import numpy as np
import math
import matplotlib.pyplot as plot

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

n_cells = 100
recipe = ring_recipe(n_cells)

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

tSim = 2000
dt = 0.025
sim.run(tSim, dt)

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

#Visualization of spiking activity
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
