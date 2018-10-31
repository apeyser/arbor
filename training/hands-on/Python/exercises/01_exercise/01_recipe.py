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
        # ================================ TASK 1 ================================ #
        # a) Make a soma cell
        cell = #TODO#
        
        # b) Add synapse at segment 0 at location 0.5
        loc = #TODO#
        #TODO#
        
        # c) Add detector with threshold 20 mV
        #TODO#
        
        # d) Add stimulus to first cell with gid 0 at t0 = 0 ms for duration of 20 ms with weight 0.01 nA
        #TODO#
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
        # ================================ TASK 1 ================================ #
        # e) Define the source of cell with gid as the previous cell with gid-1
        #    caution: close the ring at gid 0
        src = #TODO#
        return [connection(cmem(src,0), cmem(gid,0), 0.1, 10)]
