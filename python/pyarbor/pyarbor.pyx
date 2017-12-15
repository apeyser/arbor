from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython cimport bool as cbool
from libc.stdint cimport uint32_t
from cpython cimport bool

######## utility #############################
cdef class ArgvList:
    cdef vector[string] argv
    cdef int length
    cdef char** cargv
    
    def __cinit__(self, list argv):
        self.length = len(argv)
        self.argv = <vector[string]> argv

        self.cargv = <char**> malloc(len(argv)*sizeof(char*))
        for i in range(self.length):
            self.cargv[i] = &self.argv[i][0]

    def __dealloc__(self):
        del self.gpg
        free(self.cargv)

######## C++ objects ###############################
#
cdef extern from "<profiling/meter_manager.hpp>" namespace "arb::util":
    cdef cppclass meter_manager:
        meter_manager() except +
        void start() except +
        void checkpoint(string) except +

cdef extern from "<communication/global_policy.hpp>" namespace "arb::communication":
    cdef cppclass global_policy_guard:
        global_policy_guard(int argc, char**argv)

    cdef enum GlobalPolicyKind "arb::communication::global_policy_kind":
        serial "arb::communication::global_policy_kind::serial"
        mpi    "arb::communication::global_policy_kind::mpi"
        dryrun "arb::communication::global_policy_kind::dryrun"

cdef extern from "<communication/global_policy.hpp>" namespace "arb::communication::global_policy":
    int gp_id "arb::communication::global_policy::id" ()
    int gp_size "arb::communication::global_policy::size" ()
    GlobalPolicyKind gp_kind "arb::communication::global_policy::kind" ()
    void gp_setup "arb::communication::global_policy::setup" (int argc, char** argv)
    void gp_teardown "arb::communication::global_policy::teardown" ()

cdef extern from "<hardware/node_info.hpp>" namespace "arb::hw":
    cdef cppclass node_info:
        node_info()
        node_info(unsigned c, unsigned g)

        unsigned num_cpu_cores
        unsigned num_gpus

######### PyObjects #################
cdef class MeterManager:
    cdef meter_manager mm

    def start(self):
        self.mm.start()
    def checkpoint(self, str name):
        self.mm.checkpoint(<string> name)

cdef class GlobalPolicy:
    @staticmethod
    def id():
        return gp_id()
    @staticmethod
    def size():
        return gp_size()
    @staticmethod
    def kind():
        return gp_kind()
    @staticmethod
    def setup(list argv):
        cdef ArgvList argv_list = ArgvList(argv)
        gp_setup(argv_list.length, argv_list.cargv)

cdef class GlobalPolicyGuard:
    cdef global_policy_guard* gpg
    
    def __cinit__(self, list argv):
        cdef ArgvList argv_list = ArgvList(argv)
        self.gpg = new global_policy_guard(argv_list.length,
                                           argv_list.cargv)

    def __dealloc__(self):
        del self.gpg

cdef class NodeInfo:
    cdef node_info nd

    def __cinit__(self, num_cpu_cores=None, num_gpus=None):
        cdef unsigned c
        cdef unsigned g
        
        if num_cpu_cores == None and num_gpus == None:
            self.nd = node_info()
        else:
            c = <unsigned> num_cpu_cores
            g = 1 if <bool> num_gpus else 0
            self.nd = node_info(c, g)

    @property
    def num_cpu_cores(self):
        return self.nd.num_cpu_cores
    @num_cpu_cores.setter
    def num_cpu_cores(self, int value):
        self.nd.num_cpu_cores = <unsigned> value

    @property
    def num_gpus(self):
        return self.nd.num_gpus
    @num_gpus.setter
    def num_gpus(self, bool value):
        self.nd.num_gpus = 1 if <unsigned> value else 0

