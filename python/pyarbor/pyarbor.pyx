from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from memory cimport unique_ptr, shared_ptr
from cpython cimport bool as cbool
from libc.stdint cimport uint32_t
from cpython cimport bool

######## utility #############################
cdef class ArgvList:
    cdef vector[string] argv
    cdef int length
    cdef char** cargv = NULL
    
    def __cinit__(self, list argv):
        self.length = len(argv)
        self.argv = <vector[string]> argv

        self.cargv = <char**> malloc(len(argv)*sizeof(char*))
        for i in range(self.length):
            self.cargv[i] = &self.argv[i][0]

    def __dealloc__(self):
        if self.cargv:
            free(self.cargv)
            self.cargv = NULL
            

######## C++ objects ###############################
#
cdef extern from "<common_types.hpp>" namespace "arb":
    ctypedef uint32_t cell_gid_type
    ctypedef uint32_t cell_size_type
    ctypedef uint32_t cell_lid_type
    ctypedef uint32_t cell_local_size_type
    ctypedef int probe_tag
    ctypedef float time_type

    cdef enum CellKind
        cable1d_neuron
        regular_spike_source
        data_spike_source

    cdef struct cell_member_type:
        cell_gid_type gid
        cell_lid_type index

cdef extern from "<communication/global_policy.hpp>" namespace "arb::communication":
    cdef cppclass global_policy_guard:
        global_policy_guard(int argc, char**argv)

    cdef enum GlobalPolicyKind "arb::communication::global_policy_kind":
        serial "arb::communication::global_policy_kind::serial"
        mpi    "arb::communication::global_policy_kind::mpi"
        dryrun "arb::communication::global_policy_kind::dryrun"

cdef extern from "<communication/global_policy.hpp>":
    int gp_id "arb::communication::global_policy::id" \
        ()
    int gp_size "arb::communication::global_policy::size" \
        ()
    GlobalPolicyKind gp_kind "arb::communication::global_policy::kind" \
        ()
    void gp_setup "arb::communication::global_policy::setup" \
        (int argc, char** argv)
    void gp_teardown "arb::communication::global_policy::teardown" \
        ()

cdef extern from "<json/json.hpp>" namespace "nlohmann":
    cdef cppclass json:
        string dump()
        
cdef extern from "<profiling/meter_manager.hpp>" namespace "arb::util":
    cdef cppclass meter_manager:
        meter_manager() except +
        void start() except +
        void checkpoint(string) except +

    cdef struct measurement:
        string name
        string units
        vector[vector[double]] measurements

    cdef struct meter_report:
        vector[string] checkpoints
        unsigned num_domains
        unsigned num_hosts
        GlobalPolicyKind communication_policy;
        vector[measurement] meters;
        vector[string] hosts
        
    cdef meter_report make_meter_report(meter_manager) except+
    json to_json(meter_report) except+

cdef extern from "<hardware/node_info.hpp>" namespace "arb::hw":
    cdef cppclass node_info:
        node_info()
        node_info(unsigned c, unsigned g)

        unsigned num_cpu_cores
        unsigned num_gpus

cdef extern from "<util/optional.hpp>" namespace "arb::util":
    cdef struct nothing_t:
        pass
    cdef nothing_t nothing

cdef extern from "<util/unique_any.hpp>" namespace "arb::util":
    cdef cppclass unique_any:
        cbool has_value()
    T* any_cast[T](unique_any*)

cdef extern from "<util/any.hpp>" namespace "arb::util":
    cdef cppclass any:
        cbool has_value()
    cdef T* any_cast[T](any*) except+
    
cdef extern from "<recipe.hpp>" namespace "arb":
    cdef struct probe_info:
        cdef cell_member_type id
        cdef probe_tag tag
        any address

    cdef cppclass recipe:
        cell_size_type num_cells() except+
        unique_any get_cell_description(cell_gid_type) except+
        probe_info get_probe(cell_member_type) except+

cdef extern from "<segment.hpp>" namespace "arb":
    cdef struct segment_location:
        cell_lid_type segment
        double position

cdef extern from "<cell.hpp>" namespace "arb":
    cdef cppclass cell:
        cell_local_size_type num_compartments() except+
    cdef enum ProbeKind  "arb::cell_probe_address::probe_kind":
        membrane_voltage "arb::cell_probe_address::membrane_voltage"
        membrane_current "arb::cell_probe_address::membrane_current"
    cdef struct cell_probe_address:
        segment_location location
        probe_kind kind

cdef extern from "morphology_pool.hpp" namespace "arb":
    cdef cppclass morphology_pool:
        size_t size() except+
        void clear() except+
    cdef morphology_pool default_morphology_pool
    void load_swc_morphology_glob(morphology_pool, string) except+

cdef extern from "miniapp_recipes.hpp" namespace "arb":
    cdef struct probe_distribution:
        float proportion
        cbool all_segments
        cbool membrane_voltage
        cbool membrane_current

    cdef cppclass opt_string:
        opt_string()
        opt_string(nothing_t)
        opt_string(string)
        opt_string operator=(string)
        operator bool()
        string get() except+
        void reset() except+
        
    struct basic_recipe_param:
        unsigned num_compartments
        unsigned num_synapses 
        string synapse_type
        float min_connection_delay_ms
        float mean_connection_delay_ms
        float syn_weight_per_cell
        morphology_pool morphologies
        cbool morphology_round_robin
        opt_string input_spike_path

    unique_ptr[recipe] make_basic_ring_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist)
    unique_ptr[recipe] make_basic_ring_recipe(
        cell_gid_type ncell,
        basic_recipe_param param)

    unique_ptr[recipe] make_basic_kgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution);
    unique_ptr[recipe] make_basic_kgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param)

    unique_ptr[recipe] make_basic_rgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param,
        probe_distribution pdist)
    unique_ptr[recipe] make_basic_rgraph_recipe(
        cell_gid_type ncell,
        basic_recipe_param param)

cdef extern from "<threading/threading.hpp>" namespace "arb::threading":
    string thr_description "arb::threading::description" \
        () except+

cdef extern from "miniapp_recipes.hpp" namespace "arb":
    cdef cppclass spike_export_function:
        pass
    spike_export_function file_exporter(string, string, string, cbool)

cdef extern from "<domain_decomposition>" namespace "arb":
    cdef cppclass group_description:
        CellKind kind
        vector[cell_gid_type] gids
        
    cdef cppclass domain_decomposition:
        vector[group_description] groups

cdef extern from "<load_balance.hpp>" namespace "arb":
    cdef domain_decomposition partition_load_balance(
        recipe, node_info
    ) except+

cdef extern from "trace.hpp":
    cdef cppclass trace_data:
        pass
    cdef struct sample_trace:
        cell_member_type probe_id
        string name
        string units
        trace_data samples
        
    void write_trace_csv(sample_trace, string)
    void write_trace_json(sample_trace, string)
    string to_string(meter_report)

    cdef cppclass simple_sampler:
        pass
    simple_sampler make_simple_sampler(trace_data) except+

cdef extern from "<schedule.hpp>" namespace "arb":
    cdef cppclass schedule:
        pass
    schedule regular_schedule(time_type dt) except+

cdef extern from "<sampling.hpp>" namespace "arb":
    ctypedef size_t sampler_association_handle
    cdef cppclass cell_member_predicate:
        pass
    cell_member_predicate one_probe(cell_member_type) except+

cdef extern from "<model.hpp>" namespace "arb":
    cdef cppclass model:
        model(recipe, domain_decomposition) except+
        void reset() except+
        time_type run(time_type, time_type) except+
        sampler_association_handler add_sampler(
            cell_member_predicate,
            schedule,
            sampler_function) except+
        void set_binning_policy(BinningKind, time_type) except+
        void set_global_spike_callback(spike_export_function) except+
        void set_local_spike_callback(spike_export_function) except+

cdef extern from "<event_binner.hpp>" namespace "arb":
    cdef enum BinningKind "arb::binning_kind"
        none      "arb::binning_kind::none"
        regular   "arb::binning_kind::regular"
        following "arb::binning_kind::following"

cdef extern from "<profiling/profiler.hpp>" namespace "arb":
    cdef profiler_output(double, cbool)

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

cdef class Threading:
    @staticmethod
    def description():
        str r = <str> thr_description()
        return r

cdef class GlobalPolicyGuard:
    cdef global_policy_guard* gpg = NULL
    
    def __cinit__(self, list argv):
        cdef ArgvList argv_list = ArgvList(argv)
        self.gpg = new global_policy_guard(argv_list.length,
                                           argv_list.cargv)

    def __dealloc__(self):
        if self.gpg:
            del self.gpg
            self.gpg = NULL

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

cdef class ProbeDistribution:
    cdef probe_distribution pd

    @property
    def proportion(self):
        return pd.proportion
    @proportion.setter
    def proportion(self, float proportion):
        self.pd.proportion = proportion

    @property
    def all_segments(self):
        return pd.all_segments
    @all_segments.setter
    def all_segments(self, bool all_segments):
        self.pd.all_segments = all_segments

    @property
    def membrane_voltage(self):
        return pd.membrane_voltage
    @membrane_voltage.setter
    def membrane_voltage(self, bool membrane_voltage):
        self.pd.membrane_voltage = membrane_voltage

    @property
    def membrane_current(self):
        return pd.membrane_current
    @membrane_current.setter
    def membrane_current(self, bool membrane_current):
        self.pd.membrane_current = membrane_current

ctypedef unique_ptr[recipe] (*_make_recipe_function)(
    cell_gid_type,
    base_recipe_param,
    probe_distribution)

cdef class Cell:
    cdef cell* c = NULL

    def __cinit__(self, cell* c):
        self.c = c

    def __dealloc__(self):
        if self.c:
            del self.c
            self.c = NULL

    def num_compartments(self):
        return self.c.num_compartments()

cdef class MorphologyPool:
    cdef morphology_pool* mp = &default_morphology_pool

    def __cinit__(self, morphology_pool* mp):
        self.mp = mp

    def size(self):
        return self.mp.size()

    def clear(self):
        self.mp.clear()

    def load_swc_morphology_glob(self, str morphologies):
        load_swc_morphology_glob(self.mp[0],
                                 <string> morphologies)
        
cdef class BasicRecipeParam:
    cdef basic_recipe_param p
    cdef MorphologyPool mp

    def __cinit__(self):
        self.mp = MorphologyPool(&self.p.morphology_pool)

    @property
    def morphologies(self):
        return self.mp

    @property
    def morphology_round_robin(self):
        return <bool> self.p.morphology_round_robin
        
    @morphology_round_robin.setter
    def morphology_round_robin(self, bool value):
        self.p.morphology_round_robin = <cbool> value

    @property
    def num_compartments(self):
        return self.p.num_compartments
        
    @num_compartments.setter
    def num_compartments(self, int value):
        self.p.num_compartments = <unsigned> value

    @property
    def num_synapses(self):
        return self.p.num_synapses
        
    @num_synapses.setter
    def num_synapses(self, int value):
        self.p.num_synapses = <unsigned> value

    @property
    def synapse_type(self):
        return self.p.synapse_type
        
    @synapse_type.setter
    def synapse_type(self, str value):
        self.p.synapse_type = <string> value

    @property
    def input_spike_path(self):
        if <bool> self.p.input_spike_path:
            return <str> self.p.input_spike_path.get()
        return None
        
    @input_spike_path.setter
    def input_spike_path(self, str value):
        self.p.input_spike_path = <string> value

    cdef _make_recipe(self,
                      cell_gid_type ncell,
                      pdist,
                      _make_recipe_function func):
        cdef unique_ptr[recipe] r
        if pdist:
            ProbeDistribution pd = <ProbeDistribution> pdist
            r = func(ncell, self.p, pd.pd)
        else:
            r = make_basic_ring_recipe(ncell, self.p)
        return Recipe(r.release())

    def make_basic_ring_recipe(
            self,
            cell_gid_type ncell,
            pdist = None):
        return self._make_recipe(ncell, pdist, make_basic_ring_recipe)

    def make_basic_kgraph_recipe(
            self,
            cell_gid_type ncell,
            pdist = None):
        return self._make_recipe(ncell, pdist, make_basic_kgraph_recipe)

    def make_basic_rgraph_recipe(
            self,
            cell_gid_type ncell,
            pdist = None):
        return self._make_recipe(ncell, pdist, make_basic_rgraph_recipe)

cdef class SegmentLocation:
    cdef segment_location l

    def __cinit__(self, segment_location l):
        self.l = l

    @property
    def segment(self):
        return self.l.segment

cdef class CellProbeAddress:
    cdef cell_probe_address cpa

    def __cinit__(self, cell_probe_address cpa):
        self.cpa = cpa

    @property
    def kind(self):
        return cpa.kind

    @property
    def location(self):
        return SegmentLocation(cpa.location)

cdef class ProbeInfo:
    cdef probe_info p

    def __init__(self, probe_info p):
        self.p = p

    @property
    def id(self):
        return CellMemberType(self.id)

    @property
    def tag(self):
        return self.p.tag
    
    @property
    def address(self):
        cdef any* addr = &self.p.address
        cdef cell_probe_address* cpa \
            = any_cast[cell_probe_address](addr)

        if cpa: return CellProbeAddress(*cpa)
        return None

class CellMemberType:
    cdef cell_member_type cmt

    def __cinit__(self, cell_member_type cmt):
        self.cmt = cmt

    @property
    def gid(self):
        return self.cmt.gid

    @property
    def index(self):
        return self.cmt.index
        
cdef class Recipe:
    cdef recipe* r = NULL

    def __cinit__(self, recipe* r):
        self.r = r

    def __dealloc__(self):
        if self.r:
            del self.r
            self.r = NULL

    def num_cells(self):
        return self.r.num_cells()

    def get_cell_description(self):
        cdef unique_any* a = &self.r.get_cell_description()
        cdef cell* c = any_cast[cell](a)

        if c: return Cell(c)
        return None

    def get_probe(self, CellMemberType cmt):        
        cdef probe_info p = self.r.get_probe(cmt.cmt)
        return ProbeInfo(p)

cdef class Exporter:
    cdef spike_export_function exporter(self):
        raise NotImplemented()
    
cdef class FileExporter(Exporter):
    cdef string file_name
    cdef string output_path
    cdef string file_extension
    cdef cbool over_write

    def __cinit__(self,
                  str file_name,
                  str output_path,
                  str file_extension,
                  bool over_write):
        self.file_name = <string> file_name
        self.output_path = <string> output_path
        self.file_extension = <string> file_extension
        self.over_write = <cbool> over_write

    cdef spike_export_function exporter(self):
        return file_exporter(
            self.file_name,
            self.output_path,
            self.file_extension,
            self.over_write
        )

cdef class GroupDescription:
    cdef group_description g

    def __cinit__(self, group_description g):
        self.g = g

    @property
    def kind(self):
        return self.g.kind

    @property
    def gids(self):
        cdef cell_gid_type gid
        for gid in self.g.gids:
            yield gid

cdef class Decomp:
    cdef domain_decomposition d

    def __cinit__(self, Recipe r, NodeInfo nd):
        self.d = partition_load_balance(r.recipe[0], nd.nd)

    @property
    def groups(self):
        cdef group_description g
        for g in self.d.groups:
            yield GroupDescription(g)

cdef class Schedule:
    cdef schedule s

    def __cinit__(self, schedule s):
        self.s = s

    @staticmethod
    def regular_schedule(self, time_type sample_dt):
        return Schedule(regular_schedule(sample_dt))


cdef class Probe:
    cdef cell_member_predicate cmp

    def __cinit__(self, cell_member_predicate cmp):
        self.cmp = cmp
    
    @staticmethod
    def one_probe(CellMemberType cmt):
        return Probe(one_probe(cmt.cmt))

cdef class SampleTrace:
    cdef sample_trace st

    def __cinit__(self, CellMemberType probe_id, str name, str units):
        self.st.probe_id = probe_id
        self.st.name = name
        self.st.units = units

    @property
    def probe_id(self):
        return CellMemberType(st.probe_id)

    def make_simple_sampler(self):
        cdef simple_sampler s = make_simple_sampler(self.st.samples)
        return SimpleSampler(s)

cdef class SimpleSampler:
    cdef simple_sampler s

    def __cinit__(self, simple_sampler s):
        self.s = s

cdef class Model:
    cdef model* m = NULL

    def __cinit__(self, Recipe r, Decomp d):
        self.m = new model(r.r, d.d)

    def __dealloc__(self):
        if self.m:
            del self.m
            self.m = NULL

    def reset(self):
        self.m.reset()

    def run(self, time_type tfinal, time_type dt):
        return self.m.run(tfinal, dt)

    def add_sampler(self, Probe p, Schedule s, SimpleSampler ss):
        return self.m.add_sampler(p.cmp, s.s, ss.s)

    def set_binning_policy(self, BinningKind bk, time_type dt):
        self.m.set_binning_policy(bk, dt)

    def set_global_spike_callback(self, Exporter e):
        m.set_global_spike_callback(e.exporter())

    def set_local_spike_callback(self, Exporter e):
        m.set_local_spike_callback(e.exporter())

cdef class MeterReport:
    cdef meter_report mr

    def __cinit__(self, meter_report mr):
        self.mr = mr

cdef class Util:
    @staticmethod
    def profiler_output(time_type dt, bool profile_only_zero):
        profiler_output(dt, profile_only_zero)

    @staticmethod
    def write_trace_csv(SampleTrace st):
        write_trace_csv(st.st)

    @staticmethod
    def write_trace_json(SampleTrace st):
        write_trace_json(st.st)

    @staticmethod
    def make_meter_report(MeterManager mm):
        cdef meter_report mr = make_meter_report(mm.mm)
        return MeterReport(mr)

    @staticmethod
    def to_json(MeterReport mr):
        json r = to_json(mr.mr)
        return <str> r.dump()

    @staticmethod
    def to_string(MeterReport mr):
        cdef string s = to_string(mr)
        return <str> s

