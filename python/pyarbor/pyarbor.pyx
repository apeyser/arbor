from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr, make_shared
#from cpython cimport bool as cbool
from libc.stdint cimport uint32_t
from cpython cimport bool
from cython.operator cimport dereference as deref

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
        if self.cargv:
            free(self.cargv)
            self.cargv = NULL

# cython: c_string_encoding=utf8, c_string_type=unicode
cdef unicode ustring(string s):
    return s.encode('utf8')

######## C++ objects ###############################
#
cdef extern from "<common_types.hpp>" namespace "arb":
    ctypedef uint32_t cell_gid_type
    ctypedef uint32_t cell_size_type
    ctypedef uint32_t cell_lid_type
    ctypedef uint32_t cell_local_size_type
    ctypedef int probe_tag
    ctypedef float time_type

    cdef enum CellKind "arb::cell_kind":
        cable1d_neuron
        regular_spike_source
        data_spike_source

    cdef cppclass cell_member_type:
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

    cdef cppclass measurement:
        string name
        string units
        vector[vector[double]] measurements

    cdef cppclass meter_report:
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
    cdef cppclass nothing_t:
        pass
    cdef nothing_t nothing

cdef extern from "<util/unique_any.hpp>" namespace "arb::util":
    cdef cppclass unique_any:
        bint has_value()
    T unique_any_cast "arb::util::any_cast" [T](unique_any) except+

cdef extern from "<util/any.hpp>" namespace "arb::util":
    cdef cppclass any:
        bint has_value()
    cdef T any_cast[T](any) except+
    
cdef extern from "<recipe.hpp>" namespace "arb":
    cdef cppclass probe_info:
        cell_member_type id
        probe_tag tag
        any address

    cdef cppclass recipe:
        cell_size_type num_cells() except+
        CellKind get_cell_kind(cell_gid_type) except+
        unique_any get_cell_description(cell_gid_type) except+
        probe_info get_probe(cell_member_type) except+

cdef extern from "<segment.hpp>" namespace "arb":
    cdef cppclass segment_location:
        cell_lid_type segment
        double position

cdef extern from "<cell.hpp>" namespace "arb":
    cdef cppclass cell:
        cell_local_size_type num_compartments() except+
    cdef enum ProbeKind  "arb::cell_probe_address::probe_kind":
        membrane_voltage "arb::cell_probe_address::membrane_voltage"
        membrane_current "arb::cell_probe_address::membrane_current"
    cdef cppclass cell_probe_address:
        segment_location location
        ProbeKind kind

cdef extern from "morphology_pool.hpp" namespace "arb":
    cdef cppclass morphology_pool:
        size_t size() except+
        void clear() except+
    cdef morphology_pool default_morphology_pool
    void load_swc_morphology_glob(morphology_pool, string) except+

cdef extern from "miniapp_recipes.hpp" namespace "arb":
    cdef cppclass probe_distribution:
        float proportion
        bint all_segments
        bint membrane_voltage
        bint membrane_current

    cdef cppclass opt_string:
        opt_string()
        opt_string(nothing_t)
        opt_string(string)
        opt_string operator=(string)
        bint operator bool();
        string get() except+
        void reset() except+
        
    cdef cppclass basic_recipe_param:
        unsigned num_compartments
        unsigned num_synapses 
        string synapse_type
        float min_connection_delay_ms
        float mean_connection_delay_ms
        float syn_weight_per_cell
        morphology_pool morphologies
        bint morphology_round_robin
        opt_string input_spike_path

    shared_ptr[recipe] make_basic_ring_recipe(
        cell_gid_type,
        basic_recipe_param,
        probe_distribution) except+

    shared_ptr[recipe] make_basic_kgraph_recipe(
        cell_gid_type,
        basic_recipe_param,
        probe_distribution) except +

    shared_ptr[recipe] make_basic_rgraph_recipe(
        cell_gid_type,
        basic_recipe_param,
        probe_distribution) except+

    shared_ptr[cell_probe_address] to_cell_probe_address(any) except+
    size_t get_num_compartments(shared_ptr[recipe], cell_gid_type) except+

cdef extern from "<threading/threading.hpp>" namespace "arb::threading":
    string thr_description "arb::threading::description" \
        () except+

cdef extern from "miniapp_recipes.hpp" namespace "arb":
    cdef cppclass spike_export_function:
        pass
    spike_export_function file_exporter(string, string, string, bint)

cdef extern from "<domain_decomposition.hpp>" namespace "arb":
    cdef cppclass group_description:
        CellKind kind
        vector[cell_gid_type] gids
        
    cdef cppclass domain_decomposition:
        vector[group_description] groups

cdef extern from "<load_balance.hpp>" namespace "arb":
    domain_decomposition partition_load_balance(
        recipe, node_info
    ) except+

cdef extern from "<sampling.hpp>" namespace "arb":
    cdef cppclass sampler_function:
        pass #void (cell_member_type, probe_tag, std::size_t, const sample_record*)

cdef extern from "trace.hpp":
    cdef cppclass trace_data:
        pass
    cdef cppclass sample_trace:
        cell_member_type probe_id
        string name
        string units
        trace_data samples
        
    void write_trace_csv(sample_trace, string)
    void write_trace_json(sample_trace, string)
    string to_string(meter_report)

    cdef cppclass simple_sampler(sampler_function):
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
        sampler_association_handle add_sampler(
            cell_member_predicate,
            schedule,
            sampler_function) except+
        void set_binning_policy(BinningKind, time_type) except+
        void set_global_spike_callback(spike_export_function) except+
        void set_local_spike_callback(spike_export_function) except+
        size_t num_spikes()

cdef extern from "<event_binner.hpp>" namespace "arb":
    cdef enum BinningKind "arb::binning_kind":
        none      "arb::binning_kind::none"
        regular   "arb::binning_kind::regular"
        following "arb::binning_kind::following"

cdef extern from "<profiling/profiler.hpp>" namespace "arb::util":
    void profiler_output(double, bint)

######### PyObjects #################
cdef class MeterManager:
    cdef meter_manager obj

    def start(self):
        self.obj.start()
    def checkpoint(self, str name):
        self.obj.checkpoint(name)

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
        cdef str r = ustring(thr_description())
        return r

cdef class GlobalPolicyGuard:
    cdef shared_ptr[global_policy_guard] ptr
    
    def __cinit__(self, list argv):
        cdef ArgvList argv_list = ArgvList(argv)
        self.ptr = make_shared[global_policy_guard](argv_list.length,
                                                    argv_list.cargv)

cdef class NodeInfo:
    cdef node_info obj

    def __cinit__(self, num_cpu_cores=None, num_gpus=None):
        cdef unsigned c
        cdef unsigned g
        
        if num_cpu_cores == None and num_gpus == None:
            self.obj = node_info()
        else:
            c = <unsigned> num_cpu_cores
            g = 1 if <bool> num_gpus else 0
            self.obj = node_info(c, g)

    @property
    def num_cpu_cores(self):
        return self.obj.num_cpu_cores
    @num_cpu_cores.setter
    def num_cpu_cores(self, int value):
        self.obj.num_cpu_cores = <unsigned> value

    @property
    def num_gpus(self):
        return self.obj.num_gpus
    @num_gpus.setter
    def num_gpus(self, bool value):
        self.obj.num_gpus = 1 if value else 0

cdef class ProbeDistribution:
    cdef probe_distribution obj

    @property
    def proportion(self):
        return self.obj.proportion
    @proportion.setter
    def proportion(self, float proportion):
        self.obj.proportion = proportion

    @property
    def all_segments(self):
        return self.obj.all_segments
    @all_segments.setter
    def all_segments(self, bool all_segments):
        self.obj.all_segments = all_segments

    @property
    def membrane_voltage(self):
        return self.obj.membrane_voltage
    @membrane_voltage.setter
    def membrane_voltage(self, bool membrane_voltage):
        self.obj.membrane_voltage = membrane_voltage

    @property
    def membrane_current(self):
        return self.obj.membrane_current
    @membrane_current.setter
    def membrane_current(self, bool membrane_current):
        self.obj.membrane_current = membrane_current

ctypedef shared_ptr[recipe] (*_make_recipe_function) (
    cell_gid_type,
    basic_recipe_param,
    probe_distribution) except+

#forward declaration
cdef class Recipe

cdef class Cell:
    cdef cell_gid_type gid
    cdef Recipe parent

    # First check that kind(gid) == CellKind.cable1d_neuron
    def __cinit__(self, cell_gid_type gid, Recipe parent):
        self.gid = gid
        self.parent = parent

    def num_compartments(self):
        return get_num_compartments(self.parent.ptr, self.gid)

cdef class CellMemberType:
    cdef cell_member_type obj

    @staticmethod
    cdef CellMemberType _new(cell_member_type obj):
      cdef CellMemberType self = CellMemberType()
      self.obj = obj
      return self
  
    @property
    def gid(self):
        return self.obj.gid

    @property
    def index(self):
        return self.obj.index

cdef class ProbeInfo

cdef class Recipe:
    cdef shared_ptr[recipe] ptr

    @staticmethod
    cdef Recipe _new(shared_ptr[recipe] ptr):
        cdef Recipe self = Recipe()
        self.ptr = ptr
        return self

    def num_cells(self):
        return deref(self.ptr).num_cells()

    def get_cell_description(self, int gid):
        cdef cell_gid_type cgid = <cell_gid_type> gid
        cdef CellKind ckind = deref(self.ptr).get_cell_kind(cgid)
        if  ckind == CellKind.cable1d_neuron:
            return Cell(cgid, self)
        return None

    def get_probe(self, CellMemberType cmt):
        cdef probe_info p = deref(self.ptr).get_probe(cmt.obj)
        return ProbeInfo._new(p, self)

cdef class MorphologyPool
cdef class BasicRecipeParam:
    cdef basic_recipe_param obj

    @property
    def morphologies(self):
        return MorphologyPool(self)

    @property
    def morphology_round_robin(self):
        return <bool> self.obj.morphology_round_robin
        
    @morphology_round_robin.setter
    def morphology_round_robin(self, bool value):
        self.obj.morphology_round_robin = <bint> value

    @property
    def num_compartments(self):
        return self.obj.num_compartments
        
    @num_compartments.setter
    def num_compartments(self, int value):
        self.obj.num_compartments = <unsigned> value

    @property
    def num_synapses(self):
        return self.obj.num_synapses
        
    @num_synapses.setter
    def num_synapses(self, int value):
        self.obj.num_synapses = <unsigned> value

    @property
    def synapse_type(self):
        return self.obj.synapse_type
        
    @synapse_type.setter
    def synapse_type(self, str value):
        self.obj.synapse_type = <string> value

    @property
    def input_spike_path(self):
        if self.obj.input_spike_path:
            return ustring(self.obj.input_spike_path.get())
        return None
        
    @input_spike_path.setter
    def input_spike_path(self, str value):
        self.obj.input_spike_path = <string> value

    cdef _make_recipe(self,
                      cell_gid_type ncell,
                      pdist, # ProbeDistribution || None
                      _make_recipe_function func):
        cdef shared_ptr[recipe] r
        cdef ProbeDistribution pd
        cdef probe_distribution pd_default
        
        if pdist:
            pd = <ProbeDistribution> pdist
            r = func(ncell, self.obj, pd.obj)
        else:
            r = func(ncell, self.obj, pd_default)
        return Recipe._new(r)

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

cdef class MorphologyPool:
    cdef BasicRecipeParam parent

    def __cinit__(self, BasicRecipeParam parent):
        self.parent = parent

    def size(self):
        return self.parent.obj.morphologies.size()

    def clear(self):
        self.mp.clear()

    def load_swc_morphology_glob(self, str morphologies):
        load_swc_morphology_glob(self.parent.obj.morphologies,
                                 <string> morphologies)
        
cdef class SegmentLocation:
    cdef shared_ptr[segment_location] ptr

    @staticmethod
    cdef SegmentLocation _new(segment_location obj):
        cdef SegmentLocation self = SegmentLocation()
        self.ptr = make_shared[segment_location](obj.segment,
                                                 obj.position)
        return self

    @property
    def segment(self):
        return deref(self.ptr).segment

cdef class CellProbeAddress:
    cdef shared_ptr[cell_probe_address] obj

    # Guarantee that parent kind(gid) == cable1d_neuron
    def __cinit__(self, ProbeInfo parent):
        self.obj = to_cell_probe_address(parent.obj.address)

    @property
    def kind(self):
        return deref(self.obj).kind

    @property
    def location(self):
        return SegmentLocation._new(deref(self.obj).location)

cdef class ProbeInfo:
    cdef probe_info obj
    cdef Recipe parent

    @staticmethod
    cdef ProbeInfo _new(probe_info obj, Recipe parent):
        cdef ProbeInfo self = ProbeInfo()
        self.obj = obj
        self.parent = parent
        return self

    @property
    def id(self):
        return CellMemberType._new(self.obj.id)

    @property
    def tag(self):
        return self.obj.tag
    
    @property
    def address(self):
        cdef cell_gid_type cgid = self.obj.id.gid
        cdef CellKind ckind \
            = deref(self.parent.ptr).get_cell_kind(cgid)
        if ckind == CellKind.cable1d_neuron:
            return CellProbeAddress(cgid, self)
        return None

cdef class Exporter:
    cdef spike_export_function exporter(self):
        raise NotImplemented()
    
cdef class FileExporter(Exporter):
    cdef string file_name
    cdef string output_path
    cdef string file_extension
    cdef bint over_write

    def __cinit__(self,
                  str file_name,
                  str output_path,
                  str file_extension,
                  bool over_write):
        self.file_name = <string> file_name
        self.output_path = <string> output_path
        self.file_extension = <string> file_extension
        self.over_write = <bint> over_write

    cdef spike_export_function exporter(self):
        return file_exporter(
            self.file_name,
            self.output_path,
            self.file_extension,
            self.over_write
        )

cdef class Decomp
cdef class GroupDescription:
    cdef size_t index
    cdef Decomp parent

    def __cinit__(self, size_t index, Decomp parent):
        self.index = index
        self.parent = parent
        
    @property
    def kind(self):
        return self.parent.groups[self.index].kind

    @property
    def gids(self):
        cdef cell_gid_type gid
        for gid in self.parent.groups[self.index].gids:
            yield gid

cdef class Decomp:
    cdef domain_decomposition obj

    def __cinit__(self, Recipe r, NodeInfo nd):
        self.obj = partition_load_balance(deref(r.ptr), nd.obj)

    @property
    def groups(self):
        cdef group_description g
        for i in range(self.obj.groups.size()):
            yield GroupDescription(i, self)

cdef class Schedule:
    cdef schedule obj

    @staticmethod
    cdef Schedule _new(schedule obj):
        cdef Schedule self = Schedule()
        self.obj = obj
        return self

    @staticmethod
    def regular_schedule(self, time_type sample_dt):
        return Schedule._new(regular_schedule(sample_dt))

cdef class Probe:
    cdef cell_member_predicate obj

    @staticmethod
    cdef Probe _new(cell_member_predicate obj):
        cdef Probe self = Probe()
        self.obj = obj
        return self
    
    @staticmethod
    def one_probe(CellMemberType cmt):
        return Probe._new(one_probe(cmt.obj))

cdef class SimpleSampler:
    cdef simple_sampler obj

    @staticmethod
    cdef SimpleSampler _new(simple_sampler obj):
        cdef SimpleSampler self = SimpleSampler()
        self.obj = obj
        return self

cdef class SampleTrace:
    cdef sample_trace obj

    def __cinit__(self, CellMemberType probe_id, str name, str units):
        self.obj.probe_id = probe_id.obj
        self.obj.name = name
        self.obj.units = units

#     @property
#     def probe_id(self):
#         return CellMemberType._new(self.obj.probe_id)

#     def make_simple_sampler(self):
#         cdef simple_sampler s = make_simple_sampler(self.obj.samples)
#         return SimpleSampler._new(s)

# cdef class Model:
#     cdef shared_ptr[model] ptr

#     def __cinit__(self, Recipe r, Decomp d):
#         self.ptr = make_shared[model](deref(r.ptr), d.obj)

#     def reset(self):
#         deref(self.ptr).reset()

#     def run(self, time_type tfinal, time_type dt):
#         return deref(self.ptr).run(tfinal, dt)

#     def add_sampler(self, Probe p, Schedule s, SimpleSampler ss):
#         return deref(self.ptr).add_sampler(p.obj, s.obj, ss.obj)

#     def set_binning_policy(self, BinningKind bk, time_type dt):
#         deref(self.ptr).set_binning_policy(bk, dt)

#     def set_global_spike_callback(self, Exporter e):
#         deref(self.ptr).set_global_spike_callback(e.exporter())

#     def set_local_spike_callback(self, Exporter e):
#         deref(self.ptr).set_local_spike_callback(e.exporter())

#     def num_spikes(self):
#         return deref(self.ptr).num_spikes()

cdef class MeterReport:
    cdef meter_report obj

    @staticmethod
    cdef MeterReport _new(meter_report obj):
        cdef MeterReport self = MeterReport
        self.obj = obj
        return self

cdef class Util:
    @staticmethod
    def profiler_output(time_type dt, bool profile_only_zero):
        profiler_output(dt, profile_only_zero)

    @staticmethod
    def write_trace_csv(SampleTrace st, str s):
        write_trace_csv(st.obj, s)

    @staticmethod
    def write_trace_json(SampleTrace st, str s):
        write_trace_json(st.obj, s)

    @staticmethod
    def make_meter_report(MeterManager mm):
        cdef meter_report mr = make_meter_report(mm.obj)
        return MeterReport._new(mr)

    @staticmethod
    def to_json(MeterReport mr):
        cdef json r = to_json(mr.obj)
        return ustring(r.dump())

    @staticmethod
    def to_string(MeterReport mr):
        cdef string s = to_string(mr.obj)
        return ustring(s)
