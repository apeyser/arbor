#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <functional>

#include <algorithms.hpp>
#include <connection.hpp>
#include <communication/gathered_vector.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/double_buffer.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>
#include <util/compare.hpp>

namespace nest {
namespace mc {
namespace communication {

// When the communicator is constructed the number of target groups and targets
// is specified, along with a mapping between local cell id and local
// target id.
//
// The user can add connections to an existing communicator object, where
// each connection is between any global cell and any local target.
//
// Once all connections have been specified, the construct() method can be used
// to build the data structures required for efficient spike communication and
// event generation.
template <typename Time, typename CommunicationPolicy>
class communicator {
public:
    using communication_policy_type = CommunicationPolicy;
    using id_type = cell_gid_type;
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    using connection_type = connection<time_type>;

    /// per-cell group lists of events to be delivered
    using event_queue =
        std::vector<postsynaptic_spike_event<time_type>>;

    using gid_partition_type =
        util::partition_range<std::vector<cell_gid_type>::const_iterator>;

    communicator() {}

    explicit communicator(gid_partition_type cell_gid_partition):
        cell_gid_partition_(cell_gid_partition)
    {}

    cell_local_size_type num_groups_local() const
    {
        return cell_gid_partition_.size();
    }

    void add_connection(connection_type con) {
        EXPECTS(is_local_cell(con.destination().gid));
        connections_.push_back(con);
    }

    /// returns true if the cell with gid is on the domain of the caller
    bool is_local_cell(id_type gid) const {
        return algorithms::in_interval(gid, cell_gid_partition_.bounds());
    }

    /// builds the optimized data structure
    /// must be called after all connections have been added
    void construct() {
        if (!std::is_sorted(connections_.begin(), connections_.end())) {
            threading::sort(connections_);
        }
    }

    /// the minimum delay of all connections in the global network.
    time_type min_delay() {
        auto local_min = std::numeric_limits<time_type>::max();
        for (auto& con : connections_) {
            local_min = std::min(local_min, con.delay());
        }

        return communication_policy_.min(local_min);
    }

    /// Perform exchange of spikes.
    ///
    /// Takes as input the list of local_spikes that were generated on the calling domain.
    /// Returns the full global set of vectors, along with meta data about their partition
    gathered_vector<spike_type> exchange(const std::vector<spike_type>& local_spikes) {
        // global all-to-all to gather a local copy of the global spike list on each node.
        auto global_spikes = communication_policy_.gather_spikes( local_spikes );
        num_spikes_ += global_spikes.size();
        return global_spikes;
    }

    /// Check each global spike in turn to see it generates local events.
    /// If so, make the events and insert them into the appropriate event list.
    /// Return a vector that contains the event queues for each local cell group.
    /// NB: ^^ the actual algorithm is O(log(neurons/node))
    ///
    /// Returns a vector of event queues, with one queue for each local cell group. The
    /// events in each queue are all events that must be delivered to targets in that cell
    /// group as a result of the global spike exchange.    
    std::vector<event_queue> make_event_queues(const gathered_vector<spike_type>& global_spikes)
    {
        // turn pair<it1, it2> into a class with begin()/end()
        using nest::mc::util::make_range;
        using nest::mc::util::lessthan;
        // Comparator operator between a spike and a spike source for equal_range
        struct extractor {
            using id_type = typename spike_type::id_type;
            id_type operator()(const id_type& s) {return s;}
            id_type operator()(const spike_type& s) {return s.source;}
        };

        // queues to return
        auto queues = std::vector<event_queue>(num_groups_local());

        auto con_next = connections_.cbegin();
        const auto con_end = connections_.cend();

        // For block of connections, search for block of spikes from
        // that sender
        while (con_next != con_end) {
            // we grab the next block of connections from the same sender
            const auto src = con_next->source();
            const auto targets = std::equal_range(con_next, con_end, src);
            con_next = targets.second; // next iteration, next conn block
            
            // we grab the block of spikes associated with the connections
            const auto domain = con_next->domain();
            const auto domain_spikes = global_spikes.values_for_partition(domain);
            const auto sources = std::equal_range(domain_spikes.first,
                                                  domain_spikes.second,
                                                  src, lessthan<extractor>());
            if (sources.first == sources.second) {
                continue; // skip if no spikes
            }
            
            // Now we just need to walk over all combinations of matching spikes and connections
            // Do it first by connection because of shared data
            for (auto&& con: make_range(targets)) {
                const auto gidx = cell_group_index(con.destination().gid);
                auto& queue = queues[gidx];

                for (auto&& spike: make_range(sources)) {
                    queue.push_back(con.make_event(spike));
                }
            }
        }

        return queues;
    }

    /// Returns the total number of global spikes over the duration of the simulation
    uint64_t num_spikes() const { return num_spikes_; }

    const std::vector<connection_type>& connections() const {
        return connections_;
    }

    communication_policy_type communication_policy() const {
        return communication_policy_;
    }

    void reset() {
        num_spikes_ = 0;
    }

private:
    std::size_t cell_group_index(cell_gid_type cell_gid) const {
        EXPECTS(is_local_cell(cell_gid));
        return cell_gid_partition_.index(cell_gid);
    }

    std::vector<connection_type> connections_;

    communication_policy_type communication_policy_;

    uint64_t num_spikes_ = 0u;

    gid_partition_type cell_gid_partition_;
};

} // namespace communication
} // namespace mc
} // namespace nest
