#pragma once

#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy>
class global_search_communicator: public base_communicator<Time, CommunicationPolicy> {
public:
    using base = base_communicator<Time, CommunicationPolicy>;
    using spike_type = typename base::spike_type;
    using event_queue = typename base::event_queue;
    using gid_partition_type = typename base::gid_partition_type;
    
    using base::num_groups_local;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    global_search_communicator(): base() {}

    explicit global_search_communicator(gid_partition_type cell_gid_partition):
        base(cell_gid_partition)
    {}
    
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

        // Do a binary search on the globally sorted spike array
        // and the sorted-by-source connection list.
        // (We could shrink these lists to eliminate impossible end points,
        // but that buys us very little for large, randomly distributed networks.)
        auto con_next = connections_.cbegin();
        const auto con_end = connections_.cend();

        const auto& spikes = global_spikes.values();
        auto spikes_next = spikes.cbegin();
        const auto spikes_end = spikes.cend();

        // Search for next block of spikes and connections with the same sender
        while (con_next != con_end && spikes_next != spikes.end()) {
            // we grab the next block of connections from the same sender
            const auto src = con_it->source();
            const auto targets = std::equal_range(con_next, con_end, src);
            con_next = targets.second; // next iteration, next conn block
            
            // and the associated block of spikes
            const auto sources = std::equal_range(spikes_next, spikes_end,
                                                  src, lessthan<extractor>());
            if (sources.first == sources.second) {
                continue; // skip if no spikes
            }
            spikes_next = sources.second; //next block starts after this

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
};

template <typename Time, typename CommunicationPolicy>
using communicator = global_search_communicator<Time, CommunicationPolicy>;


}
}
}
