#pragma once

#include <util/range.hpp>
#include <util/compare.hpp>

#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy>
class domain_search_communicator: public base_communicator<Time, CommunicationPolicy> {
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
    domain_search_communicator(): base() {}

    explicit domain_search_communicator(gid_partition_type cell_gid_partition):
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
};

template <typename Time, typename CommunicationPolicy>
using communicator = domain_search_communicator<Time, CommunicationPolicy>;

}
}
}
