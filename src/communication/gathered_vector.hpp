#pragma once

#include <cstdint>
#include <numeric>
#include <vector>
#include <utility>

#include <algorithms.hpp>

namespace nest {
namespace mc {

template <typename T>
class gathered_vector {
public:
    using value_type = T;
    using count_type = unsigned;
    using values_iter = typename std::vector<value_type>::const_iterator;

    gathered_vector(std::vector<value_type>&& v, std::vector<count_type>&& p) :
        values_(std::move(v)),
        partition_(std::move(p))
    {
        EXPECTS(std::is_sorted(partition_.begin(), partition_.end()));
        EXPECTS(std::size_t(partition_.back()) == values_.size());
    }

    /// the partition of distribution
    const std::vector<count_type>& partition() const {
        return partition_;
    }

    // iterators to a distribution partition
    std::pair<values_iter, values_iter>
    values_for_partition(std::size_t i) const {
        EXPECTS(i < size(partition_) - 1);
        const auto start = partition_[i];
        const auto end = partition_[i+1];
        return {values_.cbegin() + start, values_.cbegin() + end};
    }

    /// the number of entries in the gathered vector in partiion i
    count_type count(std::size_t i) const {
        return partition_[i+1] - partition_[i];
    }

    /// the values in the gathered vector
    const std::vector<value_type>& values() const {
        return values_;
    }

    /// the size of the gathered vector
    std::size_t size() const {
        return values_.size();
    }

private:
    std::vector<value_type> values_;
    std::vector<count_type> partition_;
};

} // namespace mc
} // namespace nest
