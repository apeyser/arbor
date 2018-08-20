#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <utility>

#include <arbor/execution_context.hpp>

namespace arb {
namespace threading {

// Forward declare task_group at bottom of this header
class task_group;

using std::mutex;
using lock = std::unique_lock<mutex>;
using std::condition_variable;
using task = std::function<void()>;

namespace impl {
class notification_queue {
private:
    // FIFO of pending tasks.
    std::deque<task> q_tasks_;

    // Lock and signal on task availability change this is the crucial bit.
    mutex q_mutex_;
    condition_variable q_tasks_available_;

    // Flag to handle exit from all threads.
    bool quit_ = false;

public:
    // Pops a task from the task queue returns false when queue is empty.
    task try_pop();
    task pop();

    // Pushes a task into the task queue and increases task group counter.
    void push(task&& tsk); // TODO: need to use value?
    bool try_push(task& tsk);

    // Finish popping all waiting tasks on queue then stop trying to pop new tasks
    void quit();
};
}// namespace impl

class task_system {
private:
    unsigned count_;

    std::vector<std::thread> threads_;

    // queue of tasks
    std::vector<impl::notification_queue> q_;

    // threads -> index
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    // total number of tasks pushed in all queues
    std::atomic<unsigned> index_{0};

public:
    task_system();
    // Create nthreads-1 new c std threads
    task_system(int nthreads);

    // task_system is a singleton.
    task_system(const task_system&) = delete;
    task_system& operator=(const task_system&) = delete;

    ~task_system();

    // Pushes tasks into notification queue.
    void async(task tsk);

    // Runs tasks until quit is true.
    void run_tasks_loop(int i);

    // Request that the task_system attempts to find and run a _single_ task.
    // Will return without executing a task if no tasks available.
    void try_run_task();

    // Includes master thread.
    int get_num_threads();

    // Returns the thread_id map
    std::unordered_map<std::thread::id, std::size_t> get_thread_ids();
};

///////////////////////////////////////////////////////////////////////
// types
///////////////////////////////////////////////////////////////////////

template <typename T>
class enumerable_thread_specific {
    std::unordered_map<std::thread::id, std::size_t> thread_ids_;

    using storage_class = std::vector<T>;
    storage_class data;

public:
    using iterator = typename storage_class::iterator;
    using const_iterator = typename storage_class::const_iterator;

    enumerable_thread_specific(const task_system_handle& ts):
        thread_ids_{ts.get()->get_thread_ids()},
        data{std::vector<T>(ts.get()->get_num_threads())}
    {}

    enumerable_thread_specific(const T& init, const task_system_handle& ts):
        thread_ids_{ts.get()->get_thread_ids()},
        data{std::vector<T>(ts.get()->get_num_threads(), init)}
    {}

    T& local() {
        return data[thread_ids_.at(std::this_thread::get_id())];
    }
    const T& local() const {
        return data[thread_ids_.at(std::this_thread::get_id())];
    }

    auto size() const { return data.size(); }

    iterator begin() { return data.begin(); }
    iterator end()   { return data.end(); }

    const_iterator begin() const { return data.begin(); }
    const_iterator end()   const { return data.end(); }

    const_iterator cbegin() const { return data.cbegin(); }
    const_iterator cend()   const { return data.cend(); }
};

inline std::string description() {
    return "CThread Pool";
}

constexpr bool multithreaded() { return true; }

class task_group {
private:
    std::atomic<std::size_t> in_flight_{0};
    /// We use a raw pointer here instead of a shared_ptr to avoid a race condition
    /// on the destruction of a task_system that would lead to a thread trying to join itself
    task_system* task_system_;

public:
    task_group(task_system* ts):
        task_system_{ts}
    {}

    task_group(const task_group&) = delete;
    task_group& operator=(const task_group&) = delete;

    template <typename F>
    class wrap {
        F f;
        std::atomic<std::size_t>& counter;

    public:

        // Construct from a compatible function and atomic counter
        template <typename F2>
        explicit wrap(F2&& other, std::atomic<std::size_t>& c):
                f(std::forward<F2>(other)),
                counter(c)
        {}

        wrap(wrap&& other):
                f(std::move(other.f)),
                counter(other.counter)
        {}

        // std::function is not guaranteed to not copy the contents on move construction
        // But the class is safe because we don't call operator() more than once on the same wrapped task
        wrap(const wrap& other):
                f(other.f),
                counter(other.counter)
        {}

        void operator()() {
            f();
            --counter;
        }
    };

    template <typename F>
    using callable = typename std::decay<F>::type;

    template <typename F>
    wrap<callable<F>> make_wrapped_function(F&& f, std::atomic<std::size_t>& c) {
        return wrap<callable<F>>(std::forward<F>(f), c);
    }

    template<typename F>
    void run(F&& f) {
        ++in_flight_;
        task_system_->async(make_wrapped_function(std::forward<F>(f), in_flight_));
    }

    // wait till all tasks in this group are done
    void wait() {
        while (in_flight_) {
            task_system_->try_run_task();
        }
    }

    // Make sure that all tasks are done before clean up
    ~task_group() {
        wait();
    }
};

///////////////////////////////////////////////////////////////////////
// algorithms
///////////////////////////////////////////////////////////////////////
struct parallel_for {
    template <typename F>
    static void apply(int left, int right, task_system* ts, F f) {
        task_group g(ts);
        for (int i = left; i < right; ++i) {
          g.run([=] {f(i);});
        }
        g.wait();
    }
};
} // namespace threading
} // namespace arb
