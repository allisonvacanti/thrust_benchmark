#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <thrust/detail/temporary_array.h>

#include <cub/device/device_merge_sort.cuh>

/*
 * Arithmetic types with the custom comparator lead to merge sort usage.
 */

class less_comparator
{
public:
  template <typename T>
  __device__ bool operator()(T i, T j) noexcept
  {
    return i < j;
  }
};

template <typename T,typename Derived>
thrust::detail::temporary_array<T, Derived> create_tmp_storage(
  thrust::execution_policy<Derived> &policy, std::size_t size)
{
  return thrust::detail::temporary_array<T, Derived>(policy, size);
}

template <typename T>
void custom_less(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<long long int>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);
  thrust::sequence(data.begin(), data.end());

  thrust::default_random_engine rng;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortKeys(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(data.data()),
    elements,
    less_comparator());

  using namespace nvbench::exec_tag;
  state.exec(timer | sync, // This benchmark needs a timer and syncs internally
             [&](nvbench::launch &launch, auto &timer) {
               auto policy = thrust::device.on(launch.get_stream());
               thrust::shuffle(policy, data.begin(), data.end(), rng);
               timer.start();

               // Shuffle uses temporary storage, so we have to reuse this
               // memory in SortKeys, just like thrust does. Otherwise, we
               // won't fit in the memory for float64.
               auto tmp = create_tmp_storage<thrust::detail::uint8_t>(policy, temp_size);
               NVBENCH_CUDA_CALL(cub::DeviceMergeSort::SortKeys(
                 thrust::raw_pointer_cast(tmp.data()),
                 temp_size,
                 thrust::raw_pointer_cast(data.data()),
                 elements,
                 less_comparator(),
                 launch.get_stream()));

               timer.stop();
             });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(custom_less, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceMergeSort::SortKeys<custom_less> (random)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
