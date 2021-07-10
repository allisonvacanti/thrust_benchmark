#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cub/device/device_merge_sort.cuh>

class less_comparator
{
public:
  template <typename T>
  __device__ bool operator()(T i, T j) noexcept
  {
    return i < j;
  }
};

template <typename T>
void custom_less(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);

  thrust::sequence(data.begin(), data.end());

  state.add_element_count(elements);

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortKeys(nullptr,
                                 temp_size,
                                 thrust::raw_pointer_cast(data.data()),
                                 elements,
                                 less_comparator());

  thrust::device_vector<char> tmp(temp_size);
  thrust::default_random_engine rng;

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &launch, auto &timer) {
               thrust::shuffle(data.begin(), data.end(), rng);
               timer.start();
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
  .set_name("cub::DeviceMergeSort::SortKeys<custom_less> (in-place)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 29, 3));
