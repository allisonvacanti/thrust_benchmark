#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

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

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  thrust::sequence(input.begin(), input.end());

  thrust::default_random_engine rng;
  thrust::shuffle(input.begin(), input.end(), rng);

  state.add_element_count(elements);

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortKeysCopy(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    elements,
    less_comparator());

  thrust::device_vector<char> tmp(temp_size);

  state.exec([&](nvbench::launch &launch) {
    NVBENCH_CUDA_CALL(cub::DeviceMergeSort::SortKeysCopy(
      thrust::raw_pointer_cast(tmp.data()),
      temp_size,
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements,
      less_comparator(),
      launch.get_stream()));
  });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(custom_less, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceMergeSort::SortKeysCopy<custom_less> (random)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 29, 2));
