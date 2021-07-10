#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

#include <cub/device/device_adjacent_difference.cuh>

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements, T(42));
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  state.exec([&](nvbench::launch &launch) {
    NVBENCH_CUDA_CALL(cub::DeviceAdjacentDifference::SubtractLeftCopy(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      elements,
      static_cast<cudaStream_t>(launch.get_stream())));
  });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceAdjacentDifference::SubtractLeftCopy (basic)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(14, 28, 2));
