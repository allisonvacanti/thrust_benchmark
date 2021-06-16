#include <nvbench/nvbench.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  state.add_element_count(elements);

  thrust::counting_iterator<T> first(T(0));
  thrust::counting_iterator<T> last = first + elements;

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               const auto policy = thrust::device.on(launch.get_stream());
               thrust::adjacent_difference(policy,
                                           first,
                                           last,
                                           thrust::make_discard_iterator());
             });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::adjacent_difference (basic)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));
