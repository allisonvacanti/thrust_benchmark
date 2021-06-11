#include <nvbench/nvbench.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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

__global__ void rng_setup_kernel(std::size_t elements, curandState *states)
{
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < elements)
    curand_init(1234, i, 0, states + i);
}

template <typename T>
__global__ void rng_generate_kernel(const unsigned int n,
                                    curandState *states,
                                    T *result)
{

  const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n)
  {
    result[i] = static_cast<T>(ceilf(curand_uniform(states + i) * n));
  }
}

template <typename T>
void custom_less(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);
  thrust::device_vector<curandState> rng_states(elements);

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortKeys(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(data.data()),
    elements,
    less_comparator());

  thrust::device_vector<char> tmp(temp_size);

  const unsigned int rng_threads_in_block = 256;
  const unsigned int rng_blocks_in_grid =
    (elements + rng_threads_in_block - 1) / rng_threads_in_block;

  rng_setup_kernel<<<rng_blocks_in_grid, rng_threads_in_block>>>(
    elements,
    thrust::raw_pointer_cast(rng_states.data()));
  cudaDeviceSynchronize();

  state.exec([&](nvbench::launch &launch) {
    rng_generate_kernel<<<rng_blocks_in_grid, rng_threads_in_block>>> (
      elements,
      thrust::raw_pointer_cast(rng_states.data()),
      thrust::raw_pointer_cast(data.data()));

    NVBENCH_CUDA_CALL(cub::DeviceMergeSort::SortKeys(
      thrust::raw_pointer_cast(tmp.data()),
      temp_size,
      thrust::raw_pointer_cast(data.data()),
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
  .set_name("cub::DeviceMergeSort::SortKeys<custom_less> (random)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 29, 2));
