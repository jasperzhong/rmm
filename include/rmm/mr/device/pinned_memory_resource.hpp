/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "device_memory_resource.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief A `host_memory_resource` that uses `cudaMallocHost` to allocate
 * pinned/page-locked host memory.
 *
 * See https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 *---------------------------------------------------------------------------**/
class pinned_memory_resource final : public device_memory_resource {
 public:
  pinned_memory_resource()                               = default;
  ~pinned_memory_resource()                              = default;
  pinned_memory_resource(pinned_memory_resource const &) = default;
  pinned_memory_resource(pinned_memory_resource &&)      = default;
  pinned_memory_resource &operator=(pinned_memory_resource const &) = default;
  pinned_memory_resource &operator=(pinned_memory_resource &&) = default;

  /**
   * @brief Query whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns false
   */
  bool supports_streams() const noexcept override { return false; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return true; }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates pinned memory on the host of size at least `bytes` bytes.
   *
   * @note Stream argument is ignored
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void *do_allocate(std::size_t bytes, cuda_stream_view) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    void *p{nullptr};
    RMM_CUDA_TRY(cudaMallocHost(&p, bytes), rmm::bad_alloc);
    return p;
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   *---------------------------------------------------------------------------**/
  void do_deallocate(void *p, std::size_t, cuda_stream_view) override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(p));
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two `managed_memory_resources` always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const &other) const noexcept override
  {
    return dynamic_cast<pinned_memory_resource const *>(&other) != nullptr;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cuda_stream_view stream) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RMM_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    return std::make_pair(free_size, total_size);
  }
};
}  // namespace mr
}  // namespace rmm
