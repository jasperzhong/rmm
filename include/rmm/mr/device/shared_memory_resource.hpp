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

#include <fcntl.h> /* For O_* constants */
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief A `device_memory_resource` that uses `cudaMallocHost` to allocate
 * pinned/page-locked host memory.
 *
 * See https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 *---------------------------------------------------------------------------**/
class shared_memory_resource final : public device_memory_resource {
 public:
  shared_memory_resource(int local_rank) : local_rank_(local_rank) {}
  ~shared_memory_resource()                              = default;
  shared_memory_resource(shared_memory_resource const &) = default;
  shared_memory_resource(shared_memory_resource &&)      = default;
  shared_memory_resource &operator=(shared_memory_resource const &) = default;
  shared_memory_resource &operator=(shared_memory_resource &&) = default;

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
   * @brief Allocates shared pinned memory on the host of size at least `bytes` bytes.
   *
   * @note Stream argument is ignored
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated or if the shared memory file cannot be created.
   *
   * @param bytes The size of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void *do_allocate(std::size_t bytes, cuda_stream_view) override
  {
    // don't allocate anything if the user requested zero bytes
    if (0 == bytes) { return nullptr; }

    // create shared memory
    int fd = -1;
    if (local_rank_ == 0) {
      fd = shm_open("/shm", O_RDWR | O_CREAT, 0666);
      if (fd == -1) {
        RMM_LOG_ERROR("shm_open failed");
        return nullptr;
      }
      if (ftruncate(fd, bytes) == 0) {
        RMM_LOG_ERROR("ftruncate failed");
        return nullptr;
      }
    } else {
      // need to wait for local rank 0 to create the shared memory
      while (fd == -1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fd = shm_open("/shm", O_RDWR, 0666);
      }
    }
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      RMM_LOG_ERROR("mmap failed");
      return nullptr;
    }

    RMM_CUDA_TRY(cudaHostRegister(p, bytes, cudaHostRegisterPortable), rmm::bad_alloc);
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
  void do_deallocate(void *p, std::size_t bytes, cuda_stream_view) override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaHostUnregister(p));
    munmap(p, bytes);
    if (local_rank_ == 0) { shm_unlink("/shm"); }
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
    return dynamic_cast<shared_memory_resource const *>(&other) != nullptr;
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

  int local_rank_;  // local rank of the process
};
}  // namespace mr
}  // namespace rmm
