/*************************************************************************
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "align.h"
#include <sys/mman.h>

template <typename T>
static ncclResult_t ncclCudaHostCalloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaHostAlloc(ptr, nelem*sizeof(T), cudaHostAllocMapped));
  memset(*ptr, 0, nelem*sizeof(T));
  INFO(NCCL_ALLOC, "Cuda Host Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaCalloc(T** ptr, size_t nelem) {
  // Need async stream for P2P pre-connect + CUDA Graph
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaMalloc(ptr, nelem*sizeof(T)));
  CUDACHECK(cudaMemsetAsync(*ptr, 0, nelem*sizeof(T), stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaStreamDestroy(stream));
  INFO(NCCL_ALLOC, "Cuda Alloc Size %ld pointer %p", nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  CUDACHECK(cudaMemcpy(dst, src, nelem*sizeof(T), cudaMemcpyDefault));
  return ncclSuccess;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static ncclResult_t ncclIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(NCCL_ALLOC, "Ib Alloc Size %ld pointer %p", size, *ptr);
  return ncclSuccess;
}

#endif
