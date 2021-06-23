/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <inttypes.h>
#include <chrono>

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"


#define FORMATS_2(identifier) identifier " " identifier " "
#define FORMATS_4(identifier) FORMATS_2(identifier) FORMATS_2(identifier)
#define FORMATS_8(identifier) FORMATS_4(identifier) FORMATS_4(identifier)
#define FORMATS_16(identifier) FORMATS_8(identifier) FORMATS_8(identifier)
#define FORMATS_32(identifier) FORMATS_16(identifier) FORMATS_16(identifier)
#define FORMATS_64(identifier) FORMATS_32(identifier) FORMATS_32(identifier)
#define FORMATS_128(identifier) FORMATS_64(identifier) FORMATS_64(identifier)
#define FORMATS_256(identifier) FORMATS_128(identifier) FORMATS_128(identifier)
#define FORMATS_512(identifier) FORMATS_256(identifier) FORMATS_256(identifier)

#define EXPAND_ARRAY_2(arr, index) arr[index + 1] - arr[index], arr[index + 2] - arr[index + 1]
#define EXPAND_ARRAY_4(arr, index) EXPAND_ARRAY_2(arr, index), EXPAND_ARRAY_2(arr, index + 2) 
#define EXPAND_ARRAY_8(arr, index) EXPAND_ARRAY_4(arr, index), EXPAND_ARRAY_4(arr, index + 4) 
#define EXPAND_ARRAY_16(arr, index) EXPAND_ARRAY_8(arr, index), EXPAND_ARRAY_8(arr, index + 8) 
#define EXPAND_ARRAY_32(arr, index) EXPAND_ARRAY_16(arr, index), EXPAND_ARRAY_16(arr, index + 16) 
#define EXPAND_ARRAY_64(arr, index) EXPAND_ARRAY_32(arr, index), EXPAND_ARRAY_32(arr, index + 32) 
#define EXPAND_ARRAY_128(arr, index) EXPAND_ARRAY_64(arr, index), EXPAND_ARRAY_64(arr, index + 64) 
#define EXPAND_ARRAY_256(arr, index) EXPAND_ARRAY_128(arr, index), EXPAND_ARRAY_128(arr, index + 128) 
#define EXPAND_ARRAY_512(arr, index) EXPAND_ARRAY_256(arr, index), EXPAND_ARRAY_256(arr, index + 256) 

#define FORMATS_N(n, identifier) FORMATS_##n(identifier)
#define EXPAND_ARRAY_N(n, arr) EXPAND_ARRAY_##n(arr, 0)

#define LOG_TIME(variable) variable = clock64(); 
#define LOG_ITER_TIME(variable, loopIter) if(threadIdx.x == 0 && loopIter == 0){variable = clock64();} 


template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    int nIters = 1 + ((size - 1) / (nranks*loopSize));

    clock_t kernelStartTime;
    clock_t kernelEndTime;
    clock_t scatterReduceTimes[10][5];
    clock_t allGatherTimes[10][5];

    LOG_TIME(kernelStartTime);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

    int curr_iter = 0;
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize, curr_iter++) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      LOG_TIME(scatterReduceTimes[curr_iter][0]);

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.send(thisInput+offset, nelem);

      LOG_TIME(scatterReduceTimes[curr_iter][1]);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSend(thisInput+offset, nelem);
        
        LOG_TIME(scatterReduceTimes[curr_iter][j]);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

      LOG_TIME(scatterReduceTimes[curr_iter][nranks]);
      LOG_TIME(allGatherTimes[curr_iter][0]);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        LOG_TIME(allGatherTimes[curr_iter][j]);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
      LOG_TIME(allGatherTimes[curr_iter][nranks - 1]);
      LOG_TIME(allGatherTimes[curr_iter][nranks]);
    }

    LOG_TIME(kernelEndTime);

    if(tid == 0){
      if (nIters == 1){
        printf("RingSimple %d %d %" PRIu64 " %d %" PRId64 \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) "\n\n", \
          bid, nthreads, kernelEndTime - kernelStartTime, nIters, size*sizeof(T), \
          scatterReduceTimes[0][nranks] - scatterReduceTimes[0][0], EXPAND_ARRAY_4(scatterReduceTimes[0], 0), \
          allGatherTimes[0][nranks] - allGatherTimes[0][0], EXPAND_ARRAY_4(allGatherTimes[0], 0));
      }
      else if(nIters == 2){
          printf("RingSimple %d %d %" PRIu64 " %d %" PRId64 \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
          "\n%" PRIu64 " " FORMATS_4("%" PRIu64) "\n\n", \
          bid, nthreads, kernelEndTime - kernelStartTime, nIters, size*sizeof(T), \
          scatterReduceTimes[0][nranks] - scatterReduceTimes[0][0], EXPAND_ARRAY_4(scatterReduceTimes[0], 0), \
          allGatherTimes[0][nranks] - allGatherTimes[0][0], EXPAND_ARRAY_4(allGatherTimes[0], 0), \
          scatterReduceTimes[1][nranks] - scatterReduceTimes[1][0], EXPAND_ARRAY_4(scatterReduceTimes[1], 0), \
          allGatherTimes[1][nranks] - allGatherTimes[1][0], EXPAND_ARRAY_4(allGatherTimes[1], 0) \
          );
      }
      else{
          printf("RingSimple %d %d %" PRIu64 " %d %" PRId64 \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
            "\n%" PRIu64 " " FORMATS_4("%" PRIu64) "\n\n", \
            bid, nthreads, kernelEndTime - kernelStartTime, nIters, size*sizeof(T), \
            scatterReduceTimes[0][nranks] - scatterReduceTimes[0][0], EXPAND_ARRAY_4(scatterReduceTimes[0], 0), \
            allGatherTimes[0][nranks] - allGatherTimes[0][0], EXPAND_ARRAY_4(allGatherTimes[0], 0), \
            scatterReduceTimes[1][nranks] - scatterReduceTimes[1][0], EXPAND_ARRAY_4(scatterReduceTimes[1], 0), \
            allGatherTimes[1][nranks] - allGatherTimes[1][0], EXPAND_ARRAY_4(allGatherTimes[1], 0), \
            scatterReduceTimes[2][nranks] - scatterReduceTimes[2][0], EXPAND_ARRAY_4(scatterReduceTimes[2], 0), \
            allGatherTimes[2][nranks] - allGatherTimes[2][0], EXPAND_ARRAY_4(allGatherTimes[2], 0) \
          );
      }
    }

  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-2*WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if(tid == 0)
      printf("** Tree Simple %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

#if 1
    if (tid < nthreads+WARP_SIZE) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (tid < nthreads+WARP_SIZE) {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.directSend(thisOutput+offset, offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.directRecv(thisOutput+offset, offset, nelem);
        } else {
          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }
      }
    }
#else
    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit == 256) nthreadsSplit += 64;
    if (tree->up == -1) {
      if (tid < nthreads+WARP_SIZE) {
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid, nthreads, tree->down, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
        }
      }
    } else {
      if (tid < nthreadsSplit + WARP_SIZE) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
          prims(tid, nthreadsSplit, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.send(thisInput+offset, nelem);
          } else {
            prims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &tree->up, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 2);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(thisOutput+offset, offset, nelem);
          } else {
            prims.directRecvCopySend(thisOutput+offset, offset, nelem);
          }
        }
      }
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->collTree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if(tid == 0)
      printf("** CollNet Simple %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.recv(thisOutput+offset, nelem);
        } else {
          prims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    int nIters = 1 + ((size - 1) / (nranks*loopSize));
    clock_t kernelStartTime;
    clock_t kernelEndTime;
    clock_t scatterReduceTimes[10][5];
    clock_t allGatherTimes[10][5];

    LOG_TIME(kernelStartTime);

    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    int curr_iter = 0;
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize, curr_iter++) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      LOG_TIME(scatterReduceTimes[curr_iter][0]);
      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(thisInput+offset, nelem);

      LOG_TIME(scatterReduceTimes[curr_iter][1]);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(thisInput+offset, nelem);
        LOG_TIME(scatterReduceTimes[curr_iter][j]);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

      LOG_TIME(scatterReduceTimes[curr_iter][nranks]);
      LOG_TIME(allGatherTimes[curr_iter][0]);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(thisOutput+offset, nelem);
        LOG_TIME(allGatherTimes[curr_iter][j]);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(thisOutput+offset, nelem);
      LOG_TIME(allGatherTimes[curr_iter][nranks - 1]);
      LOG_TIME(allGatherTimes[curr_iter][nranks]);
    }

    LOG_TIME(kernelEndTime);

    if(tid == 0){
      printf("RingLL %d %d %" PRIu64 " %d %" PRId64 \
        "\n%" PRIu64 " " FORMATS_4("%" PRIu64) \
        "\n%" PRIu64 " " FORMATS_4("%" PRIu64) "\n\n", \
        bid, nthreads, kernelEndTime - kernelStartTime, nIters, size*sizeof(T), \
        scatterReduceTimes[0][nranks] - scatterReduceTimes[0][0], EXPAND_ARRAY_4(scatterReduceTimes[0], 0), \
        allGatherTimes[0][nranks] - allGatherTimes[0][0], EXPAND_ARRAY_4(allGatherTimes[0], 0));
    }

  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if(tid == 0)
      printf("** Tree LL %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLLPrimitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } while(0);

    do {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    } while(0);
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->collTree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if(tid == 0)
      printf("** CollNet LL %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

    if(tid == 0)
      printf("** Ring LL128 %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(thisOutput+offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(thisOutput+offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    const ssize_t loopSize = nChannels*chunkSize;
    int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    if(tid == 0)
      printf("** Tree LL128 %d %d %" PRIu64 " \n\n", bid, nthreads, size);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (tree->up == -1) {
      // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
      ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY> LLprims(tid, nthreads, tree->down, tree->down, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
      }
    } else {
      if (tid < nthreadsSplit) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims(tid, nthreadsSplit, tree->down, &tree->up, stepSize, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.send(thisInput+offset, nelem);
          } else {
            LLprims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, stepSize, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.recv(thisOutput+offset, nelem);
          } else {
            LLprims.recvCopySend(thisOutput+offset, nelem);
          }
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclWorkElem* args) { }
};
