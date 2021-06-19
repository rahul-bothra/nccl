/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

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

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
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
    if (nthreadsSplit >= 256) nthreadsSplit += 64;
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
