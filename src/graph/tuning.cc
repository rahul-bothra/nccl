/*************************************************************************
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);

static int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

// Latencies in us, Bandwidths in GB/s
static const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { { 0 }, { 8.4 } };

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
// Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
static const float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree*/ { 28 }, /* Ring*/ { 3.4 } },
  /* PCI */
  { /* Tree*/ { 28 }, /* Ring*/ { 5.7 } },
  /* NET */
  { /* Tree*/ { 28 }, /* Ring*/ { 9.6 } }
};

static const double perChMaxTreeBws[2][3] = { /* Volta (N1/N2/N4) */ {26.5, 18.5, 10.0}, /* Ampere (N1/N2/N4) */ {24.0, 22.5, 16.0} };

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph) {
  int simpleDefaultThreads = (ringGraph->speedIntra*ringGraph->nChannels <= PCI_WIDTH) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] =
    getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2*WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);

  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  if (nRanks <= 1) return ncclSuccess;

  int compCap80 = minCompCap == 80 && maxCompCap == 80 ? 1 : 0;
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = nNodes <= 2 ? nNodes-1 : 2;
  double perChMaxTreeBw = perChMaxTreeBws[compCap80][index2];

  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph };
  int intraHw[NCCL_NUM_ALGORITHMS], hw[NCCL_NUM_ALGORITHMS];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) intraHw[a] = graphs[a]->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) hw[a] = nNodes == 1 ? intraHw[a] : NCCL_HW_NET;

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
      nRanks;
    int nInterSteps = coll == ncclFuncAllReduce ? 2*(nNodes-1) :
      coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nNodes-1 :
      nNodes;

    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      if (coll != ncclFuncAllReduce && a != NCCL_ALGO_RING) continue;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float speed = nNodes <= 2 ? graphs[a]->speedIntra : graphs[a]->speedInter;
        float busBw = graphs[a]->nChannels * speed;

        // Various model refinements
        if (compCap80) busBw = std::min(busBw, 235.0f);
        if (a == NCCL_ALGO_TREE) busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);

        // Convert bus BW to algorithm BW
        float ratio = (a != NCCL_ALGO_RING) ? .5 : (1.0 * nRanks) / nsteps;
        comm->bandwidths[coll][a][p] = busBw * ratio;

        comm->latencies[coll][a][p] = baseLat[a][p];
        float intraLat = hwLat[intraHw[a]][a][p];
        float interLat = hwLat[NCCL_HW_NET][a][p];
        if (a == NCCL_ALGO_RING) {
          float lat = hwLat[hw[a]][a][p];
          if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
            if (ringGraph->sameChannels) {
              comm->latencies[coll][a][p] += lat;
            } else {
              if (p == NCCL_PROTO_SIMPLE) lat = hwLat[hw[a]][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
              comm->latencies[coll][a][p] += nsteps*lat;
            }
          } else {
            comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
          }
        } else if (a == NCCL_ALGO_TREE) {
          comm->latencies[coll][a][p] +=
            2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
        } else {
          comm->latencies[coll][a][p] +=
            2 * (nRanks/nNodes-1) * intraLat + interLat;
        }
      }
    }
  }

  // Protocols/Algorithms enable/disable, and user overrides.

  int protoEnable[NCCL_NUM_PROTOCOLS] = { 1 };
  int algoEnable[NCCL_NUM_ALGORITHMS] = { 1, 1 };

  const char *protoStr = getenv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoEnable));
  }
  const char *algoStr = getenv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoEnable));
  }

  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = protoEnable[p];
    if (pEnable == 0) comm->bandwidths[c][a][p] = 0;
    // Only disable algo for Allreduce since others only have one
    if (c == ncclFuncAllReduce && algoEnable[a] == 0) comm->bandwidths[c][a][p] = 0;
  }

  if (comm->rank == 0) {
    char line[1024];
    sprintf(line, "Latency/AlgBw |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %7s/%6s |", ncclAlgoStr[a], ncclProtoStr[p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    sprintf(line, " Max NThreads |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %14d |", comm->maxThreads[a][p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
      sprintf(line, "%13s |", ncclFuncStr[c]);
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), "%8.1f/%6.1f |", comm->latencies[c][a][p], comm->bandwidths[c][a][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
    }
  }

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  }

  // Override defaults with user env
  char* str = getenv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {{ -2 }, { -2 }};
    sscanf(str, "%ld %ld", t[0], t[1]);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->threadThresholds[a][p] = t[a][p];
      }
    }
  }

  INFO(NCCL_INIT, "threadThresholds %ld | %ld",
      comm->threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE],
      comm->threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  return ncclSuccess;
}

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo* info, int algorithm, int protocol, float* time) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(info->nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && info->comm->nNodes > 1
      && info->coll == ncclFuncAllReduce && info->nBytes >= info->comm->nRanks/16.0*65536) lat *= 1.9; // Plateau effect of ring
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}
