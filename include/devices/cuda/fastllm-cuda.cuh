#include "fastllm.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <map>
#include <memory>

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
void showError(cudaError_t result, char const* const message, const char* const file, int const line);

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define CUDA_MAX(a, b) ((a) > (b) ? (a) : (b))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

#ifndef _FASTLLM_UNION_TYPES_DEFINED
#define _FASTLLM_UNION_TYPES_DEFINED
typedef union __align__(16) {
    uint2 in;
    uint8_t out[8];
} union_char8;

typedef union __align__(16) {
    uint32_t in;
    uint8_t out[4];
} union_char4;

typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {
      // Do nothing
    }
} union_half4;

typedef union __align__(16) _union_half_8 {
    uint4 in;
    half out[8];
    half2 out2[4];
    __device__ _union_half_8() {
      // Do nothing
    }
} union_half8;
#endif // _FASTLLM_UNION_TYPES_DEFINED
#elif defined(USE_ROCM)
#include "fastllm-hip.h"
typedef hipblasHandle_t cublasHandle_t;
// showError declaration needed by checkCudaErrors macro
#define CUDA_MAX(a, b) ((a) > (b) ? (a) : (b))
void showError(hipError_t result, char const* const message, const char* const file, int const line);
#else
typedef void* cublasHandle_t;
#endif

std::vector <long long> FastllmCudaGetFreeSizes();
std::vector <long long> FastllmCudaGetTotalSizes();

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

#ifdef  __cplusplus
extern "C" {
#endif

struct CudaInfos {
    int cudaArch;
    bool hasTensorCore;

    CudaInfos ();
};

const size_t ST128_FP16_COUNT = 8;

CudaInfos *getCudaInfos();

void *FastllmCudaPrepareInput(const fastllm::Data &input);
void *FastllmCudaPrepareOutput(fastllm::Data &output);
void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
void FastllmCudaFinishOutput(fastllm::Data &output, void *data);
cublasHandle_t getFastllmCublasHandle();

void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *cudaIndex);
void FastllmCudaPickOutput(uint8_t *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, fastllm::DataType dataType);

void DeviceSync();
void ForceDeviceSync();
void FastllmInitCublas(void);

void *FastllmCudaStreamCreate(bool nonBlocking = true);
void FastllmCudaStreamDestroy(void *stream);
void FastllmCudaStreamSynchronize(void *stream);

void *FastllmCudaEventCreate();
void FastllmCudaEventDestroy(void *event);
void FastllmCudaEventRecord(void *event, void *stream = nullptr);
void FastllmCudaEventSynchronize(void *event);
void FastllmCudaStreamWaitEvent(void *stream, void *event);

void FastllmCudaMallocBigBuffer(size_t size);
void FastllmCudaClearBigBuffer();
void *FastllmCudaMalloc(size_t size);
void FastllmCudaFree(void *ret);
void FastllmCudaSetWeightSlabBytes(size_t bytes);
size_t FastllmCudaGetWeightSlabBytes();
void *FastllmCudaMallocModelWeight(size_t size);
void FastllmCudaMemPoolStats();
void * FastllmCudaDirectMalloc(size_t size);
void FastllmCudaDirectFree(void *ret);
void FastllmCudaMemset0(void *ret, size_t size);

// å€Ÿç”¨ FlashInfer çš„ d_float_workspace ä½œä¸ºä¸´æ—¶ scratchï¼ˆä¾‹å¦‚ INT4 åé‡åŒ–ä¸º FP16 çš„ä¸´æ—¶ç¼“å†²ï¼‰ã€‚
// è¯­ä¹‰ï¼š
//   - å½“å‰ device çš„ workspace æŒ‡é’ˆ + å­—èŠ‚å¤§å°é€šè¿‡å‡ºå‚è¿”å›žï¼›
//   - ä»…åœ¨ä¸¤æ¬¡ attention è°ƒç”¨ä¹‹é—´ä½¿ç”¨æ˜¯å®‰å…¨çš„ï¼Œå› ä¸ºä¸‹ä¸€æ¬¡ attention ä¼šé‡æ–° plan å¹¶è¦†ç›–é‡Œé¢çš„ tmp_v/tmp_sï¼›
//   - è°ƒç”¨æ–¹éœ€è‡ªè¡Œä¿è¯è°ƒç”¨æœ¬èº«æ˜¯ä¸²è¡Œçš„ï¼ˆåŒä¸€ä¸ª streamï¼‰ï¼Œä¸”ä¸è¦åœ¨ attention kernel è¿˜åœ¨è·‘æ—¶ä½¿ç”¨ï¼›
//   - å¦‚æžœ workspace è¿˜æ²¡æœ‰åˆ›å»ºï¼Œä¼šæŒ‰é»˜è®¤å¤§å°ï¼ˆFT_FLOAT_WORKSPACE_SIZE æˆ– 256MBï¼‰æƒ°æ€§åˆ†é…ã€‚
// æ³¨æ„ï¼šè¿”å›žçš„æŒ‡é’ˆåªæ˜¯å€Ÿç”¨ï¼Œä¸éœ€è¦ freeã€‚
void *FastllmCudaGetFlashInferFloatWorkspace(size_t *outSize);

// å€Ÿ/è¿˜ dequant ç”¨çš„ä¸´æ—¶ scratch bufferã€‚
// FastllmBorrowDequantScratch:
//   - needBytes: æœŸæœ›å¤§å°ï¼ˆå­—èŠ‚ï¼‰ï¼›å¦‚æžœä¸º 0ï¼ŒæŒ‰ workspace å¤§å°è¿”å›žã€‚
//   - outBytes:  å®žé™…å¯ç”¨å­—èŠ‚æ•°ï¼ˆ>= 1ï¼Œå¯èƒ½å°äºŽ needBytesï¼Œè¡¨ç¤ºéœ€è¦åˆ†å—ï¼‰ã€‚
//   - outOwn:    true è¡¨ç¤º scratch æ˜¯ç”¨ FastllmCudaMalloc åˆ†é…çš„ï¼Œä½¿ç”¨å®Œå¿…é¡»è°ƒç”¨ Release å½’è¿˜ã€‚
// è¡Œä¸ºï¼šä¼˜å…ˆå€Ÿç”¨ FlashInfer workspaceï¼›workspace ä¸ºç©ºæˆ–å¤§å°ä¸º 0 æ—¶å›žé€€ä¸º FastllmCudaMalloc(needBytes)ã€‚
void *FastllmBorrowDequantScratch(size_t needBytes, size_t *outBytes, bool *outOwn);
// ä¸Ž Borrow é…å¯¹ã€‚ä»…å½“ outOwn==true æ—¶æ‰çœŸæ­£è°ƒç”¨ FastllmCudaFreeã€‚
void FastllmReleaseDequantScratch(void *ptr, bool own);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromPinnedHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromPinnedHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);

void *FastllmCudaHostMalloc(size_t size);
void FastllmCudaHostFree(void *ptr);
bool FastllmCudaHostRegister(void *ptr, size_t size);
void FastllmCudaHostUnregister(void *ptr);

// å°† host ç«¯æ•°æ®æ‹·åˆ° GPU ä¸´æ—¶ç¼“å†²åŒºï¼ŒæŒ‰æ•°æ®ç±»åž‹åŠ åˆ° dstï¼ˆGPUï¼‰ä¸Šï¼Œlen ä¸ºå…ƒç´ ä¸ªæ•°
void FastllmCudaAddHostToDevice(void *dst, void *hostSrc, int len, fastllm::DataType dataType);
void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);

void FastllmCudaMemcpy2DDeviceToDeviceAuto(void * 	dst, size_t 	dpitch, const void * 	src,
    size_t 	spitch, size_t 	width, size_t 	height, int dstDeviceId, int srcDeviceId);
    
void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height);
void FastllmCudaMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                       size_t *	spitchs, size_t *widths, size_t *	heights,
                                       int batch);
void FastllmCudaShiftAppendWindow(uint8_t *cache, const uint8_t *newToken, int channels, int window, int unitSize);
void FastllmCudaRepeat(void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen);
void FastllmCudaPagedCacheCopy(uint8_t *pagedData, int pageIdx, int pageLen, int numHeads, int headDim,
                               fastllm::DataType dstType, uint8_t *inputData, fastllm::DataType srcType,
                               int seqLen, int inputOffset, int copyLen, int pageOffset);
void FastllmCudaPagedCacheCopyBatch(uint8_t *pagedData, int32_t *pageIdxArray, int32_t *pageOffsetArray,
                                    int pageLen, int batch, int numHeads, int headDim,
                                    fastllm::DataType dstType, uint8_t *inputData, fastllm::DataType srcType);

bool FastllmFloatToHalf(void *a, void *b, int len);
bool FastllmHalfToFloat(void *a, void *b, int len);
bool FastllmBF16ToFloat(void *a, void *b, int len);
bool FastllmFloatToBF16(void *a, void *b, int len);
bool FastllmBF16ToHalf(void *a, void *b, int len);
bool FastllmHalfToBF16(void *a, void *b, int len);

void FastllmReduce(uint8_t *output, uint8_t* partOutput, int len, int threadNum, fastllm::DataType dataType);

bool FastllmCudaMLA(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCache, const fastllm::Data &peCache, 
                    fastllm::Data &score, fastllm::Data &output, float softmaxScale);

bool FastllmCudaMLAPaged(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCachePaged, const fastllm::Data &peCachePaged,
                         fastllm::Data &output, float softmaxScale);

bool FastllmCudaEmbedding(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
bool FastllmCudaEmbeddingDirect(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType);
bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaGeglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData);
bool FastllmCudaSigmoidMambaSoftplus(fastllm::Data &sigmoidInputOutput, const fastllm::Data &softplusInput, fastllm::Data &softplusOutput, const fastllm::Data &aLogData, const fastllm::Data &dtBiasData);
bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCopy(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaAdd(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaTransferAttn(fastllm::Data &input);
bool FastllmCudaCumSumLastDim(fastllm::Data &input);
bool FastllmCudaCausalMask(fastllm::Data &input, int base, float maskValue);
bool FastllmCudaMakeDecayMask(fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaApplyChunkDecayByLastLogG(fastllm::Data &input, const fastllm::Data &g);

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmCudaRMSNormPart(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps, int start, int end);
bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase, int startPos,
                                       int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                       float eps);
bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                      int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                      int quantDim, int blockSize, int posStep);
bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode);
bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits, fastllm::Data &tid2eid,
                                         const int *inputIds, int tokens, int topk,
                                         int scoreFuncMode, float routeScale,
                                         fastllm::Data &expertIndex, fastllm::Data &expertScore);
bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                fastllm::Data &hcScale, fastllm::Data &hcBase,
                                int hcMult, int sinkhornIters, float eps, float normEps,
                                fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb);
bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                    int hcMult, fastllm::Data &dotsFloat);
bool FastllmCudaDeepSeekV4StoreWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4BuildWindowKVPrefix(const fastllm::Data &windowKV, int startPos,
                                             int windowSize, int prefixLen, fastllm::Data &output);
bool FastllmCudaDeepSeekV4BuildCompressedKV(const fastllm::Data &kv, const fastllm::Data &score,
                                            const fastllm::Data &ape, int rawTokenBase, int rawLen,
                                            int blockStart, int blockCount, int compressRatio,
                                            int headDim, int wideDim, bool overlap,
                                            fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q, const fastllm::Data &windowKV,
                                                      const fastllm::Data &compressedKV, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, int compressedCount,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedBatch(
                                                      const std::vector<fastllm::Data*> &q,
                                                      const std::vector<fastllm::Data*> &windowKV,
                                                      const std::vector<fastllm::Data*> &compressedKV,
                                                      fastllm::Data &attnSink,
                                                      int windowSize,
                                                      const std::vector<int> &startPositions,
                                                      const std::vector<int> &compressedCounts,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionPrefill(const fastllm::Data &q, const fastllm::Data &kv,
                                                 fastllm::Data &attnSink, int windowSize, int startPos,
                                                 int compressRatio, int ropeDim, float ropeBase,
                                                 int originalSeqLen, float ropeFactor, int betaFast,
                                                 int betaSlow, float softmaxScale, fastllm::Data &output,
                                                 int prefixLen = 0);
bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA, int groups, int oRank, fastllm::Data &output);
bool FastllmCudaDeepSeekV4HcPost(const fastllm::Data &x, const fastllm::Data &residual, const float *post,
                                 const float *comb, int bsz, int seqlen, int hcMult, int dim,
                                 fastllm::Data &output);
bool FastllmCudaDeepSeekV4HcPostCudaMix(const fastllm::Data &x, const fastllm::Data &residual,
                                        const fastllm::Data &post, const fastllm::Data &comb,
                                        int bsz, int seqlen, int hcMult, int dim,
                                        fastllm::Data &output);
// è®¡ç®—æ¯ä¸ª outer è¡Œåœ¨ [start, end) èŒƒå›´å†…çš„ sum(x^2) (FP32)ï¼Œç”¨äºŽå¤šå¡ RMSNorm çš„è·¨å¡å½’çº¦ã€‚
// outer ä¸Žé€šé“çš„ç‰©ç†å¸ƒå±€æ¥è‡ª inputï¼›output sumOut é•¿åº¦ä¸º outerã€‚
// åŒæ—¶å¦‚æžœ copyInput == true ä¸” input != outputBufferï¼Œä¼šæŠŠ input å®Œæ•´å†…å®¹æ‹·åˆ° outputBufferï¼ˆç”¨äºŽåŽç»­ apply é˜¶æ®µå°±åœ°å†™å›žï¼‰ã€‚
bool FastllmCudaRMSNormPartSum2(const fastllm::Data &input, float *sumOut, int start, int end);
// ç»™å®šå¤–éƒ¨å·²ç»èšåˆå¥½çš„ sumInï¼ˆé•¿åº¦ outerï¼ŒFP32ï¼‰ï¼ŒæŒ‰ partChannelsGlobal è®¡ç®— scaleï¼Œå¹¶å¯¹ input[start:end) åš weight * scale å†™åˆ° outputã€‚
// input == output æ—¶ä¸º in-place æ“ä½œï¼›start/end å¯ä»¥æ˜¯ input å±€éƒ¨åæ ‡ï¼Œweight ç‰©ç†ä¸Šæ˜¯ä¸Ž partLocal å¯¹é½çš„å±€éƒ¨æƒé‡ã€‚
bool FastllmCudaRMSNormPartApply(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, const float *sumIn, float eps, int start, int end, int partChannelsGlobal);
bool FastllmCudaRMSNormSiluMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &gateInput, fastllm::Data &output, float eps);
bool FastllmCudaRMSNormNoScale(fastllm::Data &input, float eps);

// UMA-aware allocation for zero-copy on integrated GPUs
void *FastllmCudaMallocUMA(size_t bytes);
void FastllmCudaFreeUMA(void *ptr);
void FastllmCudaCopyToUMA(void *devPtr, const void *src, size_t bytes);
bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaFloatMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                        fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaHalfMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaConv1DPerChannelFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernel, int stride, int pad, fastllm::Data &output);
bool FastllmCudaConv1DPerChannelSiluSingleTokenFloat16(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(fastllm::Data &cache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(const std::vector<fastllm::Data*> &caches, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);

bool FastllmCudaConv2DFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, fastllm::Data &output);

bool FastllmCudaBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool FastllmCudaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int positionStride);
bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaLlamaRotatePosition2DPart(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int part);
bool FastllmCudaRopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim, float ropeTheta, float ropeScale);
bool FastllmCudaLlama3RopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                   float ropeTheta, float factor, float originalMaxPosition,
                                   float lowFreqFactor, float highFreqFactor);
bool FastllmCudaQwen35InterleavedRope(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                      int sectionT, int sectionH, int sectionW,
                                      float ropeTheta, float ropeScale);
bool FastllmCudaQKVRMSNormRope(fastllm::Data &qkv, fastllm::Data &qNormWeight, fastllm::Data &kNormWeight,
                                const fastllm::Data &positionIds,
                                int q_heads, int k_heads, int head_dim,
                                int rotateDim, float eps, float ropeTheta, float ropeScale);
// èžåˆ QKVRMSNormRope + Split KV + AppendPagedCacheBatch
// qkv: [bs, seqlen, total_dim], qOutput: [bs*q_heads, seqlen, head_dim] (permuted)
// K/V ç›´æŽ¥å†™å…¥ paged cache
bool FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qkv, fastllm::Data &qNormWeight, fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    uint8_t *pagedKData, uint8_t *pagedVData,
    int32_t *insertIndexs, int32_t *insertPositions,
    int32_t *lastPageLens,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale,
    int pageLen, fastllm::DataType pagedDataType, int batch,
    int doQKNorm);
bool FastllmCudaRepeatPenalty (fastllm::Data &input, fastllm::Data &penalty, fastllm::Data &penaltyScale);
bool FastllmCudaTopKTopPSampling(float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize);
bool FastllmCudaApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds);

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                          fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch);
bool FastllmCudaSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis);
bool FastllmCudaCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis);
bool FastllmCudaMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs);
bool FastllmCudaSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch);
bool FastllmCudaBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                      int *ns, int *ms, int *ks,
                                      int *i0Strides, int *i1Strides, float alpha, int batch);
bool FastllmCudaBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch);

bool FastllmCudaHalfAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType);
bool FastllmCudaHalfPagedAttention(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &output, int group, float scale, bool inited = false);
bool FastllmCudaHalfPagedAttentionBatch(fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches, fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs, fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale, int attentionType, bool inited = false, bool sync = true);
bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo = false);
bool FastllmCudaHalfMatMulFloat16AddToNoBias(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
void FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaHalfMergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                          fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                          bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                 fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                 const float *scores, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                  fastllm::Data **weights, int weightsBatch,
                                                  const int *routeRows, const float *routeScales,
                                                  const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                  int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                        fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                               fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                               const float *scores, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                fastllm::Data **weights, int weightsBatch,
                                                const int *routeRows, const float *routeScales,
                                                const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                       fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                       bool scoresOnCuda, int topk, int hidden, int inter);

bool FastllmCudaBFloat16MatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                              fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                              bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                      fastllm::Data **weights, int weightsBatch,
                                                      const int *routeRows, const float *routeScales,
                                                      const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                      int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                            fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                            bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                       fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                       const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                    fastllm::Data **weights, int weightsBatch,
                                                    const int *routeRows, const float *routeScales,
                                                    const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                    int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                           fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                           bool scoresOnCuda, int topk, int hidden, int inter);

bool FastllmCudaHalfMatMulFloat16Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lenght, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids);
void FastllmResetLogitsOfEOSAll(int batch, fastllm::Data *logits, const std::vector<int> &eos_ids);

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float qScale = 1.0f);
void FastllmRecurrentGatedDeltaRuleBatch(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out, float qScale = 1.0f);
void FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
void FastllmChunkGatedDeltaRulePrefill(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn, fastllm::Data &k_cumdecay,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out);

void FastllmCudaSetDevice(int gpu_id);
int FastllmCudaGetDevice();
int GetPointerDeviceId(void *ptr);
int FastllmCudaGetDeviceCount();
#ifdef  __cplusplus
}
#endif

#if defined(__CUDACC__) || defined(USE_ROCM)
/* CUDA/HIP kernel declarations (shared by linear/ggml/attention files) */
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);
extern __global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len);
extern __global__ void FastllmCudaBF162FloatKernel(uint16_t* a, float *b, int len);
extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2Bf16Kernel(float* a, __nv_bfloat16* b, int len);
extern __global__ void FastllmCudaBF162HalfKernel(uint16_t* a, half *b, int len);
extern __global__ void FastllmCudaHalf2BF16Kernel(half* a, __nv_bfloat16 *b, int len);
extern __global__ void FastllmCudaBiasKernel(__nv_bfloat16* a, __nv_bfloat16* bias, int k);
#endif
#ifdef USE_CUDA
bool FastllmCudaTanhSoftcap(const fastllm::Data &input, fastllm::Data &output, float cap);
bool FastllmCudaTanhSoftcapInplace(fastllm::Data &data, float cap);
#else
// MSVC fallback: CPU tanh loop (defined in fastllm-cuda.cu but we need inline here)
inline bool FastllmCudaTanhSoftcap(const fastllm::Data &, fastllm::Data &, float) { return false; }
inline bool FastllmCudaTanhSoftcapInplace(fastllm::Data &, float) { return false; }
#endif
#ifdef USE_CUDA
bool FastllmCudaWaveGEMV(const fastllm::Data &weight, const fastllm::Data &input, fastllm::Data &output, float scale);
#else
inline bool FastllmCudaWaveGEMV(const fastllm::Data &, const fastllm::Data &, fastllm::Data &, float) { return false; }
#endif
