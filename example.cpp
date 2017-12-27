#include <iostream>
#include <cstring>
#include <fstream>
#include "cudnn.h"

#define CUDNN_CALL(func)                                                       \
  {                                                                            \
    auto e = (func);                                                           \
    if (e != CUDNN_STATUS_SUCCESS) {                                           \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__;         \
        std::cerr << " : " << cudnnGetErrorString(e);                          \
    }                                                                          \
  }

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    auto e = (func);                                                           \
    if ((func) != cudaSuccess) {                                               \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__;          \
        std::cerr << " : " << cudaGetErrorString(e);                           \
    }                                                                          \
  }



int main() {
    const int kNumTimestamps = 10000;
    const int kNumLabels = 5;

    cudnnHandle_t handle;
    CUDNN_CALL(cudnnCreate(&handle));

    cudnnTensorDescriptor_t probs;
    float* pProbs;
    {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&probs));
        const int dims[] {kNumTimestamps, 1, kNumLabels};
        const int strides[] {kNumLabels, kNumLabels, 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(probs, CUDNN_DATA_FLOAT, 3, dims, strides));
        CUDA_CALL(cudaMallocManaged(&pProbs, sizeof(float) * kNumLabels * kNumTimestamps));
        for (size_t i = 0; i < kNumTimestamps; ++i) {
            for (size_t j = 0; j < kNumLabels; ++j) {
                pProbs[i * kNumLabels + j] = 0.2;
            }
        }
    }


    cudnnTensorDescriptor_t grads;
    float* pGrads;
    {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&grads));
        const int dims[] {kNumTimestamps, 1, kNumLabels};
        const int strides[] {kNumLabels, kNumLabels, 1};
        CUDNN_CALL(cudnnSetTensorNdDescriptor(grads, CUDNN_DATA_FLOAT, 3, dims, strides));
        CUDA_CALL(cudaMalloc(&pGrads, sizeof(float) * kNumLabels * kNumTimestamps));
    }


    cudnnCTCLossDescriptor_t ctcLossDesc;
    CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctcLossDesc));
    CUDNN_CALL(cudnnSetCTCLossDescriptor(ctcLossDesc, CUDNN_DATA_FLOAT));

    size_t workspace_size;

    int labels[] { 0, 1, 2, 3, 4 };
    int labelLengths[] { 5 };
    int inputLengths[] { kNumTimestamps };

    CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(
                handle, probs, grads, 
                labels, labelLengths, inputLengths,
                CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
                ctcLossDesc, &workspace_size));

    void *workspace;
    CUDA_CALL(cudaMalloc(&workspace, workspace_size));

    float *costs;
    CUDA_CALL(cudaMalloc(&costs, sizeof(float) * 1));

    
    CUDNN_CALL(cudnnCTCLoss(handle,
            probs,
            pProbs,
            labels,
            labelLengths,
            inputLengths,
            costs,
            grads,
            pGrads,
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
            ctcLossDesc,
            workspace,
            workspace_size));

    float cost;
    CUDA_CALL(cudaMemcpy(&cost, costs, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << cost << std::endl;

    float *grads_host = (float*)malloc(sizeof(float) * kNumLabels * kNumTimestamps);
    CUDA_CALL(cudaMemcpy(grads_host, pGrads, sizeof(float) * kNumLabels * kNumTimestamps,
                cudaMemcpyDeviceToHost));

    std::ofstream out("grads.txt");
    for (size_t i = 0; i < kNumLabels * kNumTimestamps; ++i) {
        out << grads_host[i] << "\n";
    }

    cudaFree(costs);
    cudaFree(workspace);
    cudaFree(pProbs);
    cudaFree(pGrads);


    CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctcLossDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(probs));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(grads));
    CUDNN_CALL(cudnnDestroy(handle));
}

