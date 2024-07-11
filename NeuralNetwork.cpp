//
// Created by Artur Duch on 09/07/2024.
//

#include "NeuralNetwork.h"

#include <utility>
#include <iostream>
#include <random>
#include <cmath>


//#define DEBUG1

#ifdef DEBUG1
#define LOG(x) { x }
#endif
#ifndef DEBUG1
#define LOG(x) {  }
#endif

const char* LINEAR = "linear";
const char* SIGMOID = "sigmoid";
const char* HYPERBOLIC = "hyperbolic";
const char* RELU = "relu";


ActivationFunction* findActivation(const char* name);

NeuralNetwork::NeuralNetwork(LayerConfig _inputLayer, std::vector<LayerConfig> _hiddenLayers, LayerConfig _outputLayer){
    activationFunctions.reserve(2 + _hiddenLayers.size());
    inputCount = _inputLayer.neuronCount;
    activationFunctions.push_back(findActivation(_inputLayer.activationFunction));
    for(auto & _hiddenLayer : _hiddenLayers) {
        hiddenLayers.push_back(_hiddenLayer.neuronCount);
        activationFunctions.push_back(findActivation(_hiddenLayer.activationFunction));
    }
    outputCount = _outputLayer.neuronCount;
    activationFunctions.push_back(findActivation(_outputLayer.activationFunction));

    allLayers.reserve(hiddenLayers.size() + 2);
    allLayers.push_back(inputCount);
    allLayers.insert(allLayers.end(), hiddenLayers.begin(), hiddenLayers.end());
    allLayers.push_back(outputCount);

    weightCount = 0;
    int prevLayerSize = inputCount;
    for(int allLayer : allLayers) {
        int layerSize = (prevLayerSize + 1) * allLayer;
        LOG(std::cout<<"Layer size: "<<layerSize<<std::endl;)
        weightCount += layerSize;
        prevLayerSize = allLayer;
    }

    LOG(std::cout<<"Weight count "<<weightCount<<std::endl;)

    this->weight = std::make_unique<float[]>(weightCount);
    this->weightOffset = std::make_unique<int[]>(2 + hiddenLayers.size());

    int cols = 12;
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<> d(0, std::sqrt(2.0 / cols)); // Normal distribution


    for(int i=0; i<this->weightCount; i++) {
        weight[i] = d(gen);
//        weight[i] = (2.0f * ((float)(rand()%INT16_MAX) / INT16_MAX) - 1.0f) * 0.01f;
    }

    int offset = 0;
    prevLayerSize = inputCount;
    for(int i = 0; i<allLayers.size(); i++) {
        weightOffset[i] = offset;
        offset += (prevLayerSize + 1) * allLayers[i];
        prevLayerSize = allLayers[i];
    }

    LOG(
        for(int i =0; i<allLayers.size(); i++) {
            std::cout<<"offsets: "<<weightOffset[i]<<std::endl;
        }
    )

    neuronDataCount = inputCount;
    for(int allLayer : allLayers) {
        neuronDataCount += allLayer;
    }
    LOG(std::cout<<"Neuron data count: "<<neuronDataCount<<std::endl;)

    this->neuronSum = std::make_unique<float[]>(neuronDataCount);
    this->neuronData = std::make_unique<float[]>(neuronDataCount);
    this->derivations = std::make_unique<float[]>(neuronDataCount);
    this->neuronDataOffset = std::make_unique<int[]>(3 + hiddenLayers.size());
    this->errorPropagation = std::make_unique<float[]>(neuronDataCount);

    neuronDataOffset[0] = 0;
    offset = inputCount;
    for(int i = 1; i<3 + hiddenLayers.size(); i++) {
        neuronDataOffset[i] = offset;
        offset += allLayers[i - 1];
    }

    LOG(
        for(int i = 0; i<allLayers.size() + 1; i++) {
            std::cout<<"nOffset: "<<neuronDataOffset[i]<<std::endl;
        }
    )

}


float dotProductWithBias(const float* weight, const float* input, int weightCount) {
    float sum = 0.0f;
    for(int i=0; i < weightCount - 1; i++) {
        sum += weight[i] * input[i];
    }
    // bias
    sum += weight[weightCount - 1] * 1.0f;
    return sum;
}


void NeuralNetwork::calculate(const float *input, float *output) {
    // set input
    memcpy(neuronData.get(), input, sizeof(float) * inputCount);

    int currentInputCount = inputCount + 1;
    for (int layerNo = 0; layerNo < allLayers.size(); layerNo++) {
        int outIndex = neuronDataOffset[layerNo + 1];
        int inIndex = neuronDataOffset[layerNo + 0];

        auto activation = activationFunctions[layerNo]->activation;

        int layerWeightOffset = weightOffset[layerNo];

        LOG(std::cout << "data: " << inIndex << " -> " << outIndex << " | " << layerWeightOffset << std::endl;)

        // all neurons in that layer
        for (int neuronNo = 0; neuronNo < allLayers[layerNo]; neuronNo++) {
            int neuronWeightOffset = layerWeightOffset + neuronNo * currentInputCount;
            float sum = dotProductWithBias(
                    weight.get() + neuronWeightOffset,
                    neuronData.get() + inIndex,
                    currentInputCount
            );
            neuronSum[outIndex + neuronNo] = sum;
            neuronData[outIndex + neuronNo] = activation(sum);
            LOG(std::cout << "Neuron(" << neuronNo << ") = " << neuronWeightOffset << " " << inIndex << " "
                      << currentInputCount << std::endl;)
        }

        currentInputCount = allLayers[layerNo] + 1;

    }

    // get output
    if (output != nullptr) {
        memcpy(output, neuronData.get() + neuronDataOffset[2 + hiddenLayers.size()], sizeof(float) * outputCount);
    }
}

void NeuralNetwork::training(const float *input, float *expected, float step) {
    // this will set internals, no need for output buffer
    calculate(input, nullptr);

    // calculate derivation
    // first neuronData are just input,
    // there is no neuron sum for them
    for (int layerNo = 0; layerNo < allLayers.size(); layerNo++) {
        int outIndex = neuronDataOffset[layerNo + 1];
        auto derivation = activationFunctions[layerNo]->derivation;

        for (int neuronNo = 0; neuronNo < allLayers[layerNo]; neuronNo++) {
            derivations[outIndex + neuronNo] = derivation(neuronSum[outIndex + neuronNo]);
        }
    }

    // calculate output layer error
    LOG(std::cout<<"Initial error (";)
    int outputDataIndex = neuronDataOffset[allLayers.size()];
    for(int i = 0; i<outputCount; i++) {
        errorPropagation[outputDataIndex + i] = expected[i] - neuronData[outputDataIndex + i];
        LOG(std::cout<<errorPropagation[outputDataIndex + i]<<"["<<outputDataIndex + i<<"] ";)
    }
    LOG(std::cout<<std::endl;)

    // propagate error to other layers, last index is allLayers.size - 1,
    // however output layer is calculated manually from expected, so we skip it
    for (int layerNo = allLayers.size() - 2; layerNo >= 0 ; layerNo--) {
        int outputDataIndex = neuronDataOffset[layerNo + 1]; // +1 because first row is an input row
        int errorInputIndex = neuronDataOffset[layerNo + 2];
        int neuronCount = allLayers[layerNo];
        int errorCount = allLayers[layerNo + 1];
        int wOffset = weightOffset[layerNo + 1];
        LOG(
                std::cout<<"Propagate size: "<<allLayers[layerNo]<<" range: " <<
            outputDataIndex << " -> " << outputDataIndex + neuronCount - 1<<" | "<<
            wOffset<<" err count: " <<errorCount<<" error input: "<<errorInputIndex<<std::endl;
        )

        int step = neuronCount + 1; // +1 do not use bias weight to propagate, we propagate error only to neurons
        for (int neuronNo = 0; neuronNo < neuronCount; neuronNo++) {
            float errorSum = 0.0f;

            LOG(std::cout<<"Accum errors: ";)

            for(int errNo = 0; errNo < errorCount; errNo++) {
                int wIndex = wOffset + neuronNo + step * errNo;
                errorSum += weight[wIndex] * errorPropagation[errorInputIndex + errNo];
                LOG(std::cout<<wIndex<<" ";)
            }

            errorPropagation[outputDataIndex + neuronNo] = errorSum;

            LOG(std::cout<<" = "<<errorSum<<std::endl;)

        }
    }

//    std::cout<<"error prop ";
//    for(int i = 0; i<neuronDataCount; i++) {
//        std::cout<<errorPropagation[i]<<" ";
//    }
//    std::cout<<std::endl;

    LOG(std::cout<<"TRAINING: "<<std::endl;)
    // train each neuron
    int currentInputCount = inputCount + 1;
    for (int layerNo = 0; layerNo<allLayers.size(); layerNo++) {
        int inputOffset = neuronDataOffset[layerNo];
        int dataOffset = neuronDataOffset[layerNo + 1];
        int wOffset = weightOffset[layerNo];
        int nCount = allLayers[layerNo];
        LOG(std::cout<<layerNo<<": nCount "<<currentInputCount<<" error + derivation offset : "<<
            dataOffset<<" wOffset: "<<wOffset<<
            " input offset : "<<inputOffset<<" neuron count: "<<nCount<<std::endl;)

        for(int neuronNo = 0; neuronNo <nCount; neuronNo++) {
            int neuronWeightOffset = wOffset + neuronNo * currentInputCount;
            LOG(std::cout<<"neuron("<<neuronNo<<") weight offsets "<<neuronWeightOffset<<std::endl;)

            float error = errorPropagation[dataOffset + neuronNo];
            float de = derivations[dataOffset + neuronNo];

            // need to train bias separately - no actual input value
            for (int i = 0; i < currentInputCount - 1; i++) {
                weight[neuronWeightOffset + i] += step * error * de * neuronData[inputOffset + i];
            }

            // bias:
            weight[neuronWeightOffset + currentInputCount - 1] += step * error * de * 1.0f;
        }

        currentInputCount = allLayers[layerNo] + 1;
    }
}

void NeuralNetwork::dump() {
    std::cout<<"W: ";
    for(int i = 0; i<weightCount; i++) {
        std::cout<<weight[i]<<" ";
    }
    std::cout<<std::endl;
}

float linearActivation(float x) { return x; }
float linearDerivation(float x) { return 1; }

float sigmoidActivation(float x) { return  1.0f/(1.0f + exp(-x)); }
float sigmoidDerivation(float x) { return sigmoidActivation(x) * (1.0f - sigmoidActivation(x)); }

float hyperbolicActivation(float x) {
    if (x>7.0) return 0.99999999f;
    if (x<-7.0) return -0.99999999f;
    float a = exp(x);
    float b = exp(-x);
    return (a - b) / (a + b);
}

float hyperbolicDerivation(float x) {
    float gx = hyperbolicActivation(x);
    return 1.0f - gx * gx;
}

float reluActivation(float x) {
    return x < 0 ? 0.01f * x : x;
}

float reluDerivation(float x) {
    return x < 0 ? 0.01 : 1;
}

float invalidFunction(float x) {
    return 0.0f;
}

ActivationFunction functions[] = {
{ linearActivation, linearDerivation },
{ sigmoidActivation, sigmoidDerivation },
{ hyperbolicActivation, hyperbolicDerivation },
{ reluActivation, reluDerivation },
{ invalidFunction, invalidFunction }
};


ActivationFunction* findActivation(const char* name) {
    if (strcmp(name, LINEAR) == 0) return &functions[0];
    if (strcmp(name, SIGMOID) == 0) return &functions[1];
    if (strcmp(name, HYPERBOLIC) == 0) return &functions[2];
    if (strcmp(name, RELU) == 0) return &functions[3];
    return &functions[4];
}