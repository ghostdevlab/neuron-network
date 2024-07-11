//
// Created by Artur Duch on 09/07/2024.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <vector>

extern const char* LINEAR;
extern const char* SIGMOID;
extern const char* HYPERBOLIC;
extern const char* RELU;

typedef struct {
    float (*activation)(float);
    float (*derivation)(float);
} ActivationFunction;

typedef struct {
    int neuronCount;
    const char *activationFunction;
} LayerConfig;

class NeuralNetwork {
public:
    NeuralNetwork(LayerConfig _inputLayer, std::vector<LayerConfig> _hiddenLayers, LayerConfig _outputCount);

    void calculate(const float* input, float* output);

    void training(const float* input, float* expected, float step);

    void dump();

private:
    int inputCount;
    std::vector<int> hiddenLayers;
    int outputCount;

    std::vector<int> allLayers;

    int weightCount;
    std::unique_ptr<float[]> weight;
    std::unique_ptr<int[]> weightOffset;

    int neuronDataCount;
    std::unique_ptr<float[]> neuronData;
    std::unique_ptr<int[]> neuronDataOffset;

    // pre alloc for training to speed up
    std::unique_ptr<float[]> neuronSum;
    std::unique_ptr<float[]> derivations;
    std::unique_ptr<float[]> errorPropagation;

    std::vector<ActivationFunction*> activationFunctions;
};


// Input Layer - neuron count is equal to input, every neuron is connected to every input:
// (input count + 1) * input count weights
// +1 because we add bias
// offset = 0;

// Hidden layers - It has K neuron and it takes N inputs (input count is equal to previous layer size)
// K * (N + 1) weights
// +1 is because of bias
// offset = inputLayerSize + sumof(prev hidden layers)

// output layer - neuron count is equal to output size, every neuron is connected to every input from last hidden layer
// output size * (last hidden layer size + 1)
// offset = inputLayerSize + sumof(all hidden layers)

// offset array size = hiddenLayerCount + 2

// NeuralNetwork(5, {3, 4}, 2);
// Input layer weights: 5 * (5 + 1) = 30
// hidden[0] = 3 * (5 + 1) = 18
// hidden[1] = 4 * (3 + 1) = 16
// out layer = 2 * (4 + 1) = 10
// total weight count : 74

// neuronData - every neuron calculated value
// first N reserved for input
// next N is an input layer result
// then we have output for every hidden layer
// and finally we have output from output layer (result)
// size : inputLayerSize(input) + inputLayerSize(input layer) + sumOf(hidden layer neuron count) + outputLayerSize
// NeuralNetwork(5, {3, 4}, 2);
// 5 + 5 + 3 + 4 + 2 = 19


#endif //NEURALNETWORK_NEURALNETWORK_H
