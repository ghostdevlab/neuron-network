//
// Created by Artur Duch on 11/07/2024.
//

#include <SDL.h>
#include "NeuralNetwork.h"
#include "Window.h"

typedef struct TestData {
    float x, y;
    int type;
};

int testFunc(float x, float y) {
    if (x>-0.5 && y>-0.5 && x<0.5 && y < 0.5 ) return 4;

    int aboveF1 = y > x;
    int aboveF2 = y > -x;

    return aboveF2 * 2 + aboveF1;
}

std::vector<TestData> generateDataSet(int minX, int minY, int maxX, int maxY, int count) {
    std::vector<TestData> out;

    int w = (maxX - minX);
    int h = (maxY - minY);

    int hw = w / 2;
    int hh = h / 2;

    for (int i = 0; i < count; i++) {
        int x = (rand() % w) - (w / 2);
        int y = (rand() % h) - (h / 2);

        float sx = (float)x / hw;
        float sy = (float)y / hh;

        int result = testFunc(sx, sy);

        out.push_back(
                {
                        sx, sy,
                        result
                }
        );
    }

    return out;
}

int pickColor(int d);

int main() {
    LayerConfig inputLayer = { 2, SIGMOID };
    std::vector<LayerConfig> hiddenLayer = {
            { 8, SIGMOID},
            { 8, SIGMOID},
    };
    LayerConfig outputLayer = { 5, SIGMOID };
    NeuralNetwork network(inputLayer, hiddenLayer, outputLayer);

    int resX = 1024;
    int resY = 768;

    SDL_Init(SDL_INIT_VIDEO);
    Window window("Neuron training", resX, resY);
    bool running = true;

    std::vector<TestData> data = generateDataSet(-resX/2, -resY/2, resX/2, resY/2, 1000);

    window.clear(0);
    window.lock();

    for(int i=0; i<data.size(); i++) {
        TestData& d = data[i];
        int color = pickColor(d.type);

        window.putPixel(
                resX * (0.5 * d.x + 0.5f),
                resY * (0.5 * d.y + 0.5f), 2,
                color);
    }

    float step = 0.0001f;
    float input[2];
    // Define individual arrays
    static float expected1[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    static float expected2[] = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
    static float expected3[] = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
    static float expected4[] = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    static float expected5[] = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

    // Create an array of pointers to those arrays
    float* expected[] = { expected1, expected2, expected3, expected4, expected5 };


    window.unlock();
    window.updateWindow();

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {

            switch (ev.type) {
                case SDL_QUIT:
                    running = false;
                    break;
            }


        }

        window.clear(0);
        window.lock();

        for(int j=0; j<50; j++) {
            for (int i = 0; i < data.size(); i++) {
                TestData& d = data[i];
                input[0] = d.x;
                input[1] = d.y;

                network.training(
                        input,
                        expected[d.type],
                        step);
            }
        }

        float out[5];
        for(int y = 0; y<window.getHeight(); y++) {
            for (int x = 0; x<window.getWidth(); x++) {
                input[0] = 2.f * ((float)x/resX) - 1.0f;
                input[1] = 2.f * ((float)y/resY) - 1.0f;
                network.calculate(input, out);

                int maxIndex = 0;
                for (int i = 1; i<5; i++) {
                    if (out[maxIndex] < out[i]) maxIndex = i;
                }

                if (out[maxIndex] > 0.2) {
                    int color = pickColor(maxIndex);
                    window.putPixel(x, y, color);
                }

//            float sx = 2.0f * ((float)x / resX) - 1.0f;
//            float sy = 2.0f * ((float)y / resY) - 1.0f;

//            window.putPixel(x, y, testFunc(sx, sy) == 4 ? 0xFFFF : 0);
            }
        }
        window.unlock();
        window.updateWindow();

    }

    SDL_Quit();

    return 0;
}

int pickColor(int d) {
    switch (d) {
        case 0: return (31 << 11);
        case 1: return (63 << 5);
        case 2: return (31);
        case 3: return (31<<11) + (31);
        case 4: return (31<<11) + (63 << 5);
        default:
            return 0;
    }
}