//
// Created by Artur Duch on 09/07/2024.
//

#ifndef NEURALNETWORK_WINDOW_H
#define NEURALNETWORK_WINDOW_H

#include <SDL.h>

class Window {
public:
    Window(const char *name, int width, int height);

    void clear(Uint32 color);

    void lock();
    void unlock();

    void updateWindow();

    void putPixel(int x, int y, Uint32 color);
    void putPixel(int x, int y, int size, Uint32 color);

    int getWidth() const;
    int getHeight() const;

    ~Window();
private:
    int w, h;
    SDL_Window *window;
    SDL_Surface *screen;
    SDL_Surface *pixels;
};


#endif //NEURALNETWORK_WINDOW_H
