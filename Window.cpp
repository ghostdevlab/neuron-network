//
// Created by Artur Duch on 09/07/2024.
//

#include "Window.h"


Window::Window(const char *name, int width, int height) : w(width), h(height) {
    window = SDL_CreateWindow(
            name,
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            width, height,
            SDL_WINDOW_SHOWN);

    screen = SDL_GetWindowSurface(window);
    pixels = SDL_CreateRGBSurfaceWithFormat(0, width, height, 16, SDL_PIXELFORMAT_RGB565);
}

Window::~Window() {
    SDL_DestroyWindow(window);
}

void Window::clear(Uint32 color) {
    SDL_FillRect(pixels, NULL, color);
}

void Window::lock() {
    SDL_LockSurface(pixels);
}

void Window::unlock() {
    SDL_UnlockSurface(pixels);
}

int Window::getWidth() const { return w; }
int Window::getHeight() const { return h; }

void Window::putPixel(int x, int y, Uint32 color) {
    ((unsigned short*)(((unsigned char*)pixels->pixels) + y * pixels->pitch))[x] = color;
}

void Window::putPixel(int x, int y, int size, Uint32 color) {
    for(int dy = y - size; dy < y + size; dy ++) {
        for (int dx = x - size; dx < x + size; dx++) {
            ((unsigned short *) (((unsigned char *) pixels->pixels) +
                                 (dy) * pixels->pitch))[dx] = color;
        }
    }
}

void Window::updateWindow() {
    SDL_BlitSurface(pixels, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);
}