#pragma once

//TODO: remove this and replace with a make system.
#define PLATFORM_WINDOWS 1
#define PLATFORM_ANDROID 0

#include <vulkan/vulkan.hpp>

class Window
{
public:
	virtual void init() {};
	virtual void destroy() {};
	virtual void resize() {};

	inline int getWidth() { return width; };
	inline int getHeight() { return height; };

	virtual void getActualSize(int& w, int& h) {};
	virtual void waitForSafeToRecreate() {};

	~Window() {};
protected:

	int width, height;
};

class PlatformHandler
{
public:
	PlatformHandler() {}
	Window* window;

	virtual const char** getExtensions(uint32_t& extensionCount) { return NULL; };
	virtual void createSurface(VkInstance instance, VkSurfaceKHR& surface) {};
	virtual bool safeToUpdate() { return false; };
	virtual void init() {};

	virtual void cleanup() {
		window->destroy();
	}

	~PlatformHandler() {};
};