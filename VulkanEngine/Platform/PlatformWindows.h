#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "PlatformHandler.h"

class WindowsWindow : public Window 
{
public:
	void init();
	void destroy();
	void resize();
	void getActualSize(int& w, int& h);
	void waitForSafeToRecreate();
	GLFWwindow* window;
};

class WindowsPlatformHandler : public PlatformHandler
{
private:
	WindowsWindow* windowsWindow;
public:
	WindowsPlatformHandler() { 
		windowsWindow = new WindowsWindow;
		window = windowsWindow;
	}

	void init();
	const char** getExtensions(uint32_t &extensionCount);
	void createSurface(VkInstance instance, VkSurfaceKHR& surface);
	bool safeToUpdate();
	void cleanup();
};