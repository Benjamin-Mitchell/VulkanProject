#include "PlatformWindows.h"
#include <iostream>

//TODO: implement
void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	//auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
	//app->frameBufferResized = true;
}

void WindowsWindow::init()
{
	width = 800;
	height = 600;

	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(width, height, "VulkanEngine", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void WindowsWindow::destroy()
{
	glfwDestroyWindow(window);
}

void WindowsWindow::getActualSize(int &width, int &height)
{
	glfwGetFramebufferSize(window, &width, &height);
}

void WindowsWindow::waitForSafeToRecreate()
{
	int width = 0, height = 0;
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}
}

//////////////////////////////////////////////////////////////////////////////////////////

void WindowsPlatformHandler::init()
{
	windowsWindow->init();
}

const char** WindowsPlatformHandler::getExtensions(uint32_t& extensionCount)
{
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	return glfwExtensions;
}

void WindowsPlatformHandler::createSurface(VkInstance instance, VkSurfaceKHR& surface)
{
	if (glfwCreateWindowSurface(instance, windowsWindow->window, nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("Could not create Window Surface with GLFW!");
	}
}

bool WindowsPlatformHandler::safeToUpdate()
{
	bool safe = !glfwWindowShouldClose(windowsWindow->window);
	if (safe)
		glfwPollEvents();
	return safe;
}

void WindowsPlatformHandler::cleanup()
{
	PlatformHandler::cleanup();
	glfwTerminate();
}