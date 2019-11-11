#pragma once

#include "PlatformHandler.h"

#if PLATFORM_WINDOWS
#include "PlatformWindows.h"

WindowsPlatformHandler* getCorrectPlatformHandle(PlatformHandler* handle)
{
	return new WindowsPlatformHandler;
}

#elif PLATFORM_ANDROID

#endif
