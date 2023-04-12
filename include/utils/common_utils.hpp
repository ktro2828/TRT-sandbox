#ifndef COMMON_UTILS_HPP_
#define COMMON_UTILS_HPP_

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include <string>

inline void loadLibrary(const std::string & path)
{
#ifdef _MSC_VER
  void * handle = loadLibrary(path.c_str());
#else
  int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
  flags |= RTLD_NODELETE;
#endif

  void * handle = dlopen(path.c_str(), flags);
#endif
  if (handle == nullptr) {
#ifdef _MSC_VER
    std::cerr << "Could not load plugin library: " << path << std::endl;
#else
    std::cerr << "Could not load plugin library: " << path << ", due to: " << dlerror()
              << std::endl;
#endif
  }
}
#endif  // COMMON_UTILS_HPP_