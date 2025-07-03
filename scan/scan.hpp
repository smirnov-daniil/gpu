#ifndef __scan_hpp_
#define __scan_hpp_

#include <cstdint>

enum class device_type : int {
  cpu = 0x1,
  dgpu = 0x2,
  igpu = 0x4,
  gpu = dgpu | igpu,
  all = cpu | dgpu | igpu
};

inline bool matches_type(device_type filter, device_type dt) {
  return (static_cast<int>(filter) & static_cast<int>(dt)) != 0;
}

void scan(device_type dev, std::uint32_t device, std::uint32_t* a, std::uint32_t n);

#endif __scan_hpp_ // !__scan_hpp_
