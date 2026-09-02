// light-weight header only tinmer
// Author: Xiaohan Fei
#pragma once

#include <ostream>
#include <chrono>
#include <memory>
#include <unordered_map>

namespace xivo {

/// \brief timer
class Timer {
public:
  struct Event {
    std::chrono::high_resolution_clock::time_point latest;  // latest start time
    std::chrono::nanoseconds duration; // total duration
    int occurrence;  // how many 
  };
public:

  friend std::ostream &operator<<(std::ostream &os, const Timer& t) {
    os << "....." << std::endl;
    for (const auto &p : t.data_) {
      const auto &e{p.second};
      // Average in nanoseconds, not milliseconds. Casting the *total* to integer
      // ms first quantized the per-call mean to 1/occurrence ms and biased it
      // low by up to that much -- which is how two unrelated stages whose totals
      // landed in the same integer-ms bin came to print the bit-identical
      // 0.240924 ms (73 ms / 303 calls) and cost an afternoon to explain.
      const double ms = e.duration.count() / 1e6 / (double)e.occurrence;
      os << "[" << t.name_ << "]"
        << p.first
        // The occurrence count is printed because a per-call mean is not a
        // per-frame cost: detection runs on ~8% of frames, so its 1.9 ms/call is
        // 0.16 ms/frame, and every reader has to do that division.
        << ":" << ms << " ms x" << e.occurrence << "\n";
    }
    return os;
  }

  Timer(const std::string &name = "default")
      : name_{name} {}

  void Tick(const std::string &event) {
    data_[event].latest = std::chrono::high_resolution_clock::now();
  }

  auto Tock(const std::string &event) -> std::chrono::milliseconds {
    auto duration = SingleOccurrenceDuration(event);

    if (data_.count(event)) {
      Event& e = data_[event];
      e.duration += duration; 
      e.occurrence += 1;
    } else {
      Event& e = data_[event];
      e.duration = duration; 
      e.occurrence = 1;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  }

  void Reset() {
    data_.clear();
  }
  virtual ~Timer() = default;

protected:
  std::chrono::nanoseconds SingleOccurrenceDuration(const std::string &event) const {
    auto tmp = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        tmp - data_.at(event).latest);
  }

  std::unordered_map<std::string, Event> data_;
  std::string name_;
};

}


