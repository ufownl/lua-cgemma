#ifndef CGEMMA_SESSION_HPP
#define CGEMMA_SESSION_HPP

#include <lua.hpp>
#include <gemma.h>
#include <random>

namespace cgemma {

class session {
public:
  explicit session(unsigned int seed, int argc, char* argv[]);

  std::mt19937& rnd() { return rnd_; }
  const gcpp::InferenceArgs& args() const { return args_; }
  size_t pos() const { return pos_; }

  void incr_pos(size_t n) { pos_ += n; }

private:
  std::mt19937 rnd_;
  gcpp::InferenceArgs args_;
  size_t pos_ {0};
};

}

#endif  // CGEMMA_SESSION_HPP
