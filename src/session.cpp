#include "session.hpp"
#include <stdexcept>

namespace cgemma {

session::session(unsigned int seed, int argc, char* argv[])
  : rnd_(seed)
  , args_(argc, argv) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
}

}
