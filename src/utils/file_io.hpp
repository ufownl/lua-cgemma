#ifndef CGEMMA_UTILS_FILE_IO_HPP
#define CGEMMA_UTILS_FILE_IO_HPP

#include <filesystem>

namespace cgemma { namespace utils {

class fio_base {
public:
  fio_base() = default;
  fio_base(const fio_base&) = delete;
  fio_base(fio_base&&) = delete;

  virtual ~fio_base();

  fio_base& operator=(const fio_base&) = delete;
  fio_base& operator=(fio_base&&) = delete;

  const char* buffer() const {
    return static_cast<char*>(buf_);
  }

  char* buffer() {
    return static_cast<char*>(buf_);
  }

  size_t size() const {
    return len_;
  }

protected:
  void init(int fd, void* buf, size_t len) {
    fd_ = fd;
    buf_ = buf;
    len_ = len;
  }

private:
  int fd_ {-1};
  void* buf_ {nullptr};
  size_t len_ {0};
};

class file_reader: public fio_base {
public:
  file_reader(const std::filesystem::path& path);
};

class file_writer: public fio_base {
public:
  file_writer(const std::filesystem::path& path, size_t len);
};

} }

#endif  // CGEMMA_UTILS_FILE_IO_HPP
