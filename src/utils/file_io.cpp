#include "file_io.hpp"
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

namespace cgemma { namespace utils {

fio_base::~fio_base() {
  if (buf_ && len_ > 0) {
    munmap(buf_, len_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

file_reader::file_reader(const std::filesystem::path& path) {
  auto fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::filesystem::filesystem_error("failed to open file", path, std::make_error_code(std::errc(errno)));
  }
  struct stat fs = {0};
  if (fstat(fd, &fs) == -1) {
    close(fd);
    throw std::filesystem::filesystem_error("failed to get file stat", path, std::make_error_code(std::errc(errno)));
  }
  if (fs.st_size == 0) {
    close(fd);
    return;
  }
  auto buf = mmap(nullptr, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buf == MAP_FAILED) {
    close(fd);
    throw std::filesystem::filesystem_error("failed to mmap file", path, std::make_error_code(std::errc(errno)));
  }
  madvise(buf, fs.st_size, MADV_WILLNEED | MADV_SEQUENTIAL);
  init(fd, buf, fs.st_size);
}

file_writer::file_writer(const std::filesystem::path& path, size_t len) {
  auto fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    throw std::filesystem::filesystem_error("failed to open file", path, std::make_error_code(std::errc(errno)));
  }
  if (ftruncate(fd, len) == -1) {
    close(fd);
    throw std::filesystem::filesystem_error("failed to truncate file", path, std::make_error_code(std::errc(errno)));
  }
  auto buf = mmap(nullptr, len, PROT_WRITE, MAP_SHARED, fd, 0);
  if (buf == MAP_FAILED) {
    close(fd);
    throw std::filesystem::filesystem_error("failed to mmap file", path, std::make_error_code(std::errc(errno)));
  }
  init(fd, buf, len);
}

} }
