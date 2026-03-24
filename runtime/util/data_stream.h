#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"

// A simple data stream for reading data in chunks.
// This is necessary for loading models within Wasm32, which has a 4GB memory
// limit. This limitation prevents us from loading the entire model into memory
// at once.
//
// uint64_t is used instead of size_t because we need to index into a 4GB+ model
// file in Wasm32 (where size_t is 32 bits).

namespace litert::lm {

class SubStream;  // Forward declaration

class DataStream {
 public:
  virtual ~DataStream() = default;

  virtual absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                                      uint64_t size) = 0;

  virtual absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                                       uint64_t size) = 0;

  // Note: Discarding a region that has already been discarded must be
  // supported.
  virtual absl::Status Discard(uint64_t offset, uint64_t size) = 0;

  virtual absl::StatusOr<std::unique_ptr<DataStream>> OpenSubStream(
      uint64_t offset, uint64_t size);
};

// This is not thread-safe.
// Note: SubStream holds a raw pointer to the parent DataStream, so the
// parent DataStream must outlive the SubStream.
class SubStream : public DataStream {
 public:
  SubStream(DataStream* parent, uint64_t offset, uint64_t size)
      : parent_(parent), offset_(offset), size_(size) {}

  ~SubStream() override {
    // Note: We ignore errors here as we are in a destructor.
    (void)parent_->Discard(offset_, size_);
  }

  absl::Status ReadAndDiscard(void* buffer, uint64_t offset,
                              uint64_t size) override {
    RETURN_IF_ERROR(CheckBounds(offset, size));
    return parent_->ReadAndDiscard(buffer, offset_ + offset, size);
  }

  absl::Status ReadAndPreserve(void* buffer, uint64_t offset,
                               uint64_t size) override {
    RETURN_IF_ERROR(CheckBounds(offset, size));
    return parent_->ReadAndPreserve(buffer, offset_ + offset, size);
  }

  absl::Status Discard(uint64_t offset, uint64_t size) override {
    // Equivalent to `offset + size > size_`
    RETURN_IF_ERROR(CheckBounds(offset, size));
    return parent_->Discard(offset_ + offset, size);
  }

 private:
  DataStream* parent_;
  uint64_t offset_;
  uint64_t size_;

  absl::Status CheckBounds(uint64_t offset, uint64_t size) const {
    if (size > size_ || offset > size_ - size) {
      return absl::OutOfRangeError(
          absl::StrCat("Exceeded bounds of substream. offset: ", offset,
                       ", size: ", size, ", Max size: ", size_));
    }
    return absl::OkStatus();
  }
};

inline absl::StatusOr<std::unique_ptr<DataStream>> DataStream::OpenSubStream(
    uint64_t offset, uint64_t size) {
  return std::make_unique<SubStream>(this, offset, size);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_DATA_STREAM_H_
