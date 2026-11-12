#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    // Apply POSIX_MADV_RANDOM to every page touched by [first, last).
    // Unlike dontneed_fragment(), this expands to page boundaries because
    // dropping readahead for any byte in the requested range is required.
    bool advise_random_fragment(size_t first, size_t last, size_t * len_out = nullptr);
    // Advise only whole pages contained in [first, last), preserving the mapping.
    // Success reports accepted advice, not an amount of physical memory reclaimed.
    bool dontneed_fragment(size_t first, size_t last, size_t * len_out = nullptr);
    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();

// Ask the OS to read ahead the pages that hold ROWS of a table of N_ROWS rows of ROW_BYTES bytes at
// TABLE, a file mapping: madvise(MADV_WILLNEED) on POSIX, PrefetchVirtualMemory on Windows. Advice
// only: rows outside the table are skipped, and without it the rows come in through page faults.
// The pages are worked out here and the advice is given on a background thread, in the order asked,
// so the call returns at once: the OS queues a page's read before the advice returns, so advice on
// many pages lasts as long as the drive takes to read them. A mapping is unmapped only once the
// advice given on it is done. False when the OS has no such advice.
bool llama_prefetch_rows(const void * table, size_t row_bytes, int64_t n_rows, const int32_t * rows, size_t count);
