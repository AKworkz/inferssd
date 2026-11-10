// SPDX-License-Identifier: Apache-2.0
#pragma once

// Usage events for the host that runs the server: a request_received line when an inference request reaches its
// route and at most one usage_finalized line when it ends. INFERSSD_USAGE_DIR selects the directory; without it
// nothing is written. The format is described in docs/usage-events.md.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

enum server_usage_outcome {
    SERVER_USAGE_FAILED,
    SERVER_USAGE_CANCELLED,
    SERVER_USAGE_COMPLETED,
};

struct server_usage_log;

// One HTTP request. The route and each of its tasks hold a reference; usage_finalized is written when the last
// reference goes, or earlier by finish() when the server stops.
struct server_usage_request {
    server_usage_request(std::shared_ptr<server_usage_log> log, std::string request_id);
    ~server_usage_request();

    server_usage_request(const server_usage_request &)             = delete;
    server_usage_request & operator=(const server_usage_request &) = delete;

    // the counts of one task that reached a slot
    void add(int64_t input_tokens, int64_t output_tokens);
    void set_outcome(server_usage_outcome outcome);
    // write usage_finalized unless it is already written
    void finish();

  private:
    std::shared_ptr<server_usage_log> log;
    std::string                       request_id;
    std::atomic<int64_t>              input_tokens{ 0 };
    std::atomic<int64_t>              output_tokens{ 0 };
    std::atomic<int>                  outcome{ SERVER_USAGE_FAILED };
    std::atomic<bool>                 finished{ false };
};

using server_usage_request_ptr = std::shared_ptr<server_usage_request>;

struct server_usage_log : std::enable_shared_from_this<server_usage_log> {
    // the log INFERSSD_USAGE_DIR selects, or nullptr when it is unset or its file cannot be opened
    static std::shared_ptr<server_usage_log> from_environment();

    ~server_usage_log();

    // write request_received for a new request
    server_usage_request_ptr begin();

    // finish every request still open; the server calls this after its loop has stopped
    void finish_open();

  private:
    friend struct server_usage_request;

    server_usage_log(int fd, std::filesystem::path path);

    void request_received_locked(const std::string & request_id);
    void usage_finalized(const std::string & request_id, server_usage_outcome outcome, int64_t input_tokens,
                         int64_t output_tokens);
    void forget(const server_usage_request * request);
    // append one complete line; the caller holds mutex
    void write_locked(std::string line);

    std::mutex            mutex;
    int                   fd;
    std::filesystem::path path;
    std::string           run_id;
    uint64_t              seq    = 0;
    bool                  failed = false;

    std::unordered_map<const server_usage_request *, std::weak_ptr<server_usage_request>> open;
};
