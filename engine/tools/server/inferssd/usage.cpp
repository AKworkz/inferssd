// SPDX-License-Identifier: Apache-2.0
#include "usage.h"

#include "../server-common.h"

#include <fcntl.h>
#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    include <io.h>
#    include <sys/stat.h>
#else
#    include <unistd.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <system_error>
#include <vector>

static const char * USAGE_FILE = "usage-events.jsonl";

namespace {

std::filesystem::path usage_directory() {
#if defined(_WIN32)
    const wchar_t * value = _wgetenv(L"INFERSSD_USAGE_DIR");
#else
    const char * value = std::getenv("INFERSSD_USAGE_DIR");
#endif
    if (value == nullptr || *value == 0) {
        return {};
    }
    return std::filesystem::path(value);
}

// Appending, non-inheritable, private to the user.
int open_append(const std::filesystem::path & path) {
#if defined(_WIN32)
    return _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_APPEND | _O_BINARY | _O_NOINHERIT, _S_IREAD | _S_IWRITE);
#else
    return ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
#endif
}

void close_fd(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    ::close(fd);
#endif
}

bool write_all(int fd, const std::string & data) {
    size_t done = 0;
    while (done < data.size()) {
#if defined(_WIN32)
        const int n = _write(fd, data.data() + done, (unsigned int) (data.size() - done));
#else
        const ssize_t n = ::write(fd, data.data() + done, data.size() - done);
#endif
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        done += size_t(n);
    }
    return true;
}

// UTC, to the second: 2026-11-10T04:59:58Z
std::string utc_now() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm           tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return text;
}

const char * outcome_name(server_usage_outcome outcome) {
    switch (outcome) {
        case SERVER_USAGE_COMPLETED:
            return "completed";
        case SERVER_USAGE_CANCELLED:
            return "cancelled";
        case SERVER_USAGE_FAILED:
        default:
            return "failed";
    }
}

}  // namespace

//
// server_usage_request
//

server_usage_request::server_usage_request(std::shared_ptr<server_usage_log> log, std::string request_id) :
    log(std::move(log)),
    request_id(std::move(request_id)) {}

server_usage_request::~server_usage_request() {
    try {
        finish();
        log->forget(this);
    } catch (const std::exception & e) {
        SRV_WRN("usage events: %s\n", e.what());
    } catch (...) {
    }
}

void server_usage_request::add(int64_t input, int64_t output) {
    input_tokens += input;
    output_tokens += output;
}

void server_usage_request::set_outcome(server_usage_outcome value) {
    outcome = value;
}

void server_usage_request::finish() {
    if (finished.exchange(true)) {
        return;
    }
    log->usage_finalized(request_id, server_usage_outcome(outcome.load()), input_tokens.load(), output_tokens.load());
}

//
// server_usage_log
//

// Usage events never stop the server: any failure here leaves them off and says so.
std::shared_ptr<server_usage_log> server_usage_log::from_environment() {
    try {
        const std::filesystem::path directory = usage_directory();
        if (directory.empty()) {
            return nullptr;
        }
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        const std::filesystem::path path = directory / USAGE_FILE;
        const int                   fd   = open_append(path);
        if (fd < 0) {
            SRV_WRN("usage events: cannot open %s: %s; no usage is recorded\n", path.u8string().c_str(),
                    std::strerror(errno));
            return nullptr;
        }
        std::shared_ptr<server_usage_log> log(new server_usage_log(fd, path));
        SRV_INF("usage events: run %s appends to %s\n", log->run_id.c_str(), path.u8string().c_str());
        return log;
    } catch (const std::exception & e) {
        SRV_WRN("usage events: %s; no usage is recorded\n", e.what());
        return nullptr;
    }
}

server_usage_log::server_usage_log(int fd, std::filesystem::path path) :
    fd(fd),
    path(std::move(path)),
    run_id(random_string().substr(0, 16)) {}

server_usage_log::~server_usage_log() {
    close_fd(fd);
}

server_usage_request_ptr server_usage_log::begin() {
    const std::string request_id = "req-" + random_string().substr(0, 24);
    auto              request    = std::make_shared<server_usage_request>(shared_from_this(), request_id);
    std::lock_guard<std::mutex> lock(mutex);
    open[request.get()] = request;
    request_received_locked(request_id);
    return request;
}

void server_usage_log::finish_open() {
    std::vector<server_usage_request_ptr> requests;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & [_, weak] : open) {
            if (auto request = weak.lock()) {
                requests.push_back(std::move(request));
            }
        }
    }
    for (auto & request : requests) {
        request->finish();
    }
}

void server_usage_log::request_received_locked(const std::string & request_id) {
    json line = {
        { "event_id",    run_id + ":" + std::to_string(++seq) },
        { "type",        "request_received"                   },
        { "request_id",  request_id                           },
        { "occurred_at", utc_now()                            },
    };
    write_locked(line.dump());
}

void server_usage_log::usage_finalized(const std::string & request_id, server_usage_outcome outcome,
                                       int64_t input_tokens, int64_t output_tokens) {
    std::lock_guard<std::mutex> lock(mutex);
    json line = {
        { "event_id",      run_id + ":" + std::to_string(++seq) },
        { "type",          "usage_finalized"                    },
        { "request_id",    request_id                           },
        { "occurred_at",   utc_now()                            },
        { "token_count",   input_tokens + output_tokens         },
        { "input_tokens",  input_tokens                         },
        { "output_tokens", output_tokens                        },
        { "outcome",       outcome_name(outcome)                },
    };
    write_locked(line.dump());
}

void server_usage_log::forget(const server_usage_request * request) {
    std::lock_guard<std::mutex> lock(mutex);
    open.erase(request);
}

void server_usage_log::write_locked(std::string line) {
    if (failed) {
        return;
    }
    line += '\n';
    if (!write_all(fd, line)) {
        // A failed write may leave part of a line; stop rather than glue the next record to it.
        failed = true;
        SRV_WRN("usage events: writing %s failed: %s; no further usage is recorded\n", path.u8string().c_str(),
                std::strerror(errno));
    }
}
