// SPDX-License-Identifier: Apache-2.0
#include "settings/json.h"
#include "settings/settings.h"
#include "telemetry/telemetry.h"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--events") {
        const auto values = ssd::settings::parse("", "");
        auto       sink   = ssd::telemetry::writer::open(argv[2]);
        if (!values || !sink || !sink.value->settings(values.value) ||
            !sink.value->event("ssd.place.layer", "{\"read_bytes\":12345}") || !sink.value->flush()) {
            return 1;
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--schema") {
        const auto result = ssd::settings::schema_json();
        if (!result) {
            std::cerr << result.error << '\n';
            return 1;
        }
        std::cout << result.value << '\n';
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--parse") {
        const auto result = ssd::settings::load(argv[2], {}, argv[3]);
        if (!result) {
            std::cerr << result.error << '\n';
            return 1;
        }
        const auto json = result.value.json();
        if (!json) {
            std::cerr << json.error << '\n';
            return 1;
        }
        std::cout << json.value << '\n';
        return 0;
    }
    using namespace ssd::settings;
    assert(registry().size() == 13);
    assert(parse("inferssd-settings 1\nmodel.engram_prefetch=1\nmodel.hc_fused=0\n", "model.ssd"));
    assert(!parse("inferssd-settings 1\nmodel.hc_fused=2\n", "model-invalid.ssd"));
    auto base = parse("", "");
    assert(base);
    assert(*base.value.integer("prefetch.depth") == 2);
    assert(base.value.integer("expert.source") == nullptr);
    assert(base.value.find("unknown") == nullptr);
    const std::string text =
        "inferssd-settings 1\n"
        "prefetch.depth = 4 # source=derived; input=measured CPUs\n"
        "cpu.io_cpus = 2-4\n"
        "expert.source = [\"/path # 1\",\"/unicode-\\ud83d\\udc08\"] # preserved\n"
        "place.costs = cpu_fixed_us=204,link_gbps=12.3\n";
    auto parsed = parse(text, "fixture.ssd",
                        {
                            { "cpu.decode_participants", "2" }
    });
    assert(parsed);
    assert(*parsed.value.integer("prefetch.depth") == 4);
    assert(*parsed.value.cpus("cpu.io_cpus") == std::vector<uint32_t>({ 2, 3, 4 }));
    assert(parsed.value.sources()->at(0) == "/path # 1");
    assert(parsed.value.sources()->at(1) == "/unicode-\xf0\x9f\x90\x88");
    assert(parsed.value.find("prefetch.depth")->receipt == " source=derived; input=measured CPUs");
    assert(parsed.value.find("cpu.decode_participants")->origin == "cli");
    auto copy = parsed.value;
    parsed    = parse("", "");
    assert(*copy.integer("prefetch.depth") == 4 && *parsed.value.integer("prefetch.depth") == 2);
    for (const std::string bad :
         { "cpu.threads = 4", "cpu.compute_cpus = 0-1", "prefetch.depth = \"4\"", "prefetch.depth = 18446744073709551616",
           "cpu.io_cpus = 1,1", "cpu.io_cpus = 2-0", "cpu.io_cpus = 1-99999999", "cpu.decode_participants = 256", "unknown = 0",
           "place.costs = cpu_fixed_us=nan", "place.costs = link_gbps=0",
           "place.costs = unknown=1", "place.costs = cpu_fixed_us=1,cpu_fixed_us=2", "vram.budget = \"123\"",
           "expert.source = [1]", "host.expert_cache_mib = 0", "prefetch.depth = 65" }) {
        const auto rejected = parse("inferssd-settings 1\n" + bad + "\n", "bad.ssd");
        if (rejected || rejected.error.find("bad.ssd:2") == std::string::npos) {
            std::cerr << bad << ": " << rejected.error << '\n';
        }
        assert(!rejected && rejected.code == ssd::status::invalid &&
               rejected.error.find("bad.ssd:2") != std::string::npos);
    }
    assert(!parse("inferssd-settings 1\nprefetch.depth=2\nprefetch.depth=4\n", "duplicate.ssd"));
    assert(!parse("", "empty.ssd"));
    assert(!parse("inferssd-settings 2\n", "version.ssd"));
    assert(parse("inferssd-settings 1\nprefetch.depth=8\n", "mac.ssd", {}, "darwin"));
    assert(!parse("inferssd-settings 1\nhost.expert_cache_mib=12288\n", "mac.ssd", {}, "darwin"));
    assert(parse("inferssd-settings 1\nvram.budget=0\n", "mac.ssd", {}, "darwin"));
    // The executor's keys apply on every platform. The CUDA tier's keys too: the engine refuses them where it runs
    // without the tier, whatever the platform.
    assert(parse("inferssd-settings 1\ncpu.io_cpus=3\ncpu.decode_participants=2\nprefetch.depth=4\n"
                 "place.costs=cpu_fixed_us=204\nhost.headroom_mib=1024\nhost.budget_mib=8192\n"
                 "model.hc_fused=0\n", "win.ssd", {}, "windows"));
    assert(parse("inferssd-settings 1\ncpu.io_cpus=0-2\ncpu.decode_participants=1\nhost.budget_mib=8192\n", "mac.ssd", {},
                 "darwin"));
    for (const std::string platform : { "linux", "darwin", "windows" }) {
        assert(parse("inferssd-settings 1\nvram.max_slots=1\nprefill.cpu_max_tokens=1\nvram.reserve_mib=64\n", "tier.ssd",
                     {}, platform));
    }
    assert(!parse("inferssd-settings 1\nhost.budget_mib=1\n", "linux.ssd"));
    assert(!parse("", "", {}, "freebsd"));
    {
        const auto scopes = ssd::json::parse(schema_json().value).value.object.at("properties").object;
        assert(scopes.at("host.budget_mib").object.at("scope").text == "darwin,windows");
        assert(scopes.at("cpu.io_cpus").object.at("scope").text == "all");
        assert(scopes.at("cpu.decode_participants").object.at("scope").text == "all");
        assert(scopes.at("vram.max_slots").object.at("scope").text == "all");
        assert(scopes.at("vram.budget").object.at("scope").text == "all");
        // The CUDA tier's keys name their backend; the others have none.
        for (const char * key : { "prefill.cpu_max_tokens", "vram.max_slots", "vram.reserve_mib" }) {
            assert(scopes.at(key).object.at("backend").text == "CUDA");
        }
        assert(!scopes.at("vram.budget").object.count("backend") && !scopes.at("cpu.io_cpus").object.count("backend"));
    }
    assert(reject_retired_environment({ "TASKRC", "TASKDATA", "INFERSSD_DEBUG" }));
    assert(!reject_retired_environment({ "TASK123_PROBE" }));
    assert(!reject_retired_environment({ "GGML_KIMI_CPU_SINK" }));
    std::vector<override_value> cli;
    assert(cli_override("--ssd-prefetch-depth", "6", cli));
    assert(cli.size() == 1 && cli[0].key == "prefetch.depth");
    assert(!cli_override("--ssd-cpu-threads", "6", cli));
    assert(!cli_override("--ssd-unknown", "3", cli));
    const auto unrelated = cli_override("--port", "8080", cli);
    assert(unrelated && !unrelated.value);
    const auto schema = schema_json();
    assert(schema && ssd::json::parse(schema.value).value.object.at("properties").object.size() == registry().size());
    const auto help_text = help();
    assert(help_text && help_text.value.find("--ssd-host-budget-mib") != std::string::npos);

    char directory[] = "/tmp/ssdash-config-test-XXXXXX";
    assert(mkdtemp(directory));
    const std::string path = std::string(directory) + "/events.jsonl";
    auto              sink = ssd::telemetry::writer::open(path);
    assert(sink);
    assert(!ssd::telemetry::writer::open(path));
    assert(sink.value->settings(copy));
    assert(!sink.value->event("ssd.bad", "{\"v\":99}"));
    assert(!sink.value->event("ssd.bad", "{\"x\":1,\"x\":2}"));
    assert(!sink.value->event("ssd.bad", "{\"x\":nan}"));
    assert(!sink.value->event("ssd.bad", "{\"x\":1e999}"));
    assert(!sink.value->event("ssd.bad", std::string("{\"x\":\"") + char(0xff) + "\"}"));
    assert(!sink.value->event("ssd..bad", "{}"));
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&, worker] {
            for (int n = 0; n < 100; ++n) {
                assert(sink.value->event(
                    "ssd.io.read", "{\"worker\":" + std::to_string(worker) + ",\"n\":" + std::to_string(n) + '}'));
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    assert(sink.value->flush());
    sink.value.reset();
    std::ifstream file(path);
    std::string   line;
    uint64_t      sequence = 0;
    while (std::getline(file, line)) {
        const auto node = ssd::json::parse(line);
        assert(node.value.object.at("v").text == "1" && node.value.object.at("seq").text == std::to_string(sequence++));
    }
    assert(sequence == 401);
    const std::string failed_path = std::string(directory) + "/write-failure.jsonl";
    const pid_t       child       = fork();
    assert(child >= 0);
    if (!child) {
        auto failed = ssd::telemetry::writer::open(failed_path);
        if (!failed) {
            _exit(1);
        }
        signal(SIGXFSZ, SIG_IGN);
        const struct rlimit limit = { 0, 0 };
        if (setrlimit(RLIMIT_FSIZE, &limit)) {
            _exit(2);
        }
        const auto first   = failed.value->event("ssd.io.read", "{}");
        const auto second  = failed.value->event("ssd.io.read", "{}");
        const auto third   = failed.value->flush();
        const bool refused = !first && first.code == ssd::status::io && !second && second.code == ssd::status::io &&
                             !third && third.code == ssd::status::io;
        _exit(refused ? 0 : 3);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    unlink(failed_path.c_str());
    unlink(path.c_str());
    rmdir(directory);
    std::cout << "settings/telemetry contracts passed\n";
}
