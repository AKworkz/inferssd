// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../ggml/src/ggml-ssd/settings/settings.h"
#include "../../ggml/src/ggml-cpu/ssd/api.h"
#include "../../ggml/src/ggml-ssd/gpu/provider.h"
#include <dirent.h>
#include <csignal>

static size_t configured_task_count() {
    DIR * directory = opendir("/proc/self/task");
    require(directory != nullptr, "open task inventory");
    size_t count = 0;
    while (const auto * entry = readdir(directory)) { if (entry->d_name[0] != '.') { ++count; } }
    closedir(directory);
    return count;
}

static ssd::settings::snapshot configured_settings(bool invalid_affinity = false, int prefetch = 0) {
    std::vector<ssd::settings::override_value> values = {
        { "prefetch.depth", std::to_string(prefetch) }, { "vram.budget", "0" },
    };
    if (invalid_affinity) { values.push_back({ "cpu.io_cpus", std::to_string(CPU_SETSIZE) }); }
    const auto parsed = ssd::settings::parse("", "", values);
    require(bool(parsed), parsed.error);
    return parsed.value;
}

static void configured_writer_failure_test(fixture & f, const ssd::settings::snapshot & settings,
        const std::string & path, const std::vector<float> & input) {
    const pid_t child = fork(); require(child >= 0, "fork configured writer failure");
    if (!child) {
        auto writer = ssd::telemetry::writer::open(path);
        if (!writer) { _exit(1); }
        auto owner = std::make_shared<ssd::host::process_arbiter>();
        auto made = owner->make_client(); if (!made) { _exit(5); }
        auto client = std::move(made.value);
        const ssd::runtime_services services{ writer.value.get(), nullptr, client.get() };
        char error[512] = {};
        sink_ptr sink(ssd::cpu::create_configured(&settings, &services, fixture::resolve, &f, error, sizeof(error)));
        if (!sink) { _exit(2); }
        signal(SIGXFSZ, SIG_IGN);
        const struct rlimit limit = { 0, 0 };
        if (setrlimit(RLIMIT_FSIZE, &limit)) { _exit(3); }
        tensors meta(f, 1); const int32_t id = 1; const float weight = 1;
        std::vector<float> output(f.latent); kimi_cpu_sink_stats stats = {};
        const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id,
            &weight, 1, 4, 25, output.data(), &stats, error, sizeof(error));
        sink.reset();
        _exit(!ok && stats.pending_reads == 0 && stats.read_requests == stats.read_completions && error[0] ? 0 : 4);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "real telemetry write failure did not reject publication after draining");
    unlink(path.c_str());
}

static void configured_runtime_tests(fixture_inputs & inputs) {
    fixture f; make_synthetic(f);
    const size_t before = configured_task_count();
    auto settings = configured_settings();
    const auto original = settings;
    const std::string path = f.path + ".jsonl";
    struct cleanup { std::string path; ~cleanup() { unlink(path.c_str()); } } cleanup{ path };
    auto writer = ssd::telemetry::writer::open(path);
    require(bool(writer) && bool(writer.value->settings(settings)), "configured settings publication");
    ssd::telemetry::debug_channels debug; debug.profile = true;
    auto owner = std::make_shared<ssd::host::process_arbiter>();
    auto made_first = owner->make_client(), made_second = owner->make_client();
    require(bool(made_first) && bool(made_second), "shared host client registration");
    auto first_client = std::move(made_first.value), second_client = std::move(made_second.value);
    const ssd::runtime_services services{ writer.value.get(), &debug, first_client.get() };
    char error[512] = {};
    const auto reg = ggml_backend_cpu_reg();
    const auto getter = reinterpret_cast<ssd::cpu::get_api_v2>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_ssd_api_v2"));
    require(getter && ssd::cpu::valid(getter()), "normal CPU registry table/layout");
    auto wrong_layout = *getter(); --wrong_layout.runtime_services_size;
    require(!ssd::cpu::valid(&wrong_layout), "mixed runtime services layout accepted");
    auto short_table = *getter(); --short_table.size;
    require(!ssd::cpu::valid(&short_table), "mixed CPU provider table accepted");
    auto bad_services = services;
    ssd::gpu::provider_v1 missing_gpu{}; bad_services.gpu = &missing_gpu;
    sink_ptr rejected_gpu(ssd::cpu::create_configured(&settings, &bad_services, fixture::resolve, &f, error, sizeof(error)));
    require(!rejected_gpu && error[0] && configured_task_count() == before, "GPU ABI rejection started a worker");
    sink_ptr first(ssd::cpu::create_configured(&settings, &services, fixture::resolve, &f, error, sizeof(error)));
    require(bool(first) && configured_task_count() == before + 1, std::string("configured create: ") + error);
    settings = configured_settings(false, 1); debug.profile = false;
    const ssd::runtime_services second_services{ nullptr, &debug, second_client.get() };
    sink_ptr second(ssd::cpu::create_configured(&settings, &second_services, fixture::resolve, &f, error, sizeof(error)));
    require(bool(second) && configured_task_count() == before + 5, "independent second context thread count");
    auto invalid = configured_settings(true);
    sink_ptr rejected(ssd::cpu::create_configured(&invalid, &services, fixture::resolve, &f, error, sizeof(error)));
    require(!rejected && error[0] && configured_task_count() == before + 5, "affinity rejection leaked a started worker");
    require(!ssd::cpu::cache_make_room_configured(nullptr, &original, nullptr, 0, 0),
        "missing early registration silently admitted allocation");
    auto first_hold = ssd::cpu::take_room_configured(nullptr, &original, &services, 0, 0, 11);
    auto second_hold = ssd::cpu::take_room_configured(nullptr, &original, &second_services, 0, 0, 17);
    require(bool(first_hold) && bool(second_hold) && bool(first_hold.value) && bool(second_hold.value), "zero-byte configured tickets");
    first_hold.value.cancel(); second_hold.value.cancel();
    auto held = first_client->state();
    require(bool(held) && held.value.prefill_pages == 11 && held.value.all_prefill_pages == 28, "context holds not aggregated");
    auto keep = ssd::cpu::take_room_configured(nullptr, &original, &services, 0, 0, UINT64_MAX);
    require(bool(keep) && first_client->state().value.prefill_pages == 11, "keep sentinel changed own hold"); keep.value.cancel();
    auto clear = ssd::cpu::take_room_configured(nullptr, &original, &services, 0, 0, 0);
    require(bool(clear) && first_client->state().value.all_prefill_pages == 17, "clearing own hold cleared peer"); clear.value.cancel();
    auto mismatch = ssd::cpu::take_room_configured(first.get(), &original, &second_services, 0, 0, UINT64_MAX);
    require(!mismatch && mismatch.code == ssd::status::invalid, "cross-context executor accepted wrong client");
    require(ssd::cpu::cache_make_room_configured(nullptr, &original, &services, 0, 0) &&
        ssd::cpu::cache_prefill_room_configured(nullptr, &original, &second_services, 0, 0, 0), "early configured shared admission");
    const std::vector<int32_t> ids{ 1, 2 }; const std::vector<float> weights{ .4f, .6f };
    const std::vector<float> input(f.latent, .13f);
    const auto expected = reference(f, 1, input, ids, weights, 4, 25);
    tensors meta(f, 1); std::vector<float> a, b;
    run_checked(inputs, "configured_profile", f, first.get(), meta, input, ids, weights, expected, 1, 25, 0, 0, &a);
    run_checked(inputs, "configured_quiet", f, second.get(), meta, input, ids, weights, expected, 1, 25, 0, 0, &b);
    require(a == b, "debug channels or independent thread selection changed CPU output bits");
    require(first_client->state().value.pending_bytes == 0 && second_client->state().value.pending_bytes == 0,
        "actual ring/cache allocation leaked shared pending debit");
    first.reset(); second.reset(); writer.value.reset();
    require(configured_task_count() == before, "configured context destruction left worker threads");
    std::ifstream records(path); std::string line; size_t sequence = 0; bool profile = false;
    while (std::getline(records, line)) {
        const auto event = nlohmann::json::parse(line);
        require(event.at("v") == 1 && event.at("seq") == sequence++, "configured JSONL sequence/version");
        if (event.at("ev") == "ssd.settings") { require(event.at("values").at("cpu.decode_participants") == 0 && event.at("values").at("prefetch.depth") == 0, "owned settings receipt changed"); }
        else { profile |= event.at("compute_thread_cpu_us").get<double>() > 0; }
    }
    require(sequence == 3 && profile, "copied debug selection or actual layer telemetry missing");
    configured_writer_failure_test(f, original, path + ".failure", input);
    std::printf("CONFIGURED_RUNTIME_IO_GOLDEN_PASS contexts=2 canonical_ring=8 owned_snapshot=1 owned_debug=1\n");
}
