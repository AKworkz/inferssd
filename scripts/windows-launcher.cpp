// SPDX-License-Identifier: Apache-2.0
// A relocatable console entry point; pip's generated launcher embeds its build-time Python path.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static std::wstring quoted(const std::wstring & value) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t ch : value) {
        if (ch == L'\\') { ++slashes; continue; }
        out.append(slashes * (ch == L'"' ? 2 : 1), L'\\');
        slashes = 0;
        if (ch == L'"') out += L'\\';
        out += ch;
    }
    out.append(slashes * 2, L'\\');
    return out + L'"';
}

static BOOL WINAPI control(DWORD event) {
    // Python owns graceful Ctrl+C cleanup. Handler registrations are not inherited by children.
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

int wmain(int argc, wchar_t ** argv) {
    // The portable x64 build uses the AVX2 baseline rather than the CI host's native ISA.
    if (!IsProcessorFeaturePresent(40 /* PF_AVX2_INSTRUCTIONS_AVAILABLE */)) {
        std::wcerr << L"inferssd: this Windows x64 package requires an AVX2-capable CPU.\n";
        return 2;
    }
    std::vector<wchar_t> filename(32768);
    DWORD length = GetModuleFileNameW(nullptr, filename.data(), DWORD(filename.size()));
    if (!length || length == filename.size()) return 2;
    // Resolve current/ junctions before a later installation can replace the pointer.
    auto root = std::filesystem::canonical(
        std::filesystem::path(filename.data()).parent_path().parent_path());
    auto python = root / L"python" / L"python.exe";
    auto resources = root / L"engine-resources.json";
    if (!GetEnvironmentVariableW(L"INFERSSD_RESOURCES", nullptr, 0)) {
        SetEnvironmentVariableW(L"INFERSSD_RESOURCES", resources.c_str());
    }
    std::wstring command = quoted(python.wstring()) + L" -I -X utf8 -c " + quoted(
        L"import sys;sys.argv[0]='inferssd';from inferssd.cli import main;sys.exit(main())");
    for (int i = 1; i < argc; ++i) command += L" " + quoted(argv[i]);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    SetConsoleCtrlHandler(control, TRUE);
    if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &startup, &process)) {
        std::wcerr << L"inferssd: cannot start the packaged Python (Windows error "
                   << GetLastError() << L")\n";
        return 2;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD result = 2;
    GetExitCodeProcess(process.hProcess, &result);
    CloseHandle(process.hProcess);
    return int(result);
}
