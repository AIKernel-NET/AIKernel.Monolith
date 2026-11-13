#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>

typedef int32_t (*get_count_fn)(int32_t *);
typedef int32_t (*get_pci_fn)(int32_t, char *, int32_t);
typedef int32_t (*create_fn)(uint32_t, int32_t, void **);
typedef int32_t (*alloc_fn)(void *, size_t, uint64_t *);
typedef int32_t (*wipe_fn)(void *, uint64_t, size_t);

static int claim_first_exit(void)
{
    wchar_t path[32768];
    DWORD length = GetEnvironmentVariableW(L"AIKERNEL_TEST_PROBE_FIRST_EXIT_FILE", path, ARRAYSIZE(path));
    if (length == 0)
        return 0;
    if (length >= ARRAYSIZE(path))
        return -1;
    HANDLE marker = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (marker == INVALID_HANDLE_VALUE)
        return GetLastError() == ERROR_FILE_EXISTS ? 0 : -1;
    CloseHandle(marker);
    return 1;
}

/* Initialize the real CUDA driver and leave valid topology on stdout while
   the test triggers an abnormal exit or lets Monolith time out the probe. */
__declspec(dllexport) int32_t ak_get_device_count(int32_t *count)
{
    wchar_t path[32768];
    DWORD length = GetEnvironmentVariableW(L"AIKERNEL_TEST_REAL_CUDA_LIBRARY", path, ARRAYSIZE(path));
    if (length == 0 || length >= ARRAYSIZE(path))
        return 1;

    HMODULE driver = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!driver)
        return 2;

    get_count_fn get_count = (get_count_fn)GetProcAddress(driver, "ak_get_device_count");
    get_pci_fn get_pci = (get_pci_fn)GetProcAddress(driver, "ak_get_device_pci_bus_id");
    create_fn create = (create_fn)GetProcAddress(driver, "ak_create");
    alloc_fn allocate = (alloc_fn)GetProcAddress(driver, "ak_device_alloc");
    wipe_fn wipe = (wipe_fn)GetProcAddress(driver, "ak_wipe_device");
    if (!get_count || !get_pci || !create || !allocate || !wipe || get_count(count) != 0 || *count < 2)
        ExitProcess(74);
    int first_exit = claim_first_exit();
    if (first_exit < 0)
        ExitProcess(74);

    const size_t bytes = 8 * 1024 * 1024;
    fputc('[', stdout);
    for (int32_t ordinal = 0; ordinal < *count; ++ordinal)
    {
        char pci[32] = { 0 };
        void *context = NULL;
        uint64_t address = 0;
        if (get_pci(ordinal, pci, sizeof(pci)) != 0 ||
            create(1, ordinal, &context) != 0 || allocate(context, bytes, &address) != 0 ||
            wipe(context, address, bytes) != 0)
            ExitProcess(74);
        printf("%s\"%s\"", ordinal == 0 ? "" : ",", pci);
        fprintf(stderr, "CUDA probe %lu allocated %zu B on %s\n", GetCurrentProcessId(), bytes, pci);
    }
    fputc(']', stdout);
    fflush(stdout);
    fprintf(stderr, "CUDA probe fault fixture ready pid=%lu devices=%ld\n",
        GetCurrentProcessId(), (long)*count);
    fflush(stderr);
    if (first_exit)
        ExitProcess(73);

    // Keep the allocations live until the test releases the probe or Monolith terminates it.
    wchar_t release[32768];
    DWORD release_length = GetEnvironmentVariableW(L"AIKERNEL_TEST_PROBE_RELEASE_FILE",
        release, ARRAYSIZE(release));
    ULONGLONG deadline = GetTickCount64() + 30000;
    while (GetTickCount64() < deadline)
    {
        if (release_length > 0 && release_length < ARRAYSIZE(release) &&
            GetFileAttributesW(release) != INVALID_FILE_ATTRIBUTES)
            ExitProcess(0);
        Sleep(50);
    }
    ExitProcess(74);
}
