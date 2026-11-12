#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int32_t (*get_count_fn)(int32_t *);
typedef int32_t (*get_pci_fn)(int32_t, char *, int32_t);

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

    FARPROC count_symbol = GetProcAddress(driver, "ak_get_device_count");
    FARPROC pci_symbol = GetProcAddress(driver, "ak_get_device_pci_bus_id");
    int32_t status = 5;
    if (count_symbol && pci_symbol)
    {
        get_count_fn get_count;
        get_pci_fn get_pci;
        memcpy(&get_count, &count_symbol, sizeof(get_count));
        memcpy(&get_pci, &pci_symbol, sizeof(get_pci));
        status = get_count(count);
        if (status == 0 && *count >= 2)
        {
            fputc('[', stdout);
            for (int32_t ordinal = 0; ordinal < *count; ++ordinal)
            {
                char pci[32] = { 0 };
                status = get_pci(ordinal, pci, sizeof(pci));
                if (status != 0)
                    break;
                printf("%s%c%s%c", ordinal == 0 ? "" : ",", '"', pci, '"');
            }
            if (status == 0)
            {
                fputc(']', stdout);
                fflush(stdout);
                fprintf(stderr, "CUDA probe fault fixture ready pid=%lu devices=%ld\n",
                    GetCurrentProcessId(), (long)*count);
                fflush(stderr);
                // Bound the fixture's lifetime even if the test runner stops.
                Sleep(30000);
                status = 1;
            }
        }
    }

    FreeLibrary(driver);
    return status;
}
