#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct ak_context ak_context;
typedef int32_t (*get_count_fn)(int32_t *);
typedef int32_t (*get_pci_fn)(int32_t, char *, int32_t);
typedef int32_t (*create_fn)(uint32_t, int32_t, ak_context **);
typedef int32_t (*alloc_fn)(ak_context *, size_t, uint64_t *);
typedef int32_t (*wipe_fn)(ak_context *, uint64_t, size_t);

__attribute__((visibility("default"))) int32_t ak_get_device_count(int32_t *count)
{
    const char *marker = getenv("AIKERNEL_TEST_PROBE_MARKER");
    const char *driver_path = getenv("AIKERNEL_TEST_REAL_CUDA_LIBRARY");
    if (!marker || !driver_path)
        return 1;

    int fd = open(marker, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd >= 0)
    {
        close(fd);
        _exit(73);
    }
    if (errno != EEXIST)
        return 1;

    void *driver = dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    if (!driver)
        return 2;

    get_count_fn get_count = (get_count_fn)dlsym(driver, "ak_get_device_count");
    get_pci_fn get_pci = (get_pci_fn)dlsym(driver, "ak_get_device_pci_bus_id");
    create_fn create = (create_fn)dlsym(driver, "ak_create");
    alloc_fn alloc = (alloc_fn)dlsym(driver, "ak_device_alloc");
    wipe_fn wipe = (wipe_fn)dlsym(driver, "ak_wipe_device");
    if (!get_count || !get_pci || !create || !alloc || !wipe)
        return 5;
    if (get_count(count) != 0 || *count != 2)
        return 3;

    const size_t bytes = 64 * 1024 * 1024;
    for (int32_t ordinal = 0; ordinal < *count; ordinal++)
    {
        char pci[32] = { 0 };
        ak_context *context = NULL;
        uint64_t address = 0;
        if (get_pci(ordinal, pci, sizeof(pci)) != 0 ||
            create(1, ordinal, &context) != 0 ||
            alloc(context, bytes, &address) != 0 ||
            wipe(context, address, bytes) != 0)
            return 4;
        fprintf(stderr, "CUDA retry allocated %zu B on %s\n", bytes, pci);
    }
    fprintf(stderr, "CUDA retry ready pid=%ld\n", (long)getpid());
    fflush(stderr);

    // Keep the contexts alive while the test terminates the host.
    sleep(30);
    _exit(74);
}
