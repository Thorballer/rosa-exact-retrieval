// Cost of each index variant: one process per cell, getrusage for memory, clock for time.
//   ./upgrades_bench <shape> <n> <variant> <maxmis> <gate>
// shapes: random | dup | const | codefile (int32 tokens on stdin)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

int predict_variants(const int32_t *x, int n, int variant, int maxmis, int gate,
                     int32_t *pred, int32_t *mlen, int32_t *fired);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint64_t s = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 33);
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s shape n variant maxmis gate\n", argv[0]); return 2; }
    const char *shape = argv[1];
    int n = atoi(argv[2]), variant = atoi(argv[3]), maxmis = atoi(argv[4]), gate = atoi(argv[5]);

    int32_t *x = malloc(sizeof(int32_t) * (size_t)n);
    if (!strcmp(shape, "codefile")) {
        if (fread(x, sizeof(int32_t), (size_t)n, stdin) != (size_t)n) return 3;
    } else if (!strcmp(shape, "const")) {
        for (int i = 0; i < n; i++) x[i] = 7;
    } else if (!strcmp(shape, "random")) {
        for (int i = 0; i < n; i++) x[i] = (int32_t)(rnd() % 32000);
    } else if (!strcmp(shape, "dup")) {
        int32_t *base = malloc(sizeof(int32_t) * 1024);
        for (int i = 0; i < 1024; i++) base[i] = (int32_t)(rnd() % 32000);
        for (int i = 0; i < n; i++) x[i] = base[i % 1024];
        free(base);
    } else { fprintf(stderr, "unknown shape\n"); return 2; }

    int32_t *p = malloc(sizeof(int32_t) * (size_t)n), *ml = malloc(sizeof(int32_t) * (size_t)n),
            *f = malloc(sizeof(int32_t) * (size_t)n);
    double t0 = now();
    int rc = predict_variants(x, n, variant, maxmis, gate, p, ml, f);
    double dt = now() - t0;

    long fired = 0;
    for (int i = 0; i < n; i++) fired += f[i];
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("{\"shape\":\"%s\",\"n\":%d,\"variant\":%d,\"maxmis\":%d,\"gate\":%d,\"rc\":%d,"
           "\"seconds\":%.6f,\"us_per_token\":%.3f,\"fire_rate\":%.4f,\"maxrss_bytes\":%ld}\n",
           shape, n, variant, maxmis, gate, rc, dt, 1e6 * dt / n, (double)fired / n,
           (long)ru.ru_maxrss);
    return 0;
}
