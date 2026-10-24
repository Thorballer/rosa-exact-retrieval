// One benchmark cell per process, so getrusage's high-water mark belongs to that cell alone.
//   ./rosa_bench <shape> <n> <mode>
// shapes: random | dup | const | codefile
// prints one JSON line: build time, ns/token, fire rate, peak RSS, and the exact
// allocation the builder requested (bytes/token), so RSS can be cross-checked.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

int build_src_c(const int32_t *x, int n, int32_t *src, int mode);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {  // xorshift64*
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (uint32_t)((s * 0x2545F4914F6CDD1Dull) >> 33);
}

// Total bytes build_src_c asks calloc/malloc for, mirroring its size formulas.
static double exact_alloc_bytes(int n) {
    double maxs = 2.0 * n + 2, maxe = 4.0 * n + 24;
    double hcap = 1;
    while (hcap < 6.0 * n + 16) hcap *= 2;
    return 4.0 * maxs * 4      // link, len, end, head
         + 3.0 * maxe * 4      // esym, eto, enext
         + 3.0 * hcap * 4;     // hkst, hksym, he
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s shape n mode\n", argv[0]); return 2; }
    const char *shape = argv[1];
    int n = atoi(argv[2]), mode = atoi(argv[3]);

    int32_t *x = malloc(sizeof(int32_t) * (size_t)n);
    if (!x) return 1;

    if (!strcmp(shape, "codefile")) {  // token ids piped in from stdin
        if (fread(x, sizeof(int32_t), (size_t)n, stdin) != (size_t)n) return 3;
    } else if (!strcmp(shape, "const")) {
        for (int i = 0; i < n; i++) x[i] = 7;
    } else if (!strcmp(shape, "random")) {
        for (int i = 0; i < n; i++) x[i] = (int32_t)(rnd() % 32000);
    } else if (!strcmp(shape, "dup")) {  // 512-token span, 512-token filler, span again
        int32_t *base = malloc(sizeof(int32_t) * 1024);
        for (int i = 0; i < 1024; i++) base[i] = (int32_t)(rnd() % 32000);
        for (int i = 0; i < n; i++) x[i] = base[i % 1024];
        free(base);
    } else { fprintf(stderr, "unknown shape\n"); return 2; }

    int32_t *src = malloc(sizeof(int32_t) * (size_t)n);
    double t0 = now();
    int rc = build_src_c(x, n, src, mode);
    double dt = now() - t0;

    long fired = 0;
    for (int i = 0; i < n; i++) if (src[i] >= 0) fired++;

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("{\"shape\":\"%s\",\"n\":%d,\"mode\":%d,\"rc\":%d,\"seconds\":%.6f,"
           "\"ns_per_token\":%.1f,\"fire_rate\":%.4f,\"maxrss_bytes\":%ld,"
           "\"requested_alloc_bytes\":%.0f,\"requested_bytes_per_token\":%.1f}\n",
           shape, n, mode, rc, dt, 1e9 * dt / n, (double)fired / n,
           (long)ru.ru_maxrss, exact_alloc_bytes(n), exact_alloc_bytes(n) / n);
    return 0;
}
