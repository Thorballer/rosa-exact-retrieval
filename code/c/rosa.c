// Suffix-automaton next-token predictor.
//   lookup : global open-addressed hash (state,sym)->edge   O(1) even for big alphabets
//   clone  : per-state adjacency list, so cloning never scans the table
//   mode 0 : recency  -- stamp the whole suffix chain with position i (exact ROSA;
//                        O(chain length) per token => O(n^2) on repetitive input)
//   mode 1 : first-occurrence -- no stamping at all; the deepest chain state whose
//                        first end position is earlier is the match. O(1) typical,
//                        and the quadratic disappears.
//
// Returns 0 on success, 1 on allocation failure, 2 if a capacity bound would be
// exceeded (the arrays are sized for the theoretical maxima; a guard turns the
// adversarial case into an error instead of a silent overrun).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)

static inline uint32_t hs(int32_t s, int32_t sym) {
    uint32_t h = (uint32_t)s * 2654435761u ^ (uint32_t)sym * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

int build_src_c(const int32_t *x, int n, int32_t *src, int mode) {
    // states: <= 2n-1 (root included).  edges: at most 3n-4 transitions, but this
    // implementation stores a separate entry for every clone copy of a transition,
    // so it needs the larger 4n+24 allowance; the guard below covers the rest.
    int maxs = 2 * n + 2, maxe = 4 * n + 24;
    uint32_t hcap = 1;
    while ((int)hcap < 6 * n + 16) hcap <<= 1;
    uint32_t hmask = hcap - 1;

    int32_t *link = calloc(maxs, 4), *len = calloc(maxs, 4), *end = calloc(maxs, 4);
    int32_t *head = malloc(maxs * 4), *esym = malloc(maxe * 4), *eto = malloc(maxe * 4),
            *enext = malloc(maxe * 4);
    int32_t *hkst = malloc(hcap * 4), *hksym = malloc(hcap * 4), *he = malloc(hcap * 4);
    if (!link || !len || !end || !head || !esym || !eto || !enext || !hkst || !hksym || !he) {
        free(link); free(len); free(end); free(head);
        free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
        return 1;
    }
    for (int i = 0; i < maxs; i++) { head[i] = -1; link[i] = -1; end[i] = -1; len[i] = 0; }
    for (uint32_t i = 0; i < hcap; i++) he[i] = EMPTY;

    int ne = 0, last = 0, z = 1;

    for (int i = 0; i < n; i++) {
        int32_t t = x[i];
        if (z + 1 >= maxs) { goto fail; }
        int cur = z++;
        len[cur] = len[last] + 1; end[cur] = mode ? i : -1; head[cur] = -1;

        int p = last;
        while (p != -1) {
            uint32_t h = hs(p, t) & hmask, probe = 0;
            int e = EMPTY;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { e = he[h]; break; }
                h = (h + 1) & hmask;
            }
            if (probe > hcap) goto fail;            // table full
            if (e != EMPTY) break;
            // insert edge (p, t) -> cur
            if (ne + 1 > maxe) goto fail;
            esym[ne] = t; eto[ne] = cur; enext[ne] = head[p]; head[p] = ne;
            hkst[h] = p; hksym[h] = t; he[h] = ne; ne++;
            p = link[p];
        }
        if (p == -1) {
            link[cur] = 0;
        } else {
            uint32_t h = hs(p, t) & hmask, probe = 0;
            int q = -1;
            while (he[h] != EMPTY && ++probe <= hcap) {
                if (hkst[h] == p && hksym[h] == t) { q = eto[he[h]]; break; }
                h = (h + 1) & hmask;
            }
            if (q == -1) goto fail;                 // inconsistent: extension loop found it
            if (len[p] + 1 == len[q]) {
                link[cur] = q;
            } else {
                if (z + 1 >= maxs) goto fail;
                int u = z++;
                len[u] = len[p] + 1; link[u] = link[q]; end[u] = end[q]; head[u] = -1;
                for (int e2 = head[q]; e2 != -1; e2 = enext[e2]) {
                    // each cloned transition is a NEW storage entry, so the edge array needs
                    // room for (transitions + sum of clone out-degrees)
                    if (ne + 1 > maxe) goto fail;
                    esym[ne] = esym[e2]; eto[ne] = eto[e2]; enext[ne] = head[u]; head[u] = ne;
                    uint32_t hh = hs(u, esym[e2]) & hmask, pr = 0;
                    while (he[hh] != EMPTY && ++pr <= hcap) hh = (hh + 1) & hmask;
                    if (pr > hcap) goto fail;
                    hkst[hh] = u; hksym[hh] = esym[e2]; he[hh] = ne; ne++;
                }
                while (p != -1) {
                    uint32_t hh = hs(p, t) & hmask, pr = 0;
                    int hit = EMPTY;
                    while (he[hh] != EMPTY && ++pr <= hcap) {
                        if (hkst[hh] == p && hksym[hh] == t) { hit = he[hh]; break; }
                        hh = (hh + 1) & hmask;
                    }
                    if (hit == EMPTY || eto[hit] != q) break;
                    eto[hit] = u;
                    p = link[p];
                }
                link[q] = u; link[cur] = u;
            }
        }

        if (mode == 0) {
            int v = cur, found = -1;
            while (v != -1) {
                if (found < 0 && len[v] > 0 && end[v] >= 0) found = end[v] + 1;
                end[v] = i;
                v = link[v];
            }
            src[i] = found;
        } else {
            int v = cur, found = -1;
            while (v != -1) {
                if (len[v] > 0 && end[v] >= 0 && end[v] < i) { found = end[v] + 1; break; }
                v = link[v];
            }
            src[i] = found;
        }
        last = cur;
    }
    free(link); free(len); free(end); free(head);
    free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
    return 0;

fail:  // capacity guard: refuse rather than corrupt memory
    free(link); free(len); free(end); free(head);
    free(esym); free(eto); free(enext); free(hkst); free(hksym); free(he);
    return 2;
}
