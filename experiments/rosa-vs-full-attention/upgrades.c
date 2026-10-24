// Candidate engine on top of k-gram hash chains, for testing recall upgrades to the
// exact suffix index. One call per sequence; reports pred/mlen/fired per position.
//
//   variant 0  longest match, earliest occurrence      (what ROSA mode 1 does)
//   variant 1  longest match, most recent occurrence   (what ROSA mode 0 does)
//   variant 2  majority successor among near-longest candidates
//   variant 3  majority with support >= 2, else longest
//   variant 4  weighted vote over all candidates (weight = match length)
//   variant 5  seed-and-extend: anchor on 4-grams ending 0..3 tokens before the query, so an
//              edit sitting immediately before the query does not hide the source; matches
//              tolerate `maxmis` substitutions
//
//   maxmis     mismatches tolerated while extending a match (0 = exact)
//   gate       abstain when the chosen match is shorter than this (0 = never)
//
// Matches shorter than 4 tokens are not represented here: the chains anchor on 2, 4 and 8
// grams, so this engine abstains where the suffix automaton still fires. That is deliberate
// -- the point of this file is the variants, and the SAM remains the reference for short
// matches (see rosa_speed.py and recall.py).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)
#define CAP 48          // candidates kept per chain
#define MAXMIS 8
#define MATCHCAP 8192

static inline uint32_t mix(uint32_t h, int32_t v) {
    h ^= (uint32_t)v * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

typedef struct {
    int32_t *head;      // hash slot -> most recent position
    int32_t *prev;      // position -> previous position with the same gram
    uint32_t hcap, hmask;
} Chain;

static int chain_init(Chain *c, int n) {
    c->hcap = 1;
    while (c->hcap < (uint32_t)(4 * n + 16)) c->hcap <<= 1;
    c->hmask = c->hcap - 1;
    c->head = malloc(sizeof(int32_t) * c->hcap);
    c->prev = malloc(sizeof(int32_t) * (size_t)n);
    if (!c->head || !c->prev) return 1;
    for (uint32_t i = 0; i < c->hcap; i++) c->head[i] = EMPTY;
    for (int i = 0; i < n; i++) c->prev[i] = EMPTY;
    return 0;
}

static void chain_free(Chain *c) { free(c->head); free(c->prev); }

// returns the bucket index this position hashed to, inserting it
static inline uint32_t chain_push(Chain *c, const int32_t *x, int i, int k) {
    uint32_t h = 2166136261u;
    for (int t = 0; t < k; t++) h = mix(h, x[i - t]);
    h &= c->hmask;
    c->prev[i] = c->head[h];
    c->head[h] = i;
    return h;
}

// seed at (a,j) with `gap` tokens between the anchor and the query: compare backwards from the
// anchor and forwards across the gap, tolerating maxmis substitutions either way.
static int extend_seed(const int32_t *x, int a, int j, int gap, int maxmis, int *mis_out) {
    int mis = 0, matched = 0;
    for (int t = 1; t <= gap; t++) {
        if (x[a + t] == x[j + t]) matched++;
        else if (mis < maxmis) mis++;
        else { *mis_out = mis; return matched; }
    }
    for (int m = 0; m < MATCHCAP; m++) {
        int p1 = a - m, p2 = j - m;
        if (p1 < 0 || p2 < 0) break;
        if (x[p1] == x[p2]) matched++;
        else if (mis < maxmis) mis++;
        else break;
    }
    *mis_out = mis;
    return matched;
}

static int extend(const int32_t *x, int i, int j, int maxmis, int *mis_out) {
    int m = 0, mis = 0;
    while (m < MATCHCAP && i - m >= 0 && j - m >= 0) {
        if (x[i - m] == x[j - m]) m++;
        else if (mis < maxmis) { mis++; m++; }
        else break;
    }
    *mis_out = mis;
    return m;
}

int predict_variants(const int32_t *x, int n, int variant, int maxmis, int gate,
                     int32_t *pred, int32_t *mlen, int32_t *fired) {
    Chain c2, c4, c8;
    if (chain_init(&c2, n) || chain_init(&c4, n) || chain_init(&c8, n)) return 1;

    int32_t cand[3 * CAP], candm[3 * CAP];
    int ncand;

    for (int i = 0; i < n; i++) {
        chain_push(&c2, x, i, 2);
        if (i >= 3) chain_push(&c4, x, i, 4);
        if (i >= 7) chain_push(&c8, x, i, 8);

        ncand = 0;
        Chain *cs[3] = {&c8, &c4, &c2};
        int ks[3] = {8, 4, 2};
        for (int ci = 0; ci < 3 && ncand < 3 * CAP; ci++) {
            int k = ks[ci];
            if (i - k + 1 < 0) continue;
            uint32_t h = 2166136261u;
            for (int t = 0; t < k; t++) h = mix(h, x[i - t]);
            int j = cs[ci]->head[h & cs[ci]->hmask];
            int kept = 0;
            while (j != EMPTY && kept < CAP) {
                if (j < i) {
                    int mis;
                    int m = extend(x, i, j, maxmis, &mis);
                    cand[ncand] = j;
                    candm[ncand] = m - 3 * mis;      // mismatches cost, they do not count
                    ncand++;
                    kept++;
                }
                j = cs[ci]->prev[j];
            }
        }

        int32_t p = -1, best = -1;
        if (variant == 5) {
            int32_t bestscore = -1, bj = -1, bmatch = 0, bo = 0;
            for (int o = 0; o <= 3; o++) {
                int a = i - o;
                if (a - 3 < 0) continue;
                uint32_t h = 2166136261u;
                for (int t = 0; t < 4; t++) h = mix(h, x[a - t]);
                h &= c4.hmask;
                int j = c4.head[h];
                int kept = 0;
                while (j != EMPTY && kept < CAP) {
                    if (j < a) {
                        int mis;
                        int m = extend_seed(x, a, j, o, maxmis, &mis);
                        int32_t score = m - 3 * mis;
                        if (score > bestscore || (score == bestscore && j < bj)) {
                            bestscore = score; bj = j; bmatch = m; bo = o;
                        }
                        kept++;
                    }
                    j = c4.prev[j];
                }
            }
            if (bj >= 0) { p = x[bj + bo + 1]; best = bmatch; }
        } else if (ncand > 0) {
            if (variant == 4) {                       // weighted vote, weight = match length
                int32_t tot[3 * CAP];
                for (int a = 0; a < ncand; a++) {
                    int32_t succ = x[cand[a] + 1];
                    tot[a] = 0;
                    for (int b = 0; b < ncand; b++)
                        if (x[cand[b] + 1] == succ) tot[a] += candm[b] > 0 ? candm[b] : 0;
                }
                int32_t bestt = -1, bestm2 = -1, bj = -1;
                for (int a = 0; a < ncand; a++) {
                    if (tot[a] > bestt ||
                        (tot[a] == bestt && candm[a] > bestm2) ||
                        (tot[a] == bestt && candm[a] == bestm2 && cand[a] > bj)) {
                        bestt = tot[a]; bestm2 = candm[a]; bj = cand[a];
                    }
                }
                if (bj >= 0) { p = x[bj + 1]; best = bestm2; }
            } else {
                int32_t mmax = -1;
                for (int a = 0; a < ncand; a++) if (candm[a] > mmax) mmax = candm[a];
                if (variant == 0 || variant == 1) {
                    int32_t bj = -1;
                    for (int a = 0; a < ncand; a++) {
                        if (candm[a] != mmax) continue;
                        if (bj < 0 || (variant == 0 ? cand[a] < bj : cand[a] > bj)) bj = cand[a];
                    }
                    if (bj >= 0) { p = x[bj + 1]; best = mmax; }
                } else {                          // majority among near-longest
                    int32_t thresh = mmax - 1;
                    int bestcount = 0, bestm = -1; int32_t bj = -1;
                    for (int a = 0; a < ncand; a++) {
                        if (candm[a] < thresh) continue;
                        int32_t succ = x[cand[a] + 1];
                        int cnt = 0;
                        for (int b = 0; b < ncand; b++)
                            if (candm[b] >= thresh && x[cand[b] + 1] == succ) cnt++;
                        if (cnt > bestcount || (cnt == bestcount && candm[a] > bestm) ||
                            (cnt == bestcount && candm[a] == bestm && cand[a] > bj)) {
                            bestcount = cnt; bestm = candm[a]; bj = cand[a]; p = succ;
                        }
                    }
                    if (variant == 3 && bestcount < 2) {   // no support: fall back to longest
                        int32_t bj2 = -1;
                        for (int a = 0; a < ncand; a++)
                            if (candm[a] == mmax && (bj2 < 0 || cand[a] < bj2)) bj2 = cand[a];
                        if (bj2 >= 0) { p = x[bj2 + 1]; bestm = mmax; }
                    }
                    best = bestm;
                }
            }
        }

        int m_best = best < 0 ? 0 : best;
        if (gate > 0 && m_best < gate) { p = -1; }
        pred[i] = p;
        mlen[i] = m_best;
        fired[i] = (p >= 0) ? 1 : 0;
    }
    chain_free(&c2); chain_free(&c4); chain_free(&c8);
    return 0;
}
