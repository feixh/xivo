/* A minimal SIGPROF sampling profiler, LD_PRELOADed into pyxivo.
 *
 * Why not perf or gperftools: /proc/sys/kernel/perf_event_paranoid is 4 on this
 * box (no perf_event_open) , /proc/sys/kernel/yama/ptrace_scope is 1 (gdb cannot
 * attach to a non-descendant), and libprofiler is not installed. This is the
 * same mechanism gperftools' CPU profiler uses: ITIMER_PROF at a fixed rate,
 * backtrace() in the handler, raw return addresses buffered and resolved after
 * the fact.
 *
 *   XIVO_SAMPLER_OUT=/tmp/prof.raw XIVO_SAMPLER_HZ=500 \
 *   LD_PRELOAD=.../sampler.so python3 scripts/pyxivo.py ...
 *
 * Output is a text file: one line per sample, hex return addresses innermost
 * first, followed by the process's /proc/self/maps. resolve.py turns that into a
 * flat and a callers profile.
 */
#define _GNU_SOURCE
#include <execinfo.h>
#include <stdint.h>
#include <ucontext.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_DEPTH 48
#define MAX_SAMPLES 4000000

static void *g_buf;          /* MAX_SAMPLES * MAX_DEPTH void* */
static int *g_len;           /* depth of each sample */
static volatile long g_n;
static char g_out[512];

static int g_depth = MAX_DEPTH;

/* Depth 1: take the interrupted program counter straight out of the signal's
   ucontext. That is async-signal-safe and cannot fail, whereas backtrace()
   walks the stack with libgcc's unwinder and will segfault if it is interrupted
   inside a frame it cannot unwind -- which it duly did in OpenCV's KLT inner
   loop. Use depth 1 for a flat profile you can trust, and a larger depth (at the
   risk of a crash near the end of the run, after most samples are already in
   the buffer) when you need callers. */
static void handler(int sig, siginfo_t *info, void *uc) {
  (void)sig;
  (void)info;
  long i = g_n;
  if (i >= MAX_SAMPLES) return;
  void **slot = ((void **)g_buf) + (size_t)i * MAX_DEPTH;
  int n;
  if (g_depth <= 1) {
    slot[0] = (void *)(uintptr_t)((ucontext_t *)uc)->uc_mcontext.gregs[REG_RIP];
    n = 1;
  } else {
    n = backtrace(slot, g_depth);
  }
  g_len[i] = n;
  g_n = i + 1;
}

__attribute__((constructor)) static void sampler_start(void) {
  const char *out = getenv("XIVO_SAMPLER_OUT");
  if (!out) return;
  /* One file per pid: LD_PRELOAD also lands in the taskset/setarch wrappers and
     in anything the run forks, and their (empty) dumps would otherwise truncate
     the real one. */
  snprintf(g_out, sizeof g_out, "%s.%d", out, (int)getpid());
  g_buf = malloc((size_t)MAX_SAMPLES * MAX_DEPTH * sizeof(void *));
  g_len = malloc((size_t)MAX_SAMPLES * sizeof(int));
  if (!g_buf || !g_len) { g_out[0] = 0; return; }
  /* Touch nothing: the pages fault in as samples arrive, so the profiler's own
     RSS tracks the sample count rather than the reservation. */
  int hz = 500;
  const char *h = getenv("XIVO_SAMPLER_HZ");
  if (h) hz = atoi(h);
  if (hz <= 0) hz = 500;
  const char *d = getenv("XIVO_SAMPLER_DEPTH");
  if (d) g_depth = atoi(d);
  if (g_depth < 1) g_depth = 1;
  if (g_depth > MAX_DEPTH) g_depth = MAX_DEPTH;

  /* Warm up backtrace()'s lazy unwinder initialisation outside the handler. */
  void *tmp[8];
  backtrace(tmp, 8);

  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGPROF, &sa, NULL);

  struct itimerval it;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = 1000000 / hz;
  it.it_value = it.it_interval;
  setitimer(ITIMER_PROF, &it, NULL);
}

__attribute__((destructor)) static void sampler_stop(void) {
  if (!g_out[0]) return;
  struct itimerval it;
  memset(&it, 0, sizeof it);
  setitimer(ITIMER_PROF, &it, NULL);
  signal(SIGPROF, SIG_IGN);

  long n = g_n;
  if (n == 0) return;
  FILE *f = fopen(g_out, "w");
  if (!f) return;
  for (long i = 0; i < n; i++) {
    void **slot = ((void **)g_buf) + (size_t)i * MAX_DEPTH;
    for (int k = 0; k < g_len[i]; k++) fprintf(f, "%p ", slot[k]);
    fputc('\n', f);
  }
  fputs("=== MAPS\n", f);
  FILE *m = fopen("/proc/self/maps", "r");
  if (m) {
    char line[1024];
    while (fgets(line, sizeof line, m)) fputs(line, f);
    fclose(m);
  }
  fclose(f);
  fprintf(stderr, "[sampler] %ld samples -> %s\n", n, g_out);
}
