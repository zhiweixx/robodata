#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdatomic.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Driver 595.71.05, SHA256 pinned by the launcher. The five-byte fork@plt
 * call at 0x81c6d returns here. No other fork caller is eligible. */
#ifndef NGX_TEST_RETURN_OFFSET
#define NGX_RETURN_OFFSET ((uintptr_t)0x81c72)
#else
#define NGX_RETURN_OFFSET ((uintptr_t)NGX_TEST_RETURN_OFFSET)
#endif

typedef pid_t (*fork_function)(void);
static _Atomic(fork_function) real_fork;
static int log_fd = -1;
static struct stat pinned_library;
static _Atomic int configured;
static pid_t owner_pid;

static fork_function resolve_real_fork(void) {
    fork_function function=atomic_load_explicit(&real_fork,memory_order_acquire);
    if (!function) {
        /* ELF can run another library's constructor first. Resolve the real
         * libc entry lazily, retaining its pthread_atfork behavior. Concurrent
         * early callers can resolve independently; no once-lock is inherited. */
        void *symbol=dlsym(RTLD_NEXT,"fork");
        if (symbol) {
            memcpy(&function,&symbol,sizeof(function));
            atomic_store_explicit(&real_fork,function,memory_order_release);
        }
    }
    return function;
}

static void record(const char *event, pid_t child, int error, int status,
                   int code, uintptr_t offset) {
    int saved = errno;
    char line[512];
    int n = snprintf(line, sizeof(line),
        "{\"event\":\"%s\",\"parent_pid\":%ld,\"child_pid\":%ld,"
        "\"errno\":%d,\"exit_status\":%d,\"exit_code_type\":%d,"
        "\"caller_offset\":%lu}\n",
        event, (long)getpid(), (long)child, error, status, code,
        (unsigned long)offset);
    if (log_fd >= 0 && n > 0 && (size_t)n < sizeof(line)) {
        ssize_t written;
        do { written = write(log_fd, line, (size_t)n); }
        while (written < 0 && errno == EINTR);
        /* The probe additionally requires the complete registered/reaped log. */
        (void)written;
    }
    errno = saved;
}

__attribute__((constructor)) static void initialize(void) {
    int saved = errno;
    owner_pid = getpid();
    fork_function function=resolve_real_fork();
    const char *log = getenv("NGX_REAPER_LOG");
    const char *library = getenv("NGX_REAPER_LIBRARY");
    if (log) log_fd = open(log, O_WRONLY|O_APPEND|O_CREAT|O_CLOEXEC|O_NOFOLLOW, 0600);
    int ready = function && log_fd >= 0 && library &&
        stat(library, &pinned_library) == 0 && S_ISREG(pinned_library.st_mode);
    record(ready ? "initialized" : "initialization_error", 0,
           ready ? 0 : (errno ? errno : EINVAL), 0, 0, NGX_RETURN_OFFSET);
    atomic_store_explicit(&configured,ready,memory_order_release);
    errno = saved;
}

struct child_identity {
    pid_t ppid;
    uid_t uid;
    unsigned long long start_ticks;
    char state;
};

static int read_identity(pid_t pid, struct child_identity *identity) {
    char path[64], buffer[4096];
    struct stat info;
    snprintf(path, sizeof(path), "/proc/%ld", (long)pid);
    if (stat(path, &info) < 0) return -1;
    identity->uid = info.st_uid;
    snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
    int fd = open(path, O_RDONLY|O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buffer, sizeof(buffer)-1);
    int saved = errno;
    close(fd);
    errno = saved;
    if (n <= 0) return -1;
    buffer[n] = 0;
    char *tail = strrchr(buffer, ')');
    if (!tail || tail[1] != ' ') { errno = EINVAL; return -1; }
    char *context = NULL;
    char *token = strtok_r(tail+2, " ", &context);
    for (int field=0; field<=19; ++field) {
        if (!token) { errno = EINVAL; return -1; }
        if (field == 0) identity->state = token[0];
        if (field == 1) identity->ppid = (pid_t)strtol(token, NULL, 10);
        if (field == 19) identity->start_ticks = strtoull(token, NULL, 10);
        token = strtok_r(NULL, " ", &context);
    }
    return 0;
}

static int same_identity(const struct child_identity *before,
                         const struct child_identity *after) {
    return before->ppid == owner_pid && after->ppid == owner_pid &&
        before->uid == getuid() && after->uid == getuid() &&
        before->start_ticks == after->start_ticks;
}

static double monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) < 0) return -1;
    return (double)now.tv_sec + (double)now.tv_nsec/1000000000.0;
}

static void collect_before_return(pid_t pid, uintptr_t offset) {
    struct child_identity original, current;
    if (read_identity(pid, &original) < 0 || original.ppid != owner_pid || original.uid != getuid()) {
        record("identity_error", pid, errno ? errno : ECHILD, 0, 0, offset);
        return;
    }
    record("registered", pid, 0, 0, 0, offset);
    double started = monotonic_seconds();
    if (started < 0) { record("clock_error", pid, errno, 0, 0, offset); return; }
    for (;;) {
        siginfo_t info;
        memset(&info, 0, sizeof(info));
        if (waitid(P_PID, (id_t)pid, &info, WEXITED|WNOHANG|WNOWAIT) < 0) {
            if (errno == EINTR) continue;
            record("observe_error", pid, errno, 0, 0, offset);
            return;
        }
        if (info.si_pid) {
            /* Only the traced missing-updater failure is ours to consume.
             * Any changed behavior remains available to the original caller. */
            if (info.si_code != CLD_EXITED || info.si_status != 1) {
                record("unexpected_exit_error", pid, EPROTO, info.si_status, info.si_code, offset);
                return;
            }
            if (info.si_pid != pid || read_identity(pid, &current) < 0 ||
                !same_identity(&original, &current) || current.state != 'Z') {
                record("consume_identity_error", pid, errno ? errno : EINVAL, 0, 0, offset);
                return;
            }
            int status = 0;
            pid_t waited;
            do { waited = waitpid(pid, &status, WNOHANG); } while (waited < 0 && errno == EINTR);
            if (waited != pid) {
                record("consume_error", pid, waited < 0 ? errno : EAGAIN, 0, 0, offset);
            } else {
                int exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status);
                int exit_type = WIFEXITED(status) ? CLD_EXITED : CLD_KILLED;
                record("reaped", pid, 0, exit_status, exit_type, offset);
            }
            return;
        }
        double now = monotonic_seconds();
        if (now < 0 || now-started >= 2.0) {
            record(now < 0 ? "clock_error" : "timeout_error", pid,
                   now < 0 ? errno : ETIMEDOUT, 0, 0, offset);
            return;
        }
        struct timespec delay = { .tv_sec=0, .tv_nsec=5000000 };
        while (nanosleep(&delay, &delay) < 0 && errno == EINTR) {}
    }
}

pid_t fork(void) {
    /* All inspection is before real_fork. Child returns immediately, touching
     * neither libc locks, logging, dladdr, nor thread registration. */
    int input_errno = errno;
    fork_function function=resolve_real_fork();
    if (!function) { errno=ENOSYS; return -1; }
    if (!atomic_load_explicit(&configured,memory_order_acquire)) {
        errno=input_errno;
        return function();
    }
    void *caller = __builtin_extract_return_addr(__builtin_return_address(0));
    Dl_info origin;
    struct stat actual;
    uintptr_t offset = 0;
    int targeted = 0;
    if (getpid() == owner_pid && dladdr(caller, &origin) && origin.dli_fbase) {
        offset = (uintptr_t)caller - (uintptr_t)origin.dli_fbase;
        targeted = offset == NGX_RETURN_OFFSET && origin.dli_fname &&
            stat(origin.dli_fname, &actual) == 0 &&
            actual.st_dev == pinned_library.st_dev && actual.st_ino == pinned_library.st_ino;
    }
    if (targeted) {
        struct sigaction disposition;
        if (sigaction(SIGCHLD, NULL, &disposition) < 0 ||
            disposition.sa_handler != SIG_DFL || (disposition.sa_flags & SA_NOCLDWAIT)) {
            record("sigchld_policy_error", 0, EBUSY, 0, 0, offset);
            targeted = 0;
        } else {
            record("sigchld_default", 0, 0, 0, 0, offset);
        }
    }
    errno = input_errno;
    pid_t pid = function();
    int fork_errno = errno;
    if (pid <= 0 || !targeted) { errno = fork_errno; return pid; }

    /* The missing-exec child writes just four errno bytes, then exits. Its
     * original parent cannot access this PID until this function returns. The
     * inspected NGX parent branch never waits and reads those same pipe bytes.
     * Ordinary children and other NGX callsites never enter this path. */
    collect_before_return(pid, offset);
    errno = fork_errno;
    return pid;
}
