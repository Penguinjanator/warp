# WASTE — embeddable MoE inference engine + CLI
#
#   make            library + cli
#   make test       the validation binaries
#   make WASTE_ENABLE_METAL=1     (accelerators are build-time options)

# Explicit, because per-object rules below would otherwise make the first
# of them the default goal.
.DEFAULT_GOAL := all

CC      ?= cc
# gnu11, not c11: with -std=c11 glibc sets __STRICT_ANSI__ and hides every
# POSIX extension, so pread, fcntl, posix_memalign and pthread_* would all
# be implicitly declared on Linux. Only model.c defines _GNU_SOURCE itself.
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra
LDLIBS  := -lm -lpthread
# Shared-library suffix: convert.py already looks for both through ctypes.
ifeq ($(shell uname -s),Darwin)
SOEXT   := dylib
else
SOEXT   := so
endif
VQ_SUPER ?= 2
CFLAGS  += -DVQ_SUPER=$(VQ_SUPER)
# Track header dependencies. Without this a changed struct in a header
# leaves stale objects compiled against the old layout — which links fine
# and then corrupts memory at run time.
CFLAGS  += -MMD -MP

SRC := src/model.c src/kda.c src/backend.c src/ecache.c src/version.c \
       src/tokenizer.c src/waste.c src/vq.c src/vision.c src/image.c
# Match what backend.c tests for. Linux/aarch64 reports "aarch64", which
# does not contain "arm" — the old findstring left kda_neon.c out of the
# build while backend.c still emitted the call to it, so the link failed
# with an undefined waste_kda_register_neon.
UNAME_M := $(shell uname -m)
ifneq (,$(filter arm% aarch64%,$(UNAME_M)))
SRC += src/kda_neon.c
endif

# One translation unit per x86 ISA, each built with its own flags so the
# baseline binary stays runnable on a CPU that has neither. waste_backend_init
# picks between them from CPUID, so a single binary adapts at run time —
# which is the whole reason these are separate files rather than #ifdefs
# inside model.c.
ifneq (,$(filter x86_64% amd64%,$(UNAME_M)))
X86SRC  := src/simd_avx2.c src/simd_avx512.c
SRC     += $(X86SRC)
endif

# WASTE_NATIVE=1 builds for this exact CPU, which on ARMv8.6 turns on the
# SMMLA batched matmul (still opt-in at runtime with WASTE_I8MM=1 — it
# quantizes activations to int8, so it does not produce the f32 numbers).
# The default build stays portable across ARM.
ifdef WASTE_NATIVE
CFLAGS += -mcpu=native
endif
# Accelerator backends are build-time options, and each needs a source file
# that registers it. None exists yet: the flags below were reachable but
# only produced "Undefined symbols: _waste_register_metal" at link time.
# Fail early and say why instead. Deleting a check is the last step of
# adding the backend it guards.
ifdef WASTE_ENABLE_METAL
ifeq (,$(wildcard src/metal.m))
$(error WASTE_ENABLE_METAL=1, but src/metal.m does not exist — the Metal \
backend is not implemented. Build without the flag: CPU+NEON is the only \
backend this engine has)
endif
CFLAGS += -DWASTE_ENABLE_METAL=1
OBJCSRC := src/metal.m
LDLIBS += -framework Metal -framework Foundation
endif
ifdef WASTE_ENABLE_CUDA
ifeq (,$(wildcard src/cuda.cu))
$(error WASTE_ENABLE_CUDA=1, but src/cuda.cu does not exist — the CUDA \
backend is not implemented. Build without the flag: CPU+NEON is the only \
backend this engine has)
endif
CFLAGS += -DWASTE_ENABLE_CUDA=1
SRC    += src/cuda.cu
LDLIBS += -lcudart
endif
ifdef WASTE_ENABLE_BLAS
ifeq (,$(wildcard src/blas.c))
$(error WASTE_ENABLE_BLAS=1, but src/blas.c does not exist — the BLAS \
backend is not implemented. Build without the flag: CPU+NEON is the only \
backend this engine has)
endif
CFLAGS += -DWASTE_ENABLE_BLAS=1
SRC    += src/blas.c
LDLIBS += -lblas
endif

OBJ := $(SRC:.c=.o) $(OBJCSRC:.m=.o)

src/metal.o: src/metal.m
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

src/simd_avx2.o:   CFLAGS += -mavx2 -mfma
src/simd_avx512.o: CFLAGS += -mavx512f -mavx512bw

all: waste libwaste.a libwastevq.$(SOEXT)

# `make` builds the shipped artifacts; `make test` also builds the checkers.
# They are separate targets, so remember which one you need — testing a
# stale test binary costs more time than rebuilding it.

# shared object so tools/convert.py can call the encoder through ctypes
libwastevq.$(SOEXT): src/vq.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< -lm -lpthread

libwaste.a: $(OBJ)
	ar rcs $@ $^

waste: cli/main.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: test_kda test_container test_forward test_tokenizer test_k3parts test_state test_vision test_image

test_kda: tests/test_kda.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_container: tests/test_container.o
	$(CC) $(CFLAGS) -o $@ $^
test_forward: tests/test_forward.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_tokenizer: tests/test_tokenizer.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_k3parts: tests/test_k3parts.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_image: tests/test_image.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vision: tests/test_vision.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_state: tests/test_state.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(OBJ:.o=.d) cli/main.d $(patsubst %.c,%.d,$(wildcard tests/*.c))

clean:
	rm -f $(OBJ) cli/*.o tests/*.o $(OBJ:.o=.d) cli/*.d tests/*.d libwaste.a waste \
	      test_kda test_container test_forward test_tokenizer test_k3parts test_state \
	      test_image \
	      libwastevq.dylib libwastevq.so

check: test
	@tests/run.sh

# Sanitizers. Separate targets rather than a flag on `make`, because they
# need a full rebuild: mixing instrumented and uninstrumented objects
# produces false reports and missed ones. ASan's own allocator refuses
# very large mappings, so these run against a synthetic container, which
# is what tests/run.sh falls back to when given a path that does not exist.
SAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer \
             -fno-sanitize-recover=all -g -O1

asan:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory all test \
	    CFLAGS="-std=gnu11 -Wall -Wextra -DVQ_SUPER=$(VQ_SUPER) -MMD -MP $(SAN_FLAGS)" \
	    LDLIBS="-lm -lpthread $(SAN_FLAGS)"
	@rc=0; ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
	    WASTE_SANITIZED=1 \
	    tests/run.sh /nonexistent-container-use-synthetic || rc=$$?; \
	 $(MAKE) --no-print-directory clean; exit $$rc

fuzz: test
	@python3 tools/fuzz_container.py --runs $(FUZZ_RUNS)

FUZZ_RUNS ?= 200

# What CI runs: the fuzzer against an instrumented build, so a read past
# the end of a buffer aborts instead of returning plausible garbage.
fuzz-asan:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory all test \
	    CFLAGS="-std=gnu11 -Wall -Wextra -DVQ_SUPER=$(VQ_SUPER) -MMD -MP $(SAN_FLAGS)" \
	    LDLIBS="-lm -lpthread $(SAN_FLAGS)"
	@rc=0; ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
	    python3 tools/fuzz_container.py --runs $(FUZZ_RUNS) || rc=$$?; \
	 $(MAKE) --no-print-directory clean; exit $$rc

.PHONY: all test check clean asan fuzz fuzz-asan

.PHONY: all test check clean
