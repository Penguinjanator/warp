# WASTE — embeddable MoE inference engine + CLI
#
#   make            library + cli
#   make test       the validation binaries
#   make WASTE_ENABLE_METAL=1     (accelerators are build-time options)

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
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
       src/tokenizer.c src/waste.c src/vq.c
ifneq (,$(findstring arm,$(shell uname -m)))
SRC += src/kda_neon.c
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
SRC    += src/metal.m
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

OBJ := $(SRC:.c=.o)

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

test: test_kda test_container test_forward test_tokenizer test_k3parts test_state

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
test_state: tests/test_state.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(OBJ:.o=.d) cli/main.d $(patsubst %.c,%.d,$(wildcard tests/*.c))

clean:
	rm -f $(OBJ) cli/*.o tests/*.o $(OBJ:.o=.d) cli/*.d tests/*.d libwaste.a waste \
	      test_kda test_container test_forward test_tokenizer test_k3parts test_state \
	      libwastevq.dylib libwastevq.so

check: test
	@tests/run.sh

.PHONY: all test check clean
