# WASTE — embeddable MoE inference engine + CLI
#
#   make            library + cli
#   make test       the validation binaries
#   make WASTE_ENABLE_METAL=1     (accelerators are build-time options)

CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  := -lm -lpthread
VQ_SUPER ?= 2
CFLAGS  += -DVQ_SUPER=$(VQ_SUPER)
# Track header dependencies. Without this a changed struct in a header
# leaves stale objects compiled against the old layout — which links fine
# and then corrupts memory at run time.
CFLAGS  += -MMD -MP

SRC := src/model.c src/kda.c src/backend.c src/ecache.c src/version.c \
       src/tokenizer.c src/waste.c
ifneq (,$(findstring arm,$(shell uname -m)))
SRC += src/kda_neon.c
endif
OBJ := $(SRC:.c=.o)

ifdef WASTE_ENABLE_METAL
CFLAGS += -DWASTE_ENABLE_METAL=1
LDLIBS += -framework Metal -framework Foundation
endif
ifdef WASTE_ENABLE_CUDA
CFLAGS += -DWASTE_ENABLE_CUDA=1
LDLIBS += -lcudart
endif

all: waste libwaste.a

libwaste.a: $(OBJ)
	ar rcs $@ $^

waste: cli/main.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: test_kda test_container test_forward test_tokenizer test_k3parts

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

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(OBJ:.o=.d) cli/main.d $(patsubst %.c,%.d,$(wildcard tests/*.c))

clean:
	rm -f $(OBJ) cli/*.o tests/*.o $(OBJ:.o=.d) cli/*.d tests/*.d libwaste.a waste \
	      test_kda test_container test_forward test_tokenizer test_k3parts

.PHONY: all test clean
