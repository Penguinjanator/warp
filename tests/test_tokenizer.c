/* test_tokenizer.c — encode/decode round-trip and id dump for the Python diff. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/tokenizer.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s container [text...]\n", argv[0]); return 2; }
    waste_tok *t = waste_tok_open(argv[1]);
    if (!t) { fprintf(stderr, "no tokenizer in %s\n", argv[1]); return 1; }
    if (argc == 2) {
        printf("vocab %d  bos %d  eos %d\n", waste_tok_vocab(t),
               waste_tok_bos(t), waste_tok_eos(t));
        return 0;
    }
    for (int a = 2; a < argc; a++) {
        int32_t ids[4096];
        const int n = waste_tok_encode(t, argv[a], ids, 4096);
        printf("%d", n);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
    }
    waste_tok_free(t);
    return 0;
}
