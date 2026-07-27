/* version.c — what the linked library actually is. */

#include "waste.h"
#include "waste_backend.h"
#include "waste_format.h"

#include <stdio.h>

const char *waste_version(void) { return WASTE_VERSION_STRING; }
int waste_version_number(void) { return WASTE_VERSION_NUMBER; }

const char *waste_build_info(void)
{
    static char buf[192];
    static int done = 0;
    if (!done) {
        waste_backend_init(WASTE_BE_AUTO);
        snprintf(buf, sizeof buf,
                 "WASTE %s (container v%d, backend %s, %s)",
                 WASTE_VERSION_STRING, WASTE_FORMAT_VERSION,
                 waste_backend_name(),
#if defined(__aarch64__)
                 "arm64"
#elif defined(__x86_64__)
                 "x86_64"
#else
                 "generic"
#endif
        );
        done = 1;
    }
    return buf;
}
