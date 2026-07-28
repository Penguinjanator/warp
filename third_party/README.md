# Vendored third-party code

## stb_image.h (v2.30)

Public domain / MIT — see the licence block at the end of the file.
Source: https://github.com/nothings/stb

The engine has no runtime dependencies, and this is the one exception:
decoding PNG and JPEG by hand is a project of its own, and stb_image is a
single header with no build system, no configuration and no transitive
dependencies. It is compiled into `src/image.c` and nothing else includes
it. `STBI_NO_STDIO` is *not* set — the CLI reads files by path — but the
formats are restricted to the ones an image prompt actually needs.

Vendored rather than fetched so the build stays offline and reproducible.
