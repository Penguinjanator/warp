# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
#
# This file exists to make `tests` a regular package, and that is its whole
# job. Without it the directory is only an implicit namespace portion, and
# the import system prefers a *regular* package found later on sys.path —
# so any unrelated `tests` package installed in site-packages shadows this
# one and `python3 -m unittest discover -s tests/serve -t .` fails to
# import tests.serve at all. That is not a hypothetical: it happened, and
# the whole serve suite reported as one failure with no hint that the
# module it could not find was ours.
