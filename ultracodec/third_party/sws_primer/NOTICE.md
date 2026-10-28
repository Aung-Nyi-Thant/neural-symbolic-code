# Tier 4 primer sources

`src/sws_primer.inc` holds 256 KiB that the Tier 4 model reads before block 0 when
`EngineOptions::tier4_primer` is set (the default). The bytes are frozen: primed archives need
these exact bytes to decode, and the header's `model_param` holds their CRC-32C. The bytes are the
same as the HCE Ultra v4 primer. `make_primer.py` rebuilds them.

| Bytes | Source | Licence |
|---|---|---|
| 112 KiB | CPython 3.11 `Lib/`: textwrap, string, shlex, queue, heapq, pprint, tempfile, shutil, pathlib, logging, urllib.parse, threading, subprocess, zipfile, inspect | PSF License Agreement, copy in `PYTHON-LICENSE.txt` |
| 56 KiB | CPython 3.11 `Include/` headers (object.h, abstract.h, listobject.h, ...) | PSF License Agreement |
| 24 KiB | JSON, XML, HTML, CSS, YAML, CSV and Markdown samples written by `structured.py` | this project |
| 64 KiB | Licence texts as shipped in Debian `/usr/share/common-licenses`: Apache-2.0, MPL-2.0, GPL-3 | each text permits verbatim copying |

The Python files come from the Ubuntu 24.04 `libpython3.11` packages and the licence texts from its `base-files` package. The primer
excludes the files used by the benchmark corpus (`json`, `ast`, `typing`, `functools`, and the
first 4 MiB of the sorted standard library), so primed benchmark results are not inflated by
overlap.
