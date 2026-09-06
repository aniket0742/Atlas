"""Makes `tests` a real package.

Without this, pytest's default "prepend" import mode inserts each test file's
own directory into `sys.path` and imports it as a bare top-level module, never
as `tests.something`. `from tests.conftest import make_chunk` then only works by
accident, if something *else* already put the repo root on `sys.path` -- which
`python -m pytest` does (module invocation adds the current directory) but the
bare `pytest` console-script entry point does not.

That accident is exactly what happened here: every local run in this project
used `python -m pytest`, so this was never exercised, and CI's `pytest -m ...`
failed every test file that imports from `tests.conftest` with
`ModuleNotFoundError: No module named 'tests'`.

With `__init__.py` present, pytest walks up from a test file through parent
`__init__.py` files to find the package root, inserts *that* directory into
`sys.path`, and imports the file as `tests.test_whatever` -- so `tests` is a
real, importable package regardless of invocation style.
"""
