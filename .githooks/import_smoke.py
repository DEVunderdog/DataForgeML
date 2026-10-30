"""Import every module under the Operations and splitting packages.

Run by the pre-commit hook. A module that fails to import is reported by
name with the error that broke it, and the script exits non-zero.
"""

import importlib
import pkgutil
import sys
import traceback
from collections.abc import Callable, Iterator

PACKAGES = ("dataforge_ml.operations", "dataforge_ml.splitting")


def _module_names(package_name: str) -> Iterator[str]:
    package = importlib.import_module(package_name)
    yield package_name
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        yield info.name


def _attempt(name: str, action: Callable[[str], object], failures: list) -> object:
    try:
        return action(name)
    except Exception as exc:  # noqa: BLE001 - any import-time error fails the smoke test
        failures.append((name, exc))
        return None


def main() -> int:
    failures: list[tuple[str, Exception]] = []
    for package_name in PACKAGES:
        names = _attempt(package_name, lambda n: list(_module_names(n)), failures)
        for name in names or []:
            _attempt(name, importlib.import_module, failures)

    if not failures:
        return 0
    print("import smoke test FAILED:", file=sys.stderr)
    for name, exc in failures:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"    at {frame.filename}:{frame.lineno}: {frame.line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
