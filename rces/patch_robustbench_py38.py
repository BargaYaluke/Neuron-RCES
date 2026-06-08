"""
One-shot fix: make an installed RobustBench import under Python 3.8.

Current RobustBench uses PEP 585 builtin generics in annotations, e.g.
    norm_layer: list[Callable[..., nn.Module]]
which raises `TypeError: 'type' object is not subscriptable` at import time on
Python 3.8 (PEP 585 only works at runtime from 3.9+). The standard fix is to add
`from __future__ import annotations` (PEP 563) to each affected module, which
turns annotations into un-evaluated strings.

This script inserts that future-import into every robustbench .py that lacks it,
placing it right after the module docstring (or after leading comments). It is
idempotent and only edits the installed robustbench package.

Usage (in the same env that runs training):
    python rces/patch_robustbench_py38.py

If a *non-annotation* 3.9-only construct remains (e.g. a module-level
`X = list[int]`), a different error will surface afterwards — send it back and we
will patch that line specifically, or switch to a Python>=3.9 environment.
"""

import ast
import glob
import os
import sys

FUTURE = "from __future__ import annotations\n"


def main():
    try:
        import robustbench
    except ImportError:
        print("robustbench is not installed in this interpreter; nothing to patch.")
        return 1
    except Exception:
        # Even a failing import exposes __file__ via the partially-initialized module.
        import importlib.util
        spec = importlib.util.find_spec("robustbench")
        if spec is None or not spec.submodule_search_locations:
            print("Could not locate the robustbench package directory.")
            return 1

        class _Stub:
            __file__ = os.path.join(list(spec.submodule_search_locations)[0],
                                    "__init__.py")
        robustbench = _Stub()

    base = os.path.dirname(robustbench.__file__)
    print(f"robustbench package dir: {base}")

    patched, skipped = [], 0
    for path in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        if "from __future__ import annotations" in src:
            skipped += 1
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            skipped += 1
            continue

        lines = src.splitlines(keepends=True)
        insert = 0
        body = getattr(tree, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            # after the module docstring (end_lineno is 1-based last line)
            insert = body[0].end_lineno
        else:
            # skip leading shebang / coding / blank / comment lines
            i = 0
            while i < len(lines) and (
                    lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
                i += 1
            insert = i

        lines.insert(insert, FUTURE)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        patched.append(os.path.relpath(path, base))

    print(f"patched {len(patched)} file(s); {skipped} already-ok/unparseable skipped.")
    for p in patched:
        print("   +", p)

    # quick verification (drop any half-initialized robustbench modules first so
    # the re-import re-executes the freshly patched source files)
    for mod in [m for m in list(sys.modules) if m == "robustbench"
                or m.startswith("robustbench.")]:
        del sys.modules[mod]
    print("\nverifying 'from robustbench.utils import load_model' ...")
    try:
        from robustbench.utils import load_model  # noqa: F401
        print("OK: robustbench imports cleanly now.")
        return 0
    except Exception as e:
        print(f"STILL FAILING: {type(e).__name__}: {e}")
        print("Send this error back; it is likely a non-annotation 3.9-only line.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
