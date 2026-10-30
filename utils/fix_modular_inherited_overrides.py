"""Make modular files explicit: copy generated members whose inherited version would use the parent's classes."""

import builtins
import json
import subprocess
import sys

import libcst as cst

sys.path.insert(0, "utils")
import modular_model_converter as mmc  # noqa: E402


captured = {}
mmc.check_inherited_code_uses_parent_definitions = lambda mapper, files: captured.update(mapper=mapper, files={k: dict(v) for k, v in files.items()})


def top_level_names(module):
    out = []
    for node in module.body:
        if isinstance(node, (cst.ClassDef, cst.FunctionDef)):
            out.append(node.name.value)
        else:
            out.append(None)
    return out


def import_lines_for(module, name):
    """Return an import statement code importing `name` if `module` imports it at top level."""
    for node in module.body:
        stmts = [node]
        if isinstance(node, cst.If):  # `if is_torch_available():` blocks
            stmts = list(node.body.body)
        for stmt in stmts:
            if not m_simple(stmt):
                continue
            imp = stmt.body[0]
            if isinstance(imp, cst.ImportFrom) and not isinstance(imp.names, cst.ImportStar):
                for alias in imp.names:
                    asname = alias.asname.name.value if alias.asname else None
                    real = mmc.get_full_attribute_name(alias.name)
                    if (asname or real) == name:
                        mod = cst.Module(body=[]).code_for_node(imp.module) if imp.module else ""
                        dots = "." * len(imp.relative)
                        return f"from {dots}{mod} import {real}" + (f" as {asname}" if asname else "")
            if isinstance(imp, cst.Import):
                for alias in imp.names:
                    asname = alias.asname.name.value if alias.asname else None
                    real = mmc.get_full_attribute_name(alias.name)
                    if (asname or real.split(".")[0]) == name:
                        return f"import {real}" + (f" as {asname}" if asname else "")
    return None


def m_simple(stmt):
    return isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1


class Inserter(cst.CSTTransformer):
    def __init__(self, members):  # {class_name: {key: stmt}}
        self.members = members

    def leave_ClassDef(self, original, updated):
        todo = dict(self.members.get(updated.name.value, {}))
        if not todo:
            return updated
        body = []
        for stmt in updated.body.body:
            key = mmc._statement_key(stmt)
            if key in todo:
                body.append(todo.pop(key))
            elif m_simple(stmt) and isinstance(stmt.body[0], cst.Pass):
                continue
            else:
                body.append(stmt)
        body.extend(todo.values())
        return updated.with_changes(body=updated.body.with_changes(body=body))


class Renamer(cst.CSTTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def leave_Name(self, original, updated):
        return updated.with_changes(value=self.mapping.get(updated.value, updated.value))


def undefined_names(path):
    out = subprocess.run(
        ["ruff", "check", "--select", "F821", "--output-format", "json", "--no-cache", path],
        capture_output=True,
        text=True,
    ).stdout
    return sorted({d["message"].split("`")[1] for d in json.loads(out or "[]")})


def fix(path):
    captured.clear()
    mmc.convert_modular_file(path)
    mapper, files = captured["mapper"], captured["files"]
    mismatches = mmc.find_inherited_code_mismatches(mapper, files)
    if not mismatches:
        return 0, []
    generated = {name: entry["node"] for body in files.values() for name, entry in body.items()}
    members = {}
    for class_name, key, _, _ in mismatches:
        if key is None:
            continue
        stmt = next(s for s in generated[class_name].body.body if mmc._statement_key(s) == key)
        members.setdefault(class_name, {})[key] = stmt
    source = open(path).read()
    module = cst.parse_module(source).visit(Inserter(members))
    open(path, "w").write(module.code)

    # Resolve names the copied code uses but the modular file does not define
    renamed_to_original, import_for = {}, {}
    for file, original in mapper.model_specific_modules.items():
        renamed = original.visit(mapper.renamers[file])
        rel = "..%s.%s" % tuple(file.split(".")[-2:])
        for o, r in zip(top_level_names(original), top_level_names(renamed)):
            if r is not None and r not in renamed_to_original:
                renamed_to_original[r] = (o, f"from {rel} import {o}")
        for name in undefined_names(path):
            if name not in import_for and name not in renamed_to_original:
                line = import_lines_for(renamed, name)
                if line is not None:
                    import_for[name] = line
    missing, rename, imports = [], {}, []
    for name in undefined_names(path):
        if name in renamed_to_original:
            original_name, line = renamed_to_original[name]
            rename[name] = original_name
            imports.append(line)
        elif name in import_for:
            imports.append(import_for[name])
        elif not hasattr(builtins, name):
            missing.append(name)
    module = cst.parse_module(open(path).read())
    if rename:
        module = module.visit(Renamer(rename))
    if imports:
        idx = max(
            i
            for i, node in enumerate(module.body)
            if m_simple(node) and isinstance(node.body[0], (cst.Import, cst.ImportFrom))
        )
        new_nodes = [cst.parse_statement(line) for line in sorted(set(imports))]
        module = module.with_changes(body=[*module.body[: idx + 1], *new_nodes, *module.body[idx + 1 :]])
    code = module.code
    open(path, "w").write(code)
    return len(mismatches), missing


if __name__ == "__main__":
    import time

    paths = sys.argv[1:]
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        try:
            n, missing = fix(path)
            status = f"fixed {n}" + (f" | UNRESOLVED {missing}" if missing else "") if n else "clean"
        except Exception as e:
            status = f"CRASH {type(e).__name__}: {e}"[:200]
        el = time.time() - t0
        print(f"[{i}/{len(paths)}] {path.split('/')[-1]} {status} | elapsed {el:.0f}s | eta ~{el / i * (len(paths) - i):.0f}s", flush=True)
