"""Check project layout without importing Isaac, torch or starting tasks."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    mapping = json.loads((ROOT / "docs/script_migration.json").read_text(encoding="utf-8"))
    assert len(mapping) == len(set(mapping.values())), "Duplicate implementations"
    checked = 0
    for folder in ("apps", "src", "perception", "tool", "tools", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            checked += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "PROJECT_ROOT" for t in node.targets
                ):
                    # Root expressions are simple pathlib expressions, not arbitrary code.
                    value = node.value
                    assert isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant)
                    assert path.resolve().parents[value.slice.value] == ROOT, path
                if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                        and isinstance(node.left, ast.Name) and node.left.id == "PROJECT_ROOT"
                        and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                        and node.right.value.endswith(".py")):
                    assert (ROOT / node.right.value).is_file(), (path, node.right.value)
                    assert not node.right.value.startswith("scripts/"), "Core depends on legacy entry"
    for old, new in mapping.items():
        assert (ROOT / new).is_file(), new
        entry = (ROOT / "scripts" / old).read_text(encoding="utf-8")
        assert repr(new) in entry and "exec(compile(" in entry, old
    print(f"PASS: {checked} Python files; {len(mapping)} compatibility entries; roots and dependencies valid")


if __name__ == "__main__":
    main()
