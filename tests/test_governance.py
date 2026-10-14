from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the locked backport.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_declaration_and_internal_release_tags_are_locked() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    workspace = project["tool"]["quant-workspace"]

    assert project["project"]["version"] == "0.4.2"
    assert workspace["layer"] == "portfolio-risk"
    assert workspace["schemas"] == [
        {"id": "puresaber.instrument-spec", "version": "2.0.0"},
        {"id": "puresaber.execution.account-snapshot", "version": "1.1.0"},
        {"id": "puresaber.execution.order-intent", "version": "1.1.0"},
    ]
    assert workspace["lock-files"] == ["requirements.lock"]

    dependencies = project["project"]["dependencies"]
    assert (
        "quant-data-kit @ git+https://github.com/PureSaber/quant-data-kit.git@ad3440700b226497c40093197f1df4321b49d1e8"
        in dependencies
    )
    assert (
        "quant-execution @ git+https://github.com/PureSaber/quant-execution.git@8423726a9d80b6509d2c016a7478fc0f40a29774"
        in dependencies
    )

    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    assert (
        "quant-data-kit @ git+https://github.com/PureSaber/quant-data-kit.git@ad3440700b226497c40093197f1df4321b49d1e8"
        in lock
    )
    assert (
        "quant-execution @ git+https://github.com/PureSaber/quant-execution.git@8423726a9d80b6509d2c016a7478fc0f40a29774"
        in lock
    )
    assert 'tomli==2.4.1 ; python_version < "3.11"' in lock
    assert "exceptiongroup==1.3.1" in lock
