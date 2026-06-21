from __future__ import annotations

from pathlib import Path


class DocumentReader:
    def __init__(self, supported_extensions: list[str]) -> None:
        self.supported_extensions = {ext.lower() for ext in supported_extensions}

    def scan(self, root_folder: str | Path) -> list[Path]:
        root = Path(root_folder)
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.supported_extensions
            and not path.name.startswith("~$")
        )
