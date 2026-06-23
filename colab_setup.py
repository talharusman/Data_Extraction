from __future__ import annotations

import os
import shutil
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def mount_drive() -> None:
    if not is_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def ensure_env_file(project_dir: Path) -> Path:
    env_path = project_dir / ".env"
    example_path = project_dir / ".env.example"
    if not env_path.exists() and example_path.exists():
        shutil.copyfile(example_path, env_path)
    return env_path


def main() -> None:
    project_dir = Path(os.environ.get("COLAB_PROJECT_DIR", Path.cwd())).resolve()
    os.chdir(project_dir)

    if is_colab():
        mount_drive()

    env_path = ensure_env_file(project_dir)

    print("Colab setup is ready.")
    print(f"Project dir: {project_dir}")
    print(f".env file: {env_path}")
    print("")
    print("Next:")
    print("1. Edit .env and set HF_MODEL_NAME_OR_PATH")
    print("2. Install deps: pip install -r requirements.txt")
    print("3. Run: python run_all.py")


if __name__ == "__main__":
    main()
