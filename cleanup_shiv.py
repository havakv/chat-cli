# Example taken from https://shiv.readthedocs.io/en/latest/#preamble
import shutil
from pathlib import Path

import shiv.bootstrap  # type: ignore


def remove_old_versions(site_packages: Path) -> None:
    # Get a handle of the current PYZ's site_packages directory
    current = site_packages.parent
    name, build_id = current.name.split("_")
    cache_path = current.parent

    for path in cache_path.iterdir():
        if path.name.startswith(f"{name}_") and not path.name.endswith(build_id):
            shutil.rmtree(path)


if __name__ == "__main__":
    # These variables are injected by shiv.bootstrap
    site_packages: Path
    env: shiv.bootstrap.environment.Environment

    remove_old_versions(site_packages)  # type: ignore # noqa: F821
