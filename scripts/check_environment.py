"""Check Python packages and local inputs without modifying the workspace."""

from __future__ import annotations

import importlib
import sys

from project_paths import HLOC_ROOT, LIGHTGLUE_ROOT, PILOT_ROOT, SHIRT_ROOT


PACKAGES = {
    "cv2": "opencv-python",
    "h5py": "h5py",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "PIL": "Pillow",
    "psutil": "psutil",
    "pycolmap": "pycolmap",
    "scipy": "scipy",
    "torch": "torch",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    failed = False

    for module_name, package_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] package {package_name}: {version}")
        except Exception as exc:  # Import failures can include missing native DLLs.
            failed = True
            print(f"[MISSING] package {package_name}: {exc}")

    paths = {
        "SHIRT_ROOT": SHIRT_ROOT,
        "POSE_PILOT_ROOT": PILOT_ROOT,
        "HLOC_ROOT": HLOC_ROOT,
        "LIGHTGLUE_ROOT": LIGHTGLUE_ROOT,
    }
    for name, path in paths.items():
        state = "OK" if path.exists() else "MISSING"
        print(f"[{state}] {name}: {path}")
        failed |= not path.exists()

    if sys.version_info < (3, 10):
        failed = True
        print("[UNSUPPORTED] Python 3.10 or newer is recommended.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
