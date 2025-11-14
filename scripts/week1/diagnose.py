"""Diagnose Week 1 setup issues."""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False


def check_dependencies():
    """Check required packages."""
    print("\nChecking Python packages...")

    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy.stats",
        "matplotlib": "matplotlib.pyplot",
    }

    all_ok = True
    for name, import_path in required.items():
        try:
            __import__(import_path)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} NOT INSTALLED")
            all_ok = False

    return all_ok


def check_project_files():
    """Check required project files."""
    print("\nChecking project files...")

    required_files = [
        "src/planner/q_learning.py",
        "src/planner/q_learning_init.py",
        "src/planner/alns_matheuristic.py",
        "src/config/__init__.py",
        "src/core/route.py",
        "src/strategy/charging_strategies.py",
        "tests/optimization/common.py",
        "tests/optimization/presets.py",
    ]

    all_ok = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} NOT FOUND")
            all_ok = False

    return all_ok


def check_pythonpath():
    """Check if src is in Python path."""
    print("\nChecking PYTHONPATH...")

    src_path = Path("src").absolute()
    if str(src_path) in sys.path or "src" in sys.path:
        print(f"  ✓ src directory in path")
        return True
    else:
        print(f"  ✗ src directory NOT in path")
        print(f"    Current sys.path: {sys.path[:3]}...")
        return False


def test_imports():
    """Test critical imports."""
    print("\nTesting imports...")

    imports = [
        ("planner.q_learning_init", "QInitStrategy"),
        ("planner.q_learning", "QLearningOperatorAgent"),
        ("config", "QLearningParams"),
        ("tests.optimization.common", "get_scale_config"),
    ]

    all_ok = True
    for module, item in imports:
        try:
            mod = __import__(module, fromlist=[item])
            getattr(mod, item)
            print(f"  ✓ from {module} import {item}")
        except ImportError as e:
            print(f"  ✗ from {module} import {item}")
            print(f"    Error: {e}")
            all_ok = False
        except AttributeError as e:
            print(f"  ✗ {module}.{item} not found")
            print(f"    Error: {e}")
            all_ok = False

    return all_ok


def main():
    """Run all diagnostics."""
    print("=" * 70)
    print("Week 1 Setup Diagnostics")
    print("=" * 70)
    print()

    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Project files": check_project_files(),
        "PYTHONPATH": check_pythonpath(),
        "Imports": test_imports(),
    }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    if all(results.values()):
        print("\n✓ All checks passed! System is ready.")
        print("\nNext step:")
        print("  python scripts\\week1\\run_experiment.py --scenario small --init_strategy zero --seed 2025 --output test.json --verbose")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")

        # Provide specific help
        if not results["Dependencies"]:
            print("\nTo install missing packages:")
            print("  pip install numpy pandas scipy matplotlib")

        if not results["PYTHONPATH"]:
            print("\nTo set PYTHONPATH (PowerShell):")
            print("  $env:PYTHONPATH = \".\\src;$env:PYTHONPATH\"")

        return 1


if __name__ == "__main__":
    sys.exit(main())
