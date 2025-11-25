import pathlib
import os

import pathlib
import sys

import sys
import pathlib
import os
import importlib.util
def build_simulator():
    file_absolute_location = pathlib.Path(__file__).parent.resolve()
    snn_simulator_dir = file_absolute_location

    if not snn_simulator_dir.exists():
        print(f"Le répertoire {snn_simulator_dir} n'existe pas.")
        sys.exit(1)

    setup_py = snn_simulator_dir / "setup.py"

    if not setup_py.exists():
        print(f"Le fichier setup.py n'existe pas dans le répertoire {snn_simulator_dir}.")
        sys.exit(1)

    original_cwd = os.getcwd()
    original_argv = sys.argv.copy()

    os.chdir(str(snn_simulator_dir))

    sys.argv = [str(setup_py), "build_ext", "--inplace"]

    try:
        spec = importlib.util.spec_from_file_location("setup", str(setup_py))
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)
        print("build.sh success\n\n")
    except Exception as e:
        print("build.sh failed")
        print("Erreur :", e)
        sys.exit(1)
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv

# build_simulator()
# clean_simulator()