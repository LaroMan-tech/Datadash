from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "Data support mail" / "Status en distribution.py"

module_globals = runpy.run_path(str(APP_PATH))
main_fn = module_globals.get("main")
if not callable(main_fn):
    raise RuntimeError(f"Geen callable main() gevonden in {APP_PATH}")

main_fn()
