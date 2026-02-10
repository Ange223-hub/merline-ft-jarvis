import os
import runpy


def main() -> None:
    script = os.path.join(os.path.dirname(__file__), "scripts", "launch_safe.py")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
