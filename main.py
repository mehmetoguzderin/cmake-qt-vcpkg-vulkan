import os
import sys

directory_path = "build"
full_path = os.path.abspath(directory_path)

if full_path not in sys.path:
    sys.path.append(full_path)

# if main
if __name__ == "__main__":
    import PythonCxx

    lib = PythonCxx.Lib()
    lib.print_help()
