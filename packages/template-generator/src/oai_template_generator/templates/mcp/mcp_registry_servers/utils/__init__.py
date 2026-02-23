import os
import sys

file_root = os.path.dirname(os.path.abspath(__file__))
path_list = [
    file_root,
    os.path.dirname(file_root)
]
for path in path_list:
    if path not in sys.path:
        sys.path.append(path)