"""
Cleans dataset folder by removing everything
"""

import os
import shutil

from Misc.config import config

excluded_files = ["splits", "__init__.py", ".gitignore"]

def main():
    # https://stackoverflow.com/a/185941
    folder = config.DATA_PATH
    for filename in os.listdir(folder):
        if filename in excluded_files:
            continue
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    main()