r"""Script to drop convert docstring from those based on pydocstring to normal docstring."""
import os
import sys
import pydocstring.scripts.pydocstring_convert


# find parent module directory
# ASSUMES: this file is in wrapper directory of the python module
cwd = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.split(cwd)[0]
parent_len = len(parent_dir.split(os.sep))

# add parent directory to python path to ensure that it is called first (just in case there are more
# than one wfns module)
sys.path.insert(0, parent_dir)

# get modules
module_paths = []
for dirname, _, filenames in os.walk(parent_dir):
    for filename in filenames:
        module, ext = os.path.splitext(filename)
        if ext != '.py':
            continue
        if module in ['horton_gaussian_fchk', 'horton_hartreefock']:
            continue
        full_filename = os.path.join(dirname, filename)
        pydocstring.scripts.pydocstring_convert.replace_docstrings(full_filename, 'update')
        print(full_filename)
