"""Handy tools for lazy people."""
import os
import glob


def find_datafile(file_name):
    """Find file from data directory.

    Parameters
    ----------
    file_name : str
        Name of the file

    Returns
    -------
    file_path : str
        Absolute path of the file.

    Raises
    ------
    IOError
        If file cannot be found.
        If more than one file is found.

    """
    try:
        rel_path, = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'data', file_name))
    except ValueError:
        raise IOError('Having trouble finding the file, {0}.'.format(file_name))
    return os.path.abspath(rel_path)
