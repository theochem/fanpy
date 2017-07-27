"""Test wfns.tools."""
import os
from nose.tools import assert_raises
import wfns.tools as tools

def test_find_datafile():
    """Test wfns.tools.find_datafile."""
    assert os.path.samefile(os.path.join('..', '..', 'data', 'test', 'h2.xyz'),
                            tools.find_datafile('test/h2.xyz'))
    assert_raises(IOError, tools.find_datafile, 'doesnotexist.txt')
