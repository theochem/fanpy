def raises_exception(fun, exception=Exception):
    """
    Check if a test function (with no arguments) raises an exception.

    Parameters
    ----------
    fun : function
        The function that should raise an exception.
    exception : Exception, optional
        The exception that is expected to be raised.  Defaults to the base Exception
        object.

    Returns
    -------
    exception_raised : bool

    """

    try:
        fun()
    except exception:
        return True
    except:
        return False
    else:
        return False
