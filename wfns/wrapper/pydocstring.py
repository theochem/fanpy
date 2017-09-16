"""Wrapper for the pydocstring module."""
from functools import wraps


# FIXME: this function is copied from pydocstring
def kwarg_wrapper(wrapper):
    """Wraps the keyword arguments into the wrapper.

    The wrapper behaves differently when used as a decorator if the arguments are given. i.e.
    @decorator vs @decorator(). Therefore, the default keyword values of the wrapper must be changed
    (with another wrapper).
    """
    @wraps(wrapper)
    def new_wrapper(obj=None, **kwargs):
        """Reconstruction of the provided wrapper so that keyword arguments are rewritten.

        When a wrapper is used as a decorator and is not called (i.e. no parenthesis),
        then the wrapee (wrapped object) is automatically passed into the decorator. This function
        changes the "default" keyword arguments so that the decorator is not called (so that the
        wrapped object is automatically passed in). If the decorator is called,
        e.g. `x = decorator(x)`, then it simply needs to return the wrapped value.
        """
        # the decorator is evaluated.
        if obj is None and len(kwargs) > 0:
            # Since no object is provided, we need to turn the wrapper back into a form so that it
            # will automatically pass in the object (i.e. turn it into a function) after overwriting
            # the keyword arguments
            return lambda orig_obj: wrapper(orig_obj, **kwargs)
        else:
            # Here, the object is provided OR keyword argument is not provided.
            # If the object is provided, then the wrapper can be executed.
            # If the object is not provided and keyword argument is not provided, then an error will
            # be raised.
            return wrapper(obj, **kwargs)

    return new_wrapper


try:
    import pydocstring
except ModuleNotFoundError:
    @kwarg_wrapper
    def docstring(obj, **kwargs):
        return obj

    @kwarg_wrapper
    def docstring_class(obj, **kwargs):
        return obj
else:
    docstring = pydocstring.wrapper.docstring
    docstring_class = pydocstring.wrapper.docstring_class
