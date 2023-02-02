from inspect import signature as _signature

_OBJECT_DOC = """An object which acts as a type for the Operator(...).type attribute.

It's main purpose is to be used conventially as a type checker, for example
using isinstance. This object is wrapped around operator types when they fall
under the {obj} "genre".

This object subclasses OperatorType which is accessible directly from the module
namespace.
"""


def _format_operator_type_doc():
    def decorator(cls):
        cls.__doc__ = _OBJECT_DOC.format(obj=cls.__name__[:-12].lower())
        return cls

    return decorator
