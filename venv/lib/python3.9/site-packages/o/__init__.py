__version__ = "0.0.2.b2"

from typing import List as _List
from typing import Literal as _Literal

from ._utils import _format_operator_type_doc
from .errors import UnknownOperator

OperatorSelection = _Literal[
    "arithmetic", "assignment", "comparison", "bitewise", "matrix"
]

_map = {
    "+": {
        "name": "addition",
        "func": "add",
        "type": "arithmetic",
    },
    "-": {
        "name": "subtraction",
        "func": "neg",
        "type": "arithmetic",
    },
    "*": {
        "name": "multiplication",
        "func": "mul",
        "type": "arithmetic",
    },
    "/": {
        "name": "division",
        "func": "truediv",
        "type": "arithmetic",
    },
    "%": {
        "name": "modulus",
        "func": "mod",
        "type": "arithmetic",
    },
    "**": {
        "name": "exponentiation",
        "func": "pow",
        "type": "arithmetic",
    },
    "//": {
        "name": "floor division",
        "func": "floordiv",
        "type": "arithmetic",
    },
    "=": {
        "name": "assignment",
        "func": None,
        "type": "assignment",
    },
    "+=": {
        "name": "addition assignment",
        "func": "iadd",
        "type": "assignment",
    },
    "-=": {
        "name": "subtraction assignment",
        "func": "isub",
        "type": "assignment",
    },
    "*=": {
        "name": "multiplication assignment",
        "func": "imul",
        "type": "assignment",
    },
    "/=": {
        "name": "division assignment",
        "func": "itruediv",
        "type": "assignment",
    },
    "%=": {
        "name": "modulus assignment",
        "func": "imod",
        "type": "assignment",
    },
    "//=": {
        "name": "floor division assignment",
        "func": "ifloordiv",
        "type": "assignment",
    },
    "**=": {
        "name": "exponentiation assignment",
        "func": "ipow",
        "type": "assignment",
    },
    ":=": {
        "name": "walrus",
        "func": None,
        "type": "assignment",
    },
    "|=": {
        "name": "or assignment",
        "func": "ior",
        "type": "assignment",
    },
    "^=": {
        "name": "exclusive or assignment",
        "func": "ixor",
        "type": "assignment",
    },
    ">>=": {
        "name": "right shift assignment",
        "func": "irshift",
        "type": "assignment",
    },
    "<<=": {
        "name": "left shift assignment",
        "func": "ilshift",
        "type": "assignment",
    },
    "@=": {
        "name": "matmul assignment",
        "func": "imatmul",
        "type": "assignment",
    },
    "==": {
        "name": "equal",
        "func": "eq",
        "type": "comparison",
    },
    "!=": {
        "name": "not equal",
        "func": "ne",
        "type": "comparison",
    },
    ">": {
        "name": "greater than",
        "func": "gt",
        "type": "comparison",
    },
    "<": {
        "name": "less than",
        "func": "lt",
        "type": "comparison",
    },
    ">=": {
        "name": "greater than or equal to",
        "func": "ge",
        "type": "comparison",
    },
    "<=": {
        "name": "less than or equal to",
        "func": "le",
        "type": "comparison",
    },
    "&": {
        "name": "and",
        "func": "and_",
        "type": "bitwise",
    },
    "|": {
        "name": "or",
        "func": "or_",
        "type": "bitwise",
    },
    "^": {
        "name": "either or",
        "func": "xor",
        "type": "bitwise",
    },
    "~": {
        "name": "invert",
        "func": "invert",
        "type": "bitwise",
    },
    "<<": {
        "name": "left shift",
        "func": "lshift",
        "type": "bitwise",
    },
    ">>": {
        "name": "right shift",
        "func": "rshift",
        "type": "bitwise",
    },
    "@": {
        "name": "matmul",
        "func": "matmul",
        "type": "matrix",
    },
}


class OperatorType:
    def __init__(self, operator):
        self.name = operator

    def __repr__(self):
        return self.__class__.__name__.lower()[:-12]

    __str__ = __repr__


@_format_operator_type_doc()
class ArithmeticOperatorType(OperatorType):
    pass


@_format_operator_type_doc()
class AssignmentOperatorType(OperatorType):
    pass


@_format_operator_type_doc()
class BitwiseOperatorType(OperatorType):
    pass


@_format_operator_type_doc()
class ComparisonOperatorType(OperatorType):
    pass


@_format_operator_type_doc()
class MatrixOperatorType(OperatorType):
    pass


class Operator:
    """Get an operator name, methods and other information.

    This class acts as a container for an operator, providing
    various attributes/properties.

    Attributes
    ----------
    operator: str
        The operator provided through initialization.
        This is the only crafted attribute accessible
        if UnknownOperator is raised upon initialization.

        o = Operator("+")
        o.operator
        >>> "+"

    name: str
        The name for the operator.

        o = Operator("+")
        o.name
        >>> "Addition"

    methods: List[str]
        The various "special methods" that you can use in classes
        in order to customize the behaviour of the operator.
        They are provided in dunder form.

        o = Operator("+")
        o.methods
        >>> ['__add__']

    function: Union[types.BuiltinFunctionType, None]
        The function for this operator, derived from the operator
        module. Returns None if the operator has no function.

        o = Operator("+")
        o.function
        >>> "<built-in function add>"

        o = Operator(":=")
        o.function
        >>> None

    type: OperatorType
        The type for the operator, provided appropriately based on what
        it aims to operate upon. Currently, there are 4 different types
        of operator:

        - Arithmetic
        - Assignment
        - Bitewise
        - Comparison
        - Matrix

        This type attribute returns an object in the form <type>OperatorType.
        All these operator types subclass OperatorType.

        o = Operator("+")
        isinstance(o.type, ArithmeticOperatorType)
        >>> True


    Parameters
    ----------
    operator: str
        The operator to get information for.

    Returns
    -------
    Operator
        The operator object with information.

    Raises
    ------
    UnknownOperator
        The provided operator is unknown.
    """

    import operator as _operator_mod

    type_mapping = {
        "arithmetic": ArithmeticOperatorType,
        "assignment": AssignmentOperatorType,
        "bitwise": BitwiseOperatorType,
        "comparison": ComparisonOperatorType,
        "matrix": MatrixOperatorType,
    }

    def _format_f(self, f):
        if f is None:
            return f
        return f"operator.{f.__name__}"

    def __init__(self, operator):
        self.operator = operator
        try:
            self.__idx = _map[operator]
        except KeyError:
            raise UnknownOperator(operator) from None
        self.name = self.__idx["name"]
        t = self.__idx["type"]
        self.type = self.type_mapping[t](t)
        function = self.__idx["func"]
        if isinstance(function, str):
            self.function = getattr(self._operator_mod, function, None)
        else:
            self.function = function

    def __repr__(self):
        quote = lambda x: f"'{x}'"
        kwargs = {
            "operator": quote(self.operator),
            "name": quote(self.name),
            "type": self.type,
            "methods": self.methods,
            "function": self._format_f(self.function),
        }

        return "<class 'Operator({})'>".format(
            ", ".join(f"{k}={v}" for k, v in kwargs.items())
        )

    @property
    def methods(self):
        dunder = lambda x: f"__{x}__"
        alias_map = {"inv": "invert"}

        f = self.function
        if f is None:
            return []

        methods = []
        if f := getattr(self.function, "__name__", None):
            if dunder(f) in dir(self._operator_mod):
                if f in alias_map.keys():
                    methods = [f, alias_map[f]]
                else:
                    methods = [f]
                methods = list(map(dunder, methods))

        return methods


def get(operator) -> Operator:
    """Get an operator's name, methods and other information.

    This function takes a single operator, and returns
    <class 'Operator'>, with various attributes available from it.

    Example Usage:

    o.get("+")
    >>> "<class 'Operator(operator='+', name='addition', type=arithmetic, methods=['__add__'], function=operator.add)'>"

    Get the methods from an operator by simply adding the methods attribute:

    operator = o.get("+")
    operator.methods
    >>> ['__add__']

    Parameters
    ----------
    operator: str
        The operator to get information for.

    Returns
    -------
    Operator
        The operator object with information.

    Raises
    ------
    UnknownOperator
        The provided operator is unknown.
    """
    return Operator(operator)


def gettypes(operator_type: OperatorSelection) -> _List[Operator]:
    """Get all operators of a certain type (e.g. assignment).

    The operator types include:

    - arithmetic
    - assignment
    - bitewise
    - comparison

    Parameters
    ----------
    operator_type: OperatorSelection
        The provided operator type. Must be one of the operators
        listed above.

    Returns
    -------
    List[Operator]
        A list of operators of the given type.
    """
    types = ["arithmetic", "assignment", "bitewise", "comparison"]
    if not operator_type in types:
        raise ValueError(f"Operator type must be one of " + ", ".join(types))

    def inner():
        for key, val in _map.items():
            if val["type"] == operator_type:
                yield Operator(key)

    return list(inner())
