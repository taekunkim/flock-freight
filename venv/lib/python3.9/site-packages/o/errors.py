class UnknownOperator(Exception):
    """Raised when an unknown operator is provided."""

    def __init__(self, operator):
        self.operator = operator
        super().__init__(f"Unknown operator: " + operator)
