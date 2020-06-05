

def diff(a, b):
    """Absolute difference between variants.
    a - control variant, b - treatment (test) variant"""
    return b - a

def lift(a, b):
    """Percentage lift between variants.
    a - control variant, b - treatment (test) variant"""
    return b / a - 1

def symmetric_lift(a, b):
    """Symmetric Percentage Diff between variants"""
    return (b - a) / ((a + b) / 2)