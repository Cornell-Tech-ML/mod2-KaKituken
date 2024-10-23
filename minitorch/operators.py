"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiplies a and b"""
    return a * b


def id(a: Any) -> Any:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Add a and b"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return 0.0 - a


def lt(a: float, b: float) -> float:
    """Checks if a number is less than b"""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if a is equal to b"""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of a and b"""
    return a if a >= b else b


def is_close(a: float, b: float) -> bool:
    """Checks if a and b numbers are close in value"""
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x >= 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm of x"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function of x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Inv - Calculates the reciprocal"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return 1.0 / x * y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -1.0 / (x * x) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return 0.0 if x <= 0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(li: List, fn: Callable) -> Iterable:
    """Applies a function to each item in an iterable and yields the results.

    Args:
    ----
        li: An iterable of any type.
        fn: A callable function that takes an item from the iterable as an argument and returns a result.

    Yields:
    ------
        The result of applying `fn` to each item in `li`.

    """
    for item in li:
        yield fn(item)


def zipWith(it_a: List, it_b: List, fn: Callable) -> Iterable:
    """Applies a function to corresponding elements from two iterables and yields the results.

    Args:
    ----
        it_a: The first iterable.
        it_b: The second iterable.
        fn: A callable function that takes two arguments, one from each iterable, and returns a result.

    Yields:
    ------
        The result of applying `fn` to corresponding elements from `it_a` and `it_b`.

    """
    for a, b in zip(it_a, it_b):
        yield fn(a, b)


def reduce(it: List, fn: Callable) -> float:
    """Reduces an iterable to a single value by applying a binary function cumulatively to the items.

    Args:
    ----
        it: An iterable of any type.
        fn: A callable function that takes two arguments and returns a single result, used to combine the elements.

    Returns:
    -------
        The result of applying `fn` cumulatively to the items of `it`, or the iterable itself if it contains fewer than two items.

    """
    if len(it) == 0:
        raise TypeError("Cannot reduce the input with length 0")
    elif len(it) == 1:
        return it[0]
    else:
        iterator = iter(it)
        result = next(iterator)  # Initialize with the first item of the iterable

        for item in iterator:
            result = fn(result, item)

        return result


def negList(li: List[float]) -> List[float]:
    """Applies a negation function to each element in a list of floats.

    Args:
    ----
        li: A list of floats.

    Returns:
    -------
        A new list where each element is the negated value of the corresponding element in `li`.

    """
    return list(map(li, neg))


def addLists(li_a: List[float], li_b: List[float]) -> List[float]:
    """Adds corresponding elements from two lists of floats.

    Args:
    ----
        li_a: The first list of floats.
        li_b: The second list of floats.

    Returns:
    -------
        A new list where each element is the sum of the corresponding elements in `li_a` and `li_b`.

    """
    return list(zipWith(li_a, li_b, add))


def sum(li: List[float]) -> float:
    """Sums all the elements in a list of floats.

    Args:
    ----
        li: A list of floats.

    Returns:
    -------
        The sum of all elements in `li`.

    """
    if len(li) == 0:
        return 0
    if len(li) == 1:
        return li[0]
    return reduce(li, add)


def prod(li: List[float]) -> float:
    """Multiplies all the elements in a list of floats.

    Args:
    ----
        li: A list of floats.

    Returns:
    -------
        The product of all elements in `li`.

    """
    if len(li) == 0:
        return 1
    if len(li) == 1:
        return li[0]

    return reduce(li, mul)


if __name__ == "__main__":
    li = [1.0, 0.0, 0.0]

    print(prod(li))
