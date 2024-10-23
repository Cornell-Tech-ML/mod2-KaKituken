from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context
from .operators import mul, neg, lt, eq, sigmoid, relu, exp, inv, inv_back, relu_back

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward pass of a scalar function and creates a new Scalar.

        This method performs the forward pass of the mathematical function
        defined in the class, processes the input values (which may be `Scalar`
        instances or other types convertible to `Scalar`), and returns a new `Scalar`
        instance. It also records the operation's context for use in backpropagation.

        Args:
        ----
            *vals (ScalarLike): The input values to the function. These can be instances
            of `Scalar` or any value that can be converted into a `Scalar`.

        Returns:
        -------
            Scalar: A new `Scalar` instance that represents the result of applying the
            function to the inputs, and stores the context for backward computation.

        Raises:
        ------
            AssertionError: If the result of the forward pass is not a float.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward function. Call `add`"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        r"""Backward function. $\partial f(x, y)/\partail x = 1$"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `log`"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = log_back(x)$"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiply function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward function. Call `mul`"""
        ctx.save_for_backward(a, b)
        return mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        r"""Backward function. $\partial f(x, y)/\partail x = y$"""
        (a, b) = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    r"""Inv function $f(x) = \frac{1}{x}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `inv`"""
        ctx.save_for_backward(a)
        return inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = inv_back(x)$"""
        (a,) = ctx.saved_values
        return inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negative function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `neg`"""
        return neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = -1$"""
        return -d_output


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `sigmoid`"""
        ctx.save_for_backward(a)
        return sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = sigmoid(x)(1-sigmoid(x))$"""
        (a,) = ctx.saved_values
        return d_output * sigmoid(a) * (1 - sigmoid(a))


class ReLU(ScalarFunction):
    """ReLU function $f(x) = x if x > 0 else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `relu`"""
        ctx.save_for_backward(a)
        return relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = relu_back(x)$"""
        (a,) = ctx.saved_values
        return relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponent function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward function. Call `exp`"""
        ctx.save_for_backward(a)
        return exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward function. $f'(x) = e^x$"""
        (a,) = ctx.saved_values
        return exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = int(x < y)$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward function. Call `lt`"""
        return lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward function. Return 0 as its subgradient."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal to function $f(x, y) = int(x == y)$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward function. Call `eq`"""
        return eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward function. Return 0 as its subgradient."""
        return 0.0, 0.0
