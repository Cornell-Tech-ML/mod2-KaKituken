from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol
import copy


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_minus = list(copy.copy(vals))
    vals_add = list(copy.copy(vals))
    vals_minus[arg] -= epsilon
    vals_add[arg] += epsilon
    return (f(*vals_add) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable.

        This method adds the given derivative to the existing gradient of the
        variable. It is typically used during backpropagation to accumulate
        the gradients across multiple operations.

        Args:
        ----
            x (Any): The derivative to be added to the variable's current gradient.

        """

    @property
    def unique_id(self) -> int:
        """Return the unique id for a node"""
        return 0

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node"""
        return False

    def is_constant(self) -> bool:
        """Check if the node is a constant"""
        return False

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent nodes of a given node"""
        return []

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Using chain rule to calculate the gradient"""
        return []


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable(Variable): The right-most variable

    Returns:
    -------
        Iterable[Variable]: Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # res = []
    # visited = set()

    # def dfs_backtrack(v: Variable) -> None:
    #     for p in v.parents:
    #         if p.unique_id not in visited:
    #             dfs_backtrack(p)
    #     visited.add(v.unique_id)
    #     res.append(v)

    # dfs_backtrack(variable)

    # return res

    res = []
    visited = set()

    def dfs_backtrack(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for p in v.parents:
                if not p.is_constant():
                    dfs_backtrack(p)
        visited.add(v.unique_id)
        res.insert(0, v)

    dfs_backtrack(variable)
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Args:
    ----
        variable(Variable): The right-most variable
        deriv(Any): Its derivative that we want to propagate backward to the leaves.

    """
    # TODO: Implement for Task 1.4.
    # nodes = list(
    #     topological_sort(variable)
    # )  # from leaves to root, so regard as a stack
    # dict_var_deriv = {}
    # for node in nodes:
    #     dict_var_deriv[node.unique_id] = 0.0
    # dict_var_deriv[variable.unique_id] = deriv
    # while len(nodes):
    #     cur_node = nodes.pop()
    #     if cur_node.is_leaf():
    #         cur_node.accumulate_derivative(dict_var_deriv[cur_node.unique_id])
    #     else:
    #         node_with_deriv = cur_node.chain_rule(dict_var_deriv[cur_node.unique_id])
    #         for n, d in node_with_deriv:
    #             print(n.unique_id)
    #             dict_var_deriv[n.unique_id] += d
    nodes = topological_sort(variable)
    dict_var_deriv = {}
    dict_var_deriv[variable.unique_id] = deriv
    for node in nodes:
        deriv = dict_var_deriv[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(deriv)
        else:
            # print(f"{deriv=}")
            for n, d in node.chain_rule(deriv):
                if n.is_constant():
                    continue
                dict_var_deriv.setdefault(n.unique_id, 0.0)
                dict_var_deriv[n.unique_id] = dict_var_deriv[n.unique_id] + d

    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrive the saved input value for computing node"""
        return self.saved_values
