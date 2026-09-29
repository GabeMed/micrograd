"""A scalar-valued reverse-mode autograd engine."""

import math


class Value:
    """A scalar node in a computation graph.

    Stores its value (``data``), the derivative of the graph output with respect to
    it (``grad``), the nodes it was computed from (``_prev``) and a closure
    (``_update_grad``) that applies the local chain rule to those nodes.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0
        self._update_grad = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):  # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _update_grad():
            self.grad += out.grad
            other.grad += out.grad

        out._update_grad = _update_grad

        return out

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _update_grad():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._update_grad = _update_grad

        return out

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _update_grad():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._update_grad = _update_grad

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _update_grad():
            self.grad += (out.data > 0) * out.grad

        out._update_grad = _update_grad

        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _update_grad():
            self.grad += (1 - t**2) * out.grad

        out._update_grad = _update_grad

        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _update_grad():
            self.grad += e * out.grad

        out._update_grad = _update_grad

        return out

    def log(self):
        assert self.data > 0, "log is only defined for positive values"
        out = Value(math.log(self.data), (self,), "log")

        def _update_grad():
            self.grad += out.grad / self.data

        out._update_grad = _update_grad

        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward_prop(self):
        # Topologically sort every node of the graph. The DFS is iterative so that
        # deep graphs (long sums over a dataset) do not hit Python's recursion limit.
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            v, children_done = stack.pop()
            if children_done:
                topo.append(v)
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            stack.extend((child, False) for child in v._prev if child not in visited)

        # Visit nodes from the output back to the leaves, applying each node's
        # _update_grad() to accumulate gradients through the chain rule.
        self.grad = 1
        for v in reversed(topo):
            v._update_grad()

    backward = backward_prop
