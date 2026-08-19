class Value:

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
        assert isinstance(
            other, (int, float)
        ), "Only supporting int/float powers for now"
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
        # Ordene topologicamente cada elemento do grafo
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Em cada operação, aplique a função _update_grad() para acumular o gradiente via regra da cadeia
        self.grad = 1
        for v in reversed(topo):
            v._update_grad()
