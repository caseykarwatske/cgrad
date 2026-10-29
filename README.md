# cgrad

**A lightweight automatic differentiation engine written in C++ using only the standard library.**

cgrad is a from-scratch exploration of tensor computation and reverse-mode automatic differentiation, inspired by [DannyTorch](https://github.com/dannypeelen/dannytorch) and [micrograd](https://github.com/karpathy/micrograd). The project aims to grow from a small computation-graph engine into a minimal framework capable of training neural networks, while keeping its underlying algorithms and data structures understandable.

> **Status:** Early development. Automatic differentiation for addition and multiplication is implemented; neural network training is not yet supported.

## Features

- **Reverse-mode automatic differentiation:** Computes gradients through dynamically constructed computation graphs using the chain rule.
- **Automatic computation graph construction:** Arithmetic operations create new nodes that retain their input dependencies and backward callbacks.
- **Element-wise operations:** Addition (`+`) and multiplication (`*`), including their backward rules.
- **Dependency-aware backpropagation:** Uses a Kahn-style topological traversal to process each operation after its downstream gradient contributions have arrived.
- **Templated values:** `Tensor<T>` stores values in `std::vector<T>` and gradients in `std::vector<double>`.
- **Standard-library-only implementation:** Uses STL containers, smart pointers, and callable objects; no external numerical or autodiff dependencies.
- **Graph ownership management:** Shared ownership keeps input nodes alive, while weak references in backward callbacks avoid reference cycles involving the output node.

## Quick start

The current interface is defined in [`src/Tensor.h`](src/Tensor.h). It supports constructing scalar tensors and composing differentiable expressions:

```cpp
#include "Tensor.h"

int main() {
    Tensor<double> x(2.0);
    Tensor<double> y(3.0);

    // z = xy + x = 8
    auto z = x * y + x;

    // Compute gradients through the expression graph.
    z.backward();

    // z.item() returns the scalar value (8.0).
}
```

With respect to the inputs, the expected derivatives are:

\[
\frac{\partial z}{\partial x}=y+1=4, \qquad
\frac{\partial z}{\partial y}=x=2.
\]

**Note:** The backward pass stores gradients internally, but the current public API does not yet expose them. A gradient accessor is on the roadmap.

For a standalone experiment, save the example as `example.cpp` and compile with a C++17-capable compiler, pointing the include path at `src/`:

```bash
g++ -std=c++17 -Isrc example.cpp -o example
./example
```

## How it works

Each `Tensor<T>` refers to a shared `TensorData<T>` object containing:

- A vector of values and a vector of gradients
- References to the input nodes that produced it (for binary operations)
- A backward callback implementing the operation's local derivative

For the expression `z = x * y + x`, cgrad builds a graph equivalent to:

```text
x -----------+------------------+
             |                  |
             v                  |
y ---------> (*)                 |
             |                  |
             +-------> (+) <----+
                         |
                         v
                         z
```

Calling `z.backward()` seeds the output gradient and walks dependencies in reverse computation order. Each backward callback adds its gradient contributions to its inputs. Consumer counts ensure a node is processed only after all of its downstream paths have contributed.

## Roadmap

The following features are **planned**, not currently implemented.

### Phase 1 — Autodiff correctness and API

- [ ] Add public accessors for tensor values and gradients
- [ ] Add `zero_grad()` for clearing accumulated gradients
- [ ] Define and test repeated `backward()` behavior and intermediate-gradient reset semantics
- [ ] Validate operand sizes and reject incompatible tensor operations
- [ ] Support explicit upstream gradients for non-scalar outputs (or restrict implicit `backward()` to scalar outputs)
- [ ] Add unit tests for shared subgraphs, reused operands, and numerical gradient checks

### Phase 2 — Tensor math

- [ ] Add subtraction, division, and scalar arithmetic
- [ ] Add shape metadata and multidimensional tensor support
- [ ] Implement matrix multiplication and its backward pass
- [ ] Add reductions such as `sum()` and `mean()`
- [ ] Implement broadcasting with correct gradient reduction
- [ ] Add nonlinear operations and activations such as ReLU, tanh, and sigmoid

### Phase 3 — Neural network training

- [ ] Add mutable trainable parameters and parameter collection
- [ ] Implement basic loss functions (MSE first, then cross-entropy)
- [ ] Implement SGD and a training-loop API
- [ ] Build a fully connected layer and a small multilayer perceptron
- [ ] Demonstrate learning on a small dataset such as XOR
- [ ] Add minibatch support and an Adam optimizer

### Phase 4 — Performance and tooling

- [ ] Benchmark forward and backward passes against a reference implementation
- [ ] Profile graph construction, allocations, and traversal overhead
- [ ] Explore contiguous storage and more efficient tensor kernels
- [ ] Add automated builds, tests, and CI
- [ ] Document performance results and tradeoffs

## Current limitations

- Public construction currently accepts a single scalar; although values are stored in vectors, there is no public multidimensional tensor constructor.
- Addition and multiplication operate element-wise, without operand-size validation.
- There is no matrix multiplication, broadcasting, reduction, activation-function, or optimizer API yet.
- Gradients are internal and cannot currently be inspected or reset through the public interface.
- Repeated backpropagation through the same graph requires additional gradient-state handling.
- This project is experimental and not yet intended for production ML workloads.

## Motivation

cgrad is intended as a hands-on way to understand how automatic differentiation systems work below the level of a high-level machine learning library: how expressions become graphs, how gradients flow across shared dependencies, and how graph ownership, correctness, and performance interact in C++.

## References

- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd)
- [Danny Peelen - DannyTorch](https://github.com/dannypeelen/dannytorch)
