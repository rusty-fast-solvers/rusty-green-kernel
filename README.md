# Fast evaluation of Greens functions in Rust

This library allows the fast evaluation of Greens functions and potential sums
for Laplace, Helmholtz, and modified Helmholtz problems. The library can evaluate
sums of the form

f(x_i) = sum_j g(x_i, y_j) * c_j

and the corresponding gradients with respect to the x_i for g(x_i, y_j) defined as one of:

* Laplace Greens function: g(x, y) = 1 / (4 * pi * | x- y | )
* Helmholtz Greens function: g(x, y) = exp ( 1j * k * | x - y |) / (4 * pi * | x- y | )
* Modified Helmholtz Greens function: g(x, y) = exp( -omega * | x- y | ) / (4 * pi * | x- y | )

The implementation is optimised for the compiler to auto-vectorize with SIMD instruction sets.
Furthermore, all routines can make use of multithreading.

The library is implemented in Rust and provides a Rust API, C API, and Python bindings.

### Installation

To make sure that the library is compiled using modern processor features build it as follows.

```
export RUSTFLAGS="-C target-feature=+avx2,+fma" 
cargo build --release
```

After compiling as described above, the Python interface can be built with `maturin`, which is available
from Pypi and conda-forge.

To build the Python bindings make sure that the desired Python virtual environment is activated and
that the above `RUSTFLAGS` definition is set. Then build the Python module with

```
maturin develop --release -b cffi
```

This creates a new Python module called `rusty_green_kernel` and installs it.

### Documentation

The documentation of the Rust library and the C API is available at [docs.rs](http://docs.rs).
The documentation for the Python module is contained in the Python help for the module `rusty_green_kernel`.



