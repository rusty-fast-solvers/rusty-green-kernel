//! Welcome to `rusty-green-kernel`. This crate contains routine for the evaluation of sums of the form
//! 
//! $$f(\mathbf{x}_i) = \sum_jg(\mathbf{x}_i, \mathbf{y}_j)c_j$$
//! 
//! and the corresponding gradients
//! 
//! $$\nabla_{\mathbf{x}}f(\mathbf{x}_i) = \sum_j\nabla\_{\mathbf{x}}g(\mathbf{x}_i, \mathbf{y}_j)c_j.$$
//|
//! The following kernels are supported.
//! 
//! * The Laplace kernel: $g(\mathbf{x}, \mathbf{y}) = \frac{1}{4\pi|\mathbf{x} - \mathbf{y}|}$.
//! * The Helmholtz kernel: $g(\mathbf{x}, \mathbf{y}) = \frac{e^{ik|\mathbf{x} - \mathbf{y}|}}{4\pi|\mathbf{x} - \mathbf{y}|}$
//! * The modified Helmholtz kernel:$g(\mathbf{x}, \mathbf{y}) = \frac{e^{-\omega|\mathbf{x} - \mathbf{y}|}}{4\pi|\mathbf{x} - \mathbf{y}|}$
//! 
//! Within the library the $\mathbf{x}_i$ are named `targets` and the $\mathbf{y}_j$ are named `sources`. We use
//! the convention that $g(\mathbf{x}_i, \mathbf{y}_j) := 0$, whenever $\mathbf{x}_i = \mathbf{y}_j$.
//! 
//! The library provides a Rust API, C API, and Python API.
//! 
//! ### Installation hints
//! 
//! The performance of the library strongly depends on being compiled with the right parameters for the underlying CPU. Almost any modern CPU
//! supports AVX2 and FMA. To activate these features compile with
//! 
//! ```
//! export RUSTFLAGS="-C target-feature=+avx2,+fma" 
//! cargo build --release
//! ```
//! 
//! The activated compiler features can also be tested with `cargo rustc -- --print cfg`.
//! 
//! To compile and install the Python module make sure that the wanted Python virtual environment is active.
//! The installation is performed using `maturin`, which is available from Pypi and conda-forge.
//! 
//! After compiling the library as described above use 
//! 
//! ``
//! maturin develop --release -b cffi
//! `` 
//! 
//! to compile and install the Python module. It is important that the `RUSTFLAGS` environment variable is set as stated above.
//! The Python module is called `rusty-green-kernel`.
//! 
//! ### Rust API
//! 
//! The `sources` and `targets` are both arrays of type `ndarray<T>` with `T=f32` or `T=f64`. For `M` targets and `N` sources
//! the `sources` are a `(3, N)` array and the `targets` are a `(3, M)` array. 
//! 
//! To evaluate the kernel matrix of all interactions between a vector of `sources` and a `vector` of targets for the Laplace kernel
//! use
//! 
//! ```kernel_matrix = make_laplace_evaluator(sources, targets).assemble()```
//! 
//! To evaluate $f(\mathbf{x}_i) = \sum_jg(\mathbf{x}_i, \mathbf{y}_j)c_j$ we define the charges as `ndarray` of
//! size `(ncharge_vecs, nsources)`, where `ncharge_vecs` is the number of charge vectors we want to evaluate and
//! `nsources` is the number of sources. For Laplace and modified Helmholtz problems `charges` must be of type `f32`
//! or `f64` and for Helmholtz problems it must be of type `Complex<f32>` or `Complex<f64>`.
//! 
//! We can then evaluate the potential sum by
//! 
//! ```rust
//! potential_sum = make_laplace_evaluator(sources, targets).evaluate(
//!         charges, EvalMode::Values, EvalMode::Value, ThreadingType::Parallel)
//! ```
//! 
//! The result `potential_sum` is a real `ndarray` (for Laplace and modified Helmholtz) or a complex `ndarray` (for Helmholtz).
//! It has the shape `(ncharge_vecs, ntargets, 1)`. For `EvalMode::Value` the function only computes the values $f(\mathbf{x}_i)$. For
//! `EvalMode::ValueGrad` the array `potential_sum` is of shape `(ncharge_vecs, ntargets, 4)` and
//! returns the function values and the three components of the gradient along the most-inner dimension. The value
//! `ThreadingType::Parallel` specifies that the evaluation is multithreaded. For this the Rayon library is used. For the
//! value `ThreadingType::Serial` the code is executed single-threaded. The enum `ThreadingType` is defined in the
//! crate `rusty-kernel-tools`.
//! 
//! Basic access to `sources` and `targets` is provided through the trait [`DirectEvaluatorAccessor`], which is implemented by
//! the struct [`DirectEvaluator`]. The Helmholtz kernel uses the trait [`ComplexDirectEvaluator`] and the Laplace and modified
//! Helmholtz kernels use the trait [`RealDirectEvaluator`].
//! 
//! ### C API
//! 
//! The C API in [`c_api`] provides direct access to the functionality in a C compatible interface. All functions come in variants
//! for `f32` and `f64` types. Details are explaineed in the documentation of the corresponding functions.
//! 
//! ### Python API
//! 
//! For details of the Python module see the Python documentation in the `rusty-green-kernel` module.
//! 

pub mod kernels;
pub mod evaluators;
pub mod c_api;

pub use evaluators::*;
pub use kernels::EvalMode;
