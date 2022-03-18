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
//! The Python module is called `rusty_green_kernel`.
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
//! For details of the Python module see the Python documentation in the `rusty_green_kernel` module.
//!

//pub mod kernels;
//pub mod evaluators;
//pub mod c_api;

//pub use evaluators::*;

pub use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
pub use ndarray_linalg::Scalar;

// Complex types
pub use ndarray_linalg::c32;
pub use ndarray_linalg::c64;

pub mod c_api;
pub(crate) mod laplace;

/// This enum defines the type of the kernel.
pub enum KernelType {
    /// The Laplace kernel defined as g(x, y) = 1 / (4 pi | x- y| )
    Laplace,
    /// The Helmholtz kernel defined as g(x, y) = exp( 1j * k * | x- y| ) / (4 pi | x- y| )
    Helmholtz(c64),
    /// The modified Helmholtz kernel defined as g(x, y) = exp( -omega * | x- y| ) / (4 * pi * | x- y |)
    ModifiedHelmholtz(f64),
}

/// This enum provides the keywords scalar and vectorial

pub enum DimensionType {
    /// For scalar kernels
    Scalar,

    /// For vectorial kernels
    Vectorial,
}

/// Return the dimension type (scalar or vectorial) for a kernel.
pub fn kernel_dimension(kernel_type: &KernelType) -> DimensionType {
    match kernel_type {
        KernelType::Laplace => DimensionType::Scalar,
        KernelType::Helmholtz(_) => DimensionType::Scalar,
        KernelType::ModifiedHelmholtz(_) => DimensionType::Scalar,
    }
}

/// Determines whether to use multithreading or serial evaluation.
pub enum ThreadingType {
    /// Use multithreading
    Parallel,
    /// Use serial evaluation
    Serial,
}

/// This enum defines the Evaluation Mode for kernels.
pub enum EvalMode {
    /// Only evaluate Green's function values.
    Value,
    /// Evaluate values and derivatives.
    ValueGrad,
}

/// Compute the number of scalar entries a single kernel evaluation requires.
pub(crate) fn get_evaluation_dimension(kernel_type: &KernelType, eval_mode: &EvalMode) -> usize {
    let dimension_type = kernel_dimension(&kernel_type);
    match eval_mode {
        EvalMode::Value => match dimension_type {
            DimensionType::Scalar => 1,
            DimensionType::Vectorial => 3,
        },
        EvalMode::ValueGrad => match dimension_type {
            DimensionType::Scalar => 4,
            DimensionType::Vectorial => {
                panic!("Only EvalMode::Value allowed as evaluation mode for vectorial kernels.")
            }
        },
    }
}

pub trait EvaluateKernel: Scalar {
    fn assemble_kernel(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        kernel_type: KernelType,
        threading_type: ThreadingType,
    ) -> Array2<Self> {
        let mut result = unsafe {
            Array2::<Self>::uninit((targets.len_of(Axis(1)), sources.len_of(Axis(1)))).assume_init()
        };

        Self::assemble_kernel_in_place(
            sources,
            targets,
            result.view_mut(),
            kernel_type,
            threading_type,
        );
        result
    }

    fn assemble_kernel_in_place(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        kernel_type: KernelType,
        threading_type: ThreadingType,
    );

    fn evaluate_kernel(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        charges: ArrayView2<Self>,
        kernel_type: KernelType,
        eval_mode: EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Self> {
        let nvalues = get_evaluation_dimension(&kernel_type, &eval_mode);
        // Use unsafe operation here as array will be filled with values during in place
        // evaluation. So avoid initializing twice.

        let mut result = unsafe {
            Array3::<Self>::uninit((charges.len_of(Axis(0)), targets.len_of(Axis(1)), nvalues))
                .assume_init()
        };

        Self::evaluate_kernel_in_place(
            sources,
            targets,
            charges,
            result.view_mut(),
            kernel_type,
            eval_mode,
            threading_type,
        );
        result
    }

    fn evaluate_kernel_in_place(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        charges: ArrayView2<Self>,
        result: ArrayViewMut3<Self>,
        kernel_type: KernelType,
        eval_mode: EvalMode,
        threading_type: ThreadingType,
    );
}

macro_rules! evaluate_kernel_impl {
    (f32) => {
        evaluate_kernel_impl!(f32, f32);
    };
    (f64) => {
        evaluate_kernel_impl!(f64, f64);
    };
    (c32) => {
        evaluate_kernel_impl!(f32, c32);
    };
    (c64) => {
        evaluate_kernel_impl!(f64, c64);
    };

    ($real:ty, $result:ty) => {
        impl EvaluateKernel for $result {
            fn assemble_kernel_in_place(
                sources: ArrayView2<<Self as Scalar>::Real>,
                targets: ArrayView2<<Self as Scalar>::Real>,
                result: ArrayViewMut2<Self>,
                kernel_type: KernelType,
                threading_type: ThreadingType,
            ) {
                let dimension_type = kernel_dimension(&kernel_type);

                let expected_shape = (targets.len_of(Axis(1)), sources.len_of(Axis(1)));
                let actual_shape = (result.len_of(Axis(0)), result.len_of(Axis(1)));

                assert!(
                    expected_shape == actual_shape,
                    "result array must have shape {:#?} but actual shape is {:#?}",
                    expected_shape,
                    actual_shape
                );

                match dimension_type {
                    DimensionType::Vectorial => {
                        panic!("Assembly of kernel matrices only allowed for scalar kernels.")
                    }
                    DimensionType::Scalar => (),
                }
            }

            fn evaluate_kernel_in_place(
                sources: ArrayView2<$real>,
                targets: ArrayView2<$real>,
                charges: ArrayView2<$result>,
                result: ArrayViewMut3<$result>,
                kernel_type: KernelType,
                eval_mode: EvalMode,
                threading_type: ThreadingType,
            ) {
                use crate::laplace::LaplaceEvaluator;

                let nvalues = get_evaluation_dimension(&kernel_type, &eval_mode);

                assert!(
                    sources.len_of(Axis(1)) == charges.len_of(Axis(1)),
                    "Charges and sources must have same length."
                );
                let expected_shape = (charges.len_of(Axis(0)), targets.len_of(Axis(1)), nvalues);
                let actual_shape = result.shape();
                let actual_shape = (actual_shape[0], actual_shape[1], actual_shape[2]);
                assert!(
                    expected_shape == actual_shape,
                    "Result has shape {:#?}, but expected shape is {:#?}",
                    actual_shape,
                    expected_shape
                );

                match kernel_type {
                    KernelType::Laplace => <$result>::evaluate_in_place_laplace(
                        sources,
                        targets,
                        charges,
                        result,
                        &eval_mode,
                        &threading_type,
                    ),
                    _ => panic!("Not implemented."),
                }
            }
        }
    };
}

evaluate_kernel_impl!(f32);
evaluate_kernel_impl!(f64);
evaluate_kernel_impl!(c32);
evaluate_kernel_impl!(c64);
