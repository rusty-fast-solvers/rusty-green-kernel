//! Defines the basic `Evaluator` type. This is a struct that contains the particle Space and define the
//! underlying Greens function.


use rusty_base::{EvalMode, KernelType, RealType, ThreadingType};
use rusty_base::ParticleContainerAccessor;

use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use num::complex::Complex;

/// This type defines an Evaluator consisting of a
/// `ParticleSpace` and a `KernelType`. The generic
pub struct DirectEvaluator<P: ParticleContainerAccessor, R> {
    pub(crate) kernel_type: KernelType,
    pub(crate) particle_container: P,
    pub(crate) _marker: std::marker::PhantomData<R>,
}

/// Basic access to the data that defines a Greens function kernel,
/// its sources and targets.
pub trait DirectEvaluatorAccessor {
    type A: RealType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::A>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::A>;

    // Return number of sources
    fn nsources(&self) -> usize;

    // Return number of targets;
    fn ntargets(&self) -> usize;
}

/// Assemblers and evaluators for real kernels.
pub trait RealDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place.
    /// 
    /// # Arguments
    /// * `result` - A real array of dimension `(ntargets, nsources)`
    ///              that contains the Green's function evaluations between
    ///              sources and targets. Note. The array must already have the right shape upon
    ///              calling the function.
    /// * `threading_type` - Determines whether the routine should use multithreading
    ///                      `ThreadingType::Parallel` or serial exectution `ThreadingType::Serial`.
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::A>,
        threading_type: ThreadingType,
    );

    /// Evaluate for each target the potential sum across all sources with given charges.
    /// 
    /// # Arguments
    /// * `charges` - A real array of dimension `(ncharge_vecs, nsources)` that contains
    ///               `ncharge_vec` vectors of charges in the rows, each of which has `nsources` entries.
    /// * `result` - A real array of shape `(ncharge_vecs, ntargets, 1)` if only Greens fct. values are
    ///              requested or  of shape `(ncharge_vecs, ntargets, 4)` if function values and gradients
    ///              are requested. The value `result[i][j][0]` contains the potential sum evaluated at
    ///              the jth target, using the ith charge vector. The values `result[i][j][k]` for k=1,..,3
    ///              contain the corresponding gradient in the x, y, and z coordinate direction.
    /// `eval_mode` - Either [`EvalMode::Value`] to only return function values or [`EvalMode::ValueGrad`] to return
    ///               function values and derivatives.
    /// * `threading_type` - Either `ThreadingType::Parallel` for parallel execution or `ThreadingType::Serial` for
    ///                      serial execution. The enum `ThreadingType` is defined in the package `rusty-kernel-tools`.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Self::A>,
        result: ArrayViewMut3<Self::A>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    );

    /// Like `assemble_in_place`, but creates and returns a new result array.
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::A>;

    /// Like `evaluate_in_place` but creates and returns a new result array.
    fn evaluate(
        &self,
        charges: ArrayView2<Self::A>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Self::A>;
}

/// Assemblers and evaluators for complex kernels.
pub trait ComplexDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place.
    /// 
    /// # Arguments
    /// * `result` - A complex array of dimension `(ntargets, nsources)`
    ///              that contains the Green's function evaluations between
    ///              sources and targets. Note. The array must already have the right shape upon
    ///              calling the function.
    /// * `threading_type` - Determines whether the routine should use multithreading
    ///                      `ThreadingType::Parallel` or serial exectution `ThreadingType::Serial`.
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<num::complex::Complex<Self::A>>,
        threading_type: ThreadingType,
    );

    /// Evaluate for each target the potential sum across all sources with given charges.
    /// 
    /// # Arguments
    /// * `charges` - A complex array of dimension `(ncharge_vecs, nsources)` that contains
    ///               `ncharge_vec` vectors of charges in the rows, each of which has `nsources` entries.
    /// * `result` - A complex array of shape `(ncharge_vecs, ntargets, 1)` if only Greens fct. values are
    ///              requested or  of shape `(ncharge_vecs, ntargets, 4)` if function values and gradients
    ///              are requested. The value `result[i][j][0]` contains the potential sum evaluated at
    ///              the jth target, using the ith charge vector. The values `result[i][j][k]` for k=1,..,3
    ///              contain the corresponding gradient in the x, y, and z coordinate direction.
    /// `eval_mode` - Either [`EvalMode::Value`] to only return function values or [`EvalMode::ValueGrad`] to return
    ///               function values and derivatives.
    /// * `threading_type` - Either `ThreadingType::Parallel` for parallel execution or `ThreadingType::Serial` for
    ///                      serial execution. The enum `ThreadingType` is defined in the package `rusty-kernel-tools`.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Complex<Self::A>>,
        result: ArrayViewMut3<Complex<Self::A>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    );

    /// Like `assemble_in_place`, but creates and returns a new result array.
    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::A>>;

    /// Like `evaluate_in_place` but creates and returns a new result array.
    fn evaluate(
        &self,
        charges: ArrayView2<Complex<Self::A>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Complex<Self::A>>;
}

impl<P: ParticleContainerAccessor, R> DirectEvaluatorAccessor for DirectEvaluator<P, R> {
    type A = P::A;

    fn kernel_type(&self) -> &KernelType {
        &self.kernel_type
    }

    fn sources(&self) -> ArrayView2<Self::A> {
        self.particle_container.sources()
    }

    fn targets(&self) -> ArrayView2<Self::A> {
        self.particle_container.targets()
    }

    fn nsources(&self) -> usize {
        self.sources().len_of(Axis(1))
    }

    fn ntargets(&self) -> usize {
        self.targets().len_of(Axis(1))
    }
}

impl<P: ParticleContainerAccessor> RealDirectEvaluator
    for DirectEvaluator<P, P::A>
{
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::A>,
        threading_type: ThreadingType,
    ) {
        use super::laplace::assemble_in_place_impl_laplace;
        use super::modified_helmholtz::assemble_in_place_impl_modified_helmholtz;
        match self.kernel_type {
            KernelType::Laplace => assemble_in_place_impl_laplace::<Self::A>(
                self.sources(),
                self.targets(),
                result,
                threading_type,
            ),
            KernelType::ModifiedHelmholtz(omega) => assemble_in_place_impl_modified_helmholtz::<Self::A>(
                self.sources(),
                self.targets(),
                result,
                omega,
                threading_type,
            ),
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::A> {
        let mut result =
            Array2::<Self::A>::zeros((self.ntargets(), self.nsources()));

        self.assemble_in_place(result.view_mut(), threading_type);
        result
    }

    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Self::A>,
        result: ArrayViewMut3<Self::A>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) {
        use super::laplace::evaluate_in_place_impl_laplace;
        use super::modified_helmholtz::evaluate_in_place_impl_modified_helmholtz;
        match self.kernel_type {
            KernelType::Laplace => evaluate_in_place_impl_laplace(
                self.sources(),
                self.targets(),
                charges,
                result,
                eval_mode,
                threading_type,
            ),
            KernelType::ModifiedHelmholtz(omega) => evaluate_in_place_impl_modified_helmholtz::<Self::A>(
                self.sources(),
                self.targets(),
                charges,
                result,
                omega,
                eval_mode,
                threading_type,
            ),

            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    fn evaluate(
        &self,
        charges: ArrayView2<Self::A>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Self::A> {
        let chunks = match eval_mode {
            EvalMode::Value => 1,
            EvalMode::ValueGrad => 4,
        };

        let ncharge_vecs = charges.len_of(Axis(1));

        let mut result =
            Array3::<Self::A>::zeros((ncharge_vecs, chunks, self.ntargets()));
        self.evaluate_in_place(charges, result.view_mut(), eval_mode, threading_type);
        result
    }
}

impl<P: ParticleContainerAccessor> ComplexDirectEvaluator
    for DirectEvaluator<P, num::complex::Complex<P::A>>
{
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<num::complex::Complex<Self::A>>,
        threading_type: ThreadingType,
    ) {
        use super::helmholtz::assemble_in_place_impl_helmholtz;
        match self.kernel_type {
            KernelType::Helmholtz(wavenumber) => {
                assemble_in_place_impl_helmholtz::<Self::A>(
                    self.sources(),
                    self.targets(),
                    result,
                    wavenumber,
                    threading_type,
                )
            }
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Complex<Self::A>>,
        result: ArrayViewMut3<Complex<Self::A>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) {
        use super::helmholtz::evaluate_in_place_impl_helmholtz;
        match self.kernel_type {
            KernelType::Helmholtz(wavenumber) => evaluate_in_place_impl_helmholtz(
                self.sources(),
                self.targets(),
                charges,
                result,
                wavenumber,
                eval_mode,
                threading_type,
            ),
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::A>> {
        let mut result = Array2::<num::complex::Complex<Self::A>>::zeros((
            self.nsources(),
            self.ntargets(),
        ));

        self.assemble_in_place(result.view_mut(), threading_type);
        result
    }

    fn evaluate(
        &self,
        charges: ArrayView2<Complex<Self::A>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Complex<Self::A>> {
        let chunks = match eval_mode {
            EvalMode::Value => 1,
            EvalMode::ValueGrad => 4,
        };

        let ncharge_vecs = charges.len_of(Axis(1));

        let mut result = Array3::<Complex<Self::A>>::zeros((
            ncharge_vecs,
            chunks,
            self.ntargets(),
        ));
        self.evaluate_in_place(charges, result.view_mut(), eval_mode, threading_type);
        result
    }
}
