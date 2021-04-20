//! Defines the basic `Evaluator` type. This is a struct that contains the particle Space and define the
//! underlying Greens function.


use rusty_kernel_tools::{KernelType, RealType, ThreadingType};
use rusty_kernel_tools::ParticleContainerAccessor;

use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use num::complex::Complex;

use crate::kernels::EvalMode;

/// This type defines an Evaluator consisting of a
/// `ParticleSpace` and a `KernelType`. The generic
pub struct DirectEvaluator<P: ParticleContainerAccessor, R> {
    pub(crate) kernel_type: KernelType,
    pub(crate) particle_container: P,
    pub(crate) _marker: std::marker::PhantomData<R>,
}

pub trait DirectEvaluatorAccessor {
    type FloatingPointType: RealType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType;

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType>;
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType>;

    // Return number of sources
    fn nsources(&self) -> usize;

    // Return number of targets;
    fn ntargets(&self) -> usize;
}

pub trait RealDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place.
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    );

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Self::FloatingPointType>,
        result: ArrayViewMut3<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    );

    /// Assemble the kernel matrix and return it.
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::FloatingPointType>;

    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView2<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Self::FloatingPointType>;
}

pub trait ComplexDirectEvaluator: DirectEvaluatorAccessor {
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<num::complex::Complex<Self::FloatingPointType>>,
        threading_type: ThreadingType,
    );

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Complex<Self::FloatingPointType>>,
        result: ArrayViewMut3<Complex<Self::FloatingPointType>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    );

    /// Assemble the kernel matrix and return it
    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::FloatingPointType>>;

    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView2<Complex<Self::FloatingPointType>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Complex<Self::FloatingPointType>>;
}

impl<P: ParticleContainerAccessor, R> DirectEvaluatorAccessor for DirectEvaluator<P, R> {
    type FloatingPointType = P::FloatingPointType;

    /// Get the kernel definition.
    fn kernel_type(&self) -> &KernelType {
        &self.kernel_type
    }

    /// Return a non-owning representation of the sources.
    fn sources(&self) -> ArrayView2<Self::FloatingPointType> {
        self.particle_container.sources()
    }
    /// Return a non-owning representation of the targets.
    fn targets(&self) -> ArrayView2<Self::FloatingPointType> {
        self.particle_container.targets()
    }

    // Return number of sources.
    fn nsources(&self) -> usize {
        self.sources().len_of(Axis(1))
    }

    fn ntargets(&self) -> usize {
        self.targets().len_of(Axis(1))
    }
}

impl<P: ParticleContainerAccessor> RealDirectEvaluator
    for DirectEvaluator<P, P::FloatingPointType>
{
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<Self::FloatingPointType>,
        threading_type: ThreadingType,
    ) {
        use super::laplace::assemble_in_place_impl_laplace;
        match self.kernel_type {
            KernelType::Laplace => assemble_in_place_impl_laplace::<Self::FloatingPointType>(
                self.sources(),
                self.targets(),
                result,
                threading_type,
            ),
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }

    /// Assemble the kernel matrix and return it
    fn assemble(&self, threading_type: ThreadingType) -> Array2<Self::FloatingPointType> {
        let mut result =
            Array2::<Self::FloatingPointType>::zeros((self.ntargets(), self.nsources()));

        self.assemble_in_place(result.view_mut(), threading_type);
        result
    }

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Self::FloatingPointType>,
        result: ArrayViewMut3<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) {
        use super::laplace::evaluate_in_place_impl_laplace;
        match self.kernel_type {
            KernelType::Laplace => evaluate_in_place_impl_laplace(
                self.sources(),
                self.targets(),
                charges,
                result,
                eval_mode,
                threading_type,
            ),
            _ => panic!("Kernel not implemented for this evaluator."),
        }
    }
    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView2<Self::FloatingPointType>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Self::FloatingPointType> {
        let chunks = match eval_mode {
            EvalMode::Value => 1,
            EvalMode::ValueGrad => 4,
        };

        let ncharge_vecs = charges.len_of(Axis(1));

        let mut result =
            Array3::<Self::FloatingPointType>::zeros((ncharge_vecs, chunks, self.ntargets()));
        self.evaluate_in_place(charges, result.view_mut(), eval_mode, threading_type);
        result
    }
}

impl<P: ParticleContainerAccessor> ComplexDirectEvaluator
    for DirectEvaluator<P, num::complex::Complex<P::FloatingPointType>>
{
    /// Assemble the kernel matrix in-place
    fn assemble_in_place(
        &self,
        result: ArrayViewMut2<num::complex::Complex<Self::FloatingPointType>>,
        threading_type: ThreadingType,
    ) {
        use super::helmholtz::assemble_in_place_impl_helmholtz;
        match self.kernel_type {
            KernelType::Helmholtz(wavenumber) => {
                assemble_in_place_impl_helmholtz::<Self::FloatingPointType>(
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

    /// Evaluate for a set of charges in-pace.
    fn evaluate_in_place(
        &self,
        charges: ArrayView2<Complex<Self::FloatingPointType>>,
        result: ArrayViewMut3<Complex<Self::FloatingPointType>>,
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

    /// Assemble the kernel matrix and return it
    fn assemble(
        &self,
        threading_type: ThreadingType,
    ) -> Array2<num::complex::Complex<Self::FloatingPointType>> {
        let mut result = Array2::<num::complex::Complex<Self::FloatingPointType>>::zeros((
            self.nsources(),
            self.ntargets(),
        ));

        self.assemble_in_place(result.view_mut(), threading_type);
        result
    }

    /// Evaluate the kernel for a set of charges.
    fn evaluate(
        &self,
        charges: ArrayView2<Complex<Self::FloatingPointType>>,
        eval_mode: &EvalMode,
        threading_type: ThreadingType,
    ) -> Array3<Complex<Self::FloatingPointType>> {
        let chunks = match eval_mode {
            EvalMode::Value => 1,
            EvalMode::ValueGrad => 4,
        };

        let ncharge_vecs = charges.len_of(Axis(1));

        let mut result = Array3::<Complex<Self::FloatingPointType>>::zeros((
            ncharge_vecs,
            chunks,
            self.ntargets(),
        ));
        self.evaluate_in_place(charges, result.view_mut(), eval_mode, threading_type);
        result
    }
}
