pub mod base;
mod helmholtz;
mod laplace;
mod modified_helmholtz;

pub use base::DirectEvaluatorAccessor;
pub use base::ComplexDirectEvaluator;
pub use base::RealDirectEvaluator;
pub use base::DirectEvaluator;


use ndarray::{ArrayView2, Array2};
use num::complex::Complex;
use rusty_base::ParticleContainerView;
use rusty_base::ParticleContainer;
use rusty_base::RealType;
use rusty_base::KernelType;
use rusty_base::make_particle_container;
use rusty_base::make_particle_container_owned;



/// Make a Laplace evaluator from references to the data.
pub fn make_laplace_evaluator<'a, T: RealType>(
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
) -> DirectEvaluator<ParticleContainerView<'a, T>, T> {
    DirectEvaluator::<ParticleContainerView<'a, T>, T> {
        kernel_type: KernelType::Laplace,
        particle_container: make_particle_container(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}

/// Make a Laplace evaluator by taking ownership of the data.
pub fn make_laplace_evaluator_owned<T: RealType>(
    sources: Array2<T>,
    targets: Array2<T>,
) -> DirectEvaluator<ParticleContainer<T>, T> {
    DirectEvaluator::<ParticleContainer<T>, T> {
        kernel_type: KernelType::Laplace,
        particle_container: make_particle_container_owned(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}

/// Make a modified Helmholtz evaluator from references to the data.
pub fn make_modified_helmholtz_evaluator<'a, T: RealType>(
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
    omega: f64,
) -> DirectEvaluator<ParticleContainerView<'a, T>, T> {
    DirectEvaluator::<ParticleContainerView<'a, T>, T> {
        kernel_type: KernelType::ModifiedHelmholtz(omega),
        particle_container: make_particle_container(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}

/// Make a modified Helmholtz evaluator by taking ownership of the data.
pub fn make_modified_helmholtz_evaluator_owned<T: RealType>(
    sources: Array2<T>,
    targets: Array2<T>,
    omega: f64,
) -> DirectEvaluator<ParticleContainer<T>, T> {
    DirectEvaluator::<ParticleContainer<T>, T> {
        kernel_type: KernelType::ModifiedHelmholtz(omega),
        particle_container: make_particle_container_owned(sources, targets),
        _marker: std::marker::PhantomData::<T>,
    }
}



/// Make a Helmholtz evaluator from references to the data.
pub fn make_helmholtz_evaluator<'a, T: RealType>(
    sources: ArrayView2<'a, T>,
    targets: ArrayView2<'a, T>,
    wavenumber: Complex<f64>,
) -> DirectEvaluator<ParticleContainerView<'a, T>, num::complex::Complex<T>> {
    DirectEvaluator::<ParticleContainerView<'a, T>, num::complex::Complex<T>> {
        kernel_type: KernelType::Helmholtz(wavenumber),
        particle_container: make_particle_container(sources, targets),
        _marker: std::marker::PhantomData::<num::complex::Complex<T>>,
    }
}

/// Make a Helmholtz evaluator by taking ownership of the data.
pub fn make_helmholtz_evaluator_owned<T: RealType>(
    sources: Array2<T>,
    targets: Array2<T>,
    wavenumber: Complex<f64>,
) -> DirectEvaluator<ParticleContainer<T>, Complex<T>> {
    DirectEvaluator::<ParticleContainer<T>, Complex<T>> {
        kernel_type: KernelType::Helmholtz(wavenumber),
        particle_container: make_particle_container_owned(sources, targets),
        _marker: std::marker::PhantomData::<Complex<T>>,
    }
}


