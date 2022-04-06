use crate::create_pool;
use crate::EvalMode;
/// Implementation of assembler function for Laplace kernels.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use ndarray_linalg::{c32, c64, Scalar};
use num::traits::FloatConst;

pub(crate) trait HelmholtzEvaluator: Scalar {
    #[allow(unused_variables)]
    fn helmholtz_kernel(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result_real: ArrayViewMut2<<Self as Scalar>::Real>,
        result_imag: ArrayViewMut2<<Self as Scalar>::Real>,
        wavenumber: c64,
        eval_mode: &EvalMode,
    ) {
        panic!("Not implemented for this type.");
    }

    /// Implementation of the Helmholtz kernel without derivatives.
    #[allow(unused_variables)]
    fn helmholtz_kernel_no_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result_real: ArrayViewMut2<<Self as Scalar>::Real>,
        result_imag: ArrayViewMut2<<Self as Scalar>::Real>,
        wavenumber: c64,
    ) {
        panic!("Not implemented for this type");
    }

    // Implementation of the Helmholtz kernel with derivatives.
    #[allow(unused_variables)]
    fn helmholtz_kernel_with_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result_real: ArrayViewMut2<<Self as Scalar>::Real>,
        result_imag: ArrayViewMut2<<Self as Scalar>::Real>,
        wavenumber: c64,
    ) {
        panic!("Not implemented for this type.");
    }

    #[allow(unused_variables)]
    fn evaluate_in_place_helmholtz(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        charges: ArrayView2<Self>,
        result: ArrayViewMut3<Self>,
        wavenumber: c64,
        eval_mode: &EvalMode,
        num_threads: usize,
    ) {
        panic!("Not implemented for this type.");
    }

    #[allow(unused_variables)]
    fn assemble_in_place_helmholtz(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        wavenumber: c64,
        num_threads: usize,
    ) {
        panic!("Not implemented for this type.");
    }
}

macro_rules! helmholtz_impl {
    ($scalar:ty) => {
        impl HelmholtzEvaluator for $scalar {
            fn helmholtz_kernel(
                target: ArrayView1<<$scalar as Scalar>::Real>,
                sources: ArrayView2<<$scalar as Scalar>::Real>,
                result_real: ArrayViewMut2<<Self as Scalar>::Real>,
                result_imag: ArrayViewMut2<<Self as Scalar>::Real>,
                wavenumber: c64,
                eval_mode: &EvalMode,
            ) {
                {
                    match eval_mode {
                        EvalMode::Value => Self::helmholtz_kernel_no_deriv(
                            target,
                            sources,
                            result_real,
                            result_imag,
                            wavenumber,
                        ),
                        EvalMode::ValueGrad => Self::helmholtz_kernel_with_deriv(
                            target,
                            sources,
                            result_real,
                            result_imag,
                            wavenumber,
                        ),
                    };
                }
            }
            /// Implementation of the Helmholtz kernel without derivatives.
            fn helmholtz_kernel_no_deriv(
                target: ArrayView1<<$scalar as Scalar>::Real>,
                sources: ArrayView2<<$scalar as Scalar>::Real>,
                mut result_real: ArrayViewMut2<<$scalar as Scalar>::Real>,
                mut result_imag: ArrayViewMut2<<$scalar as Scalar>::Real>,
                wavenumber: c64,
            ) {
                use ndarray::Zip;

                type RealType = <$scalar as Scalar>::Real;

                let zero: <$scalar as Scalar>::Real = num::traits::zero();

                let m_inv_4pi: <$scalar as Scalar>::Real =
                    num::traits::cast::<f64, <$scalar as Scalar>::Real>(0.25).unwrap()
                        * <<$scalar as Scalar>::Real>::FRAC_1_PI();

                let wavenumber_real: RealType =
                    num::traits::cast::<f64, RealType>(wavenumber.re).unwrap();
                let wavenumber_imag: RealType =
                    num::traits::cast::<f64, RealType>(wavenumber.im).unwrap();

                let mut dist = Array1::<RealType>::zeros(sources.len_of(Axis(1)));

                result_real.fill(zero);
                result_imag.fill(zero);

                Zip::from(target).and(sources.axis_iter(Axis(0))).for_each(
                    |&target_value, source_row| {
                        Zip::from(source_row).and(dist.view_mut()).for_each(
                            |&source_value, dist_ref| {
                                *dist_ref += <<$scalar as Scalar>::Real>::powi(
                                    source_value - target_value,
                                    2,
                                )
                            },
                        )
                    },
                );

                dist.mapv_inplace(|item| <RealType>::sqrt(item));

                Zip::from(dist.view())
                    .and(result_real.index_axis_mut(Axis(0), 0))
                    .and(result_imag.index_axis_mut(Axis(0), 0))
                    .for_each(|&dist_val, result_real_val, result_imag_val| {
                        let exp_val = <RealType>::exp(-wavenumber_imag * dist_val);
                        *result_real_val =
                            exp_val * <RealType>::cos(wavenumber_real * dist_val) * m_inv_4pi
                                / dist_val;
                        *result_imag_val =
                            exp_val * <RealType>::sin(wavenumber_real * dist_val) * m_inv_4pi
                                / dist_val;
                    });

                Zip::from(dist.view())
                    .and(result_real.index_axis_mut(Axis(0), 0))
                    .and(result_imag.index_axis_mut(Axis(0), 0))
                    .for_each(|&dist_val, result_real_val, result_imag_val| {
                        if dist_val == zero {
                            *result_real_val = zero;
                            *result_imag_val = zero;
                        }
                    });
            }
            /// Implementation of the Helmholtz kernel with derivatives.
            fn helmholtz_kernel_with_deriv(
                target: ArrayView1<<$scalar as Scalar>::Real>,
                sources: ArrayView2<<$scalar as Scalar>::Real>,
                mut result_real: ArrayViewMut2<<$scalar as Scalar>::Real>,
                mut result_imag: ArrayViewMut2<<$scalar as Scalar>::Real>,
                wavenumber: c64,
            ) {
                use ndarray::Zip;

                type RealType = <$scalar as Scalar>::Real;

                let zero: RealType = num::traits::zero();
                let one: RealType = num::traits::one();

                let m_inv_4pi: RealType = num::traits::cast::<f64, RealType>(0.25).unwrap()
                    * RealType::FRAC_1_PI();

                let wavenumber_real: RealType =
                    num::traits::cast::<f64, RealType>(wavenumber.re).unwrap();
                let wavenumber_imag: RealType =
                    num::traits::cast::<f64, RealType>(wavenumber.im).unwrap();

                let mut dist = Array1::<RealType>::zeros(sources.len_of(Axis(1)));

                result_real.fill(zero);
                result_imag.fill(zero);

                Zip::from(target).and(sources.axis_iter(Axis(0))).for_each(
                    |&target_value, source_row| {
                        Zip::from(source_row).and(dist.view_mut()).for_each(
                            |&source_value, dist_ref| {
                                *dist_ref += <RealType>::powi(source_value - target_value, 2)
                            },
                        )
                    },
                );

                dist.mapv_inplace(|item| <RealType>::sqrt(item));

                Zip::from(dist.view())
                    .and(result_real.index_axis_mut(Axis(0), 0))
                    .and(result_imag.index_axis_mut(Axis(0), 0))
                    .for_each(|&dist_val, result_real_val, result_imag_val| {
                        let exp_val = <RealType>::exp(-wavenumber_imag * dist_val);
                        *result_real_val =
                            exp_val * <RealType>::cos(wavenumber_real * dist_val) * m_inv_4pi
                                / dist_val;
                        *result_imag_val =
                            exp_val * <RealType>::sin(wavenumber_real * dist_val) * m_inv_4pi
                                / dist_val;
                    });

                // Now do the derivative term

                let (values_real, mut derivs_real) = result_real.view_mut().split_at(Axis(0), 1);
                let (values_imag, mut derivs_imag) = result_imag.view_mut().split_at(Axis(0), 1);

                let values_real = values_real.index_axis(Axis(0), 0);
                let values_imag = values_imag.index_axis(Axis(0), 0);

                Zip::from(derivs_real.axis_iter_mut(Axis(0)))
                    .and(derivs_imag.axis_iter_mut(Axis(0)))
                    .and(target.view())
                    .and(sources.axis_iter(Axis(0)))
                    .for_each(
                        |deriv_real_row, deriv_imag_row, &target_value, source_row| {
                            Zip::from(deriv_real_row)
                                .and(deriv_imag_row)
                                .and(source_row)
                                .and(values_real)
                                .and(values_imag)
                                .and(dist.view())
                                .for_each(
                                    |deriv_real_value,
                                     deriv_imag_value,
                                     &source_value,
                                     &value_real,
                                     &value_imag,
                                     &dist_value| {
                                        *deriv_real_value = (target_value - source_value)
                                            / <RealType>::powi(dist_value, 2)
                                            * ((-one - wavenumber_imag * dist_value) * value_real
                                                - wavenumber_real * dist_value * value_imag);
                                        *deriv_imag_value = (target_value - source_value)
                                            / <RealType>::powi(dist_value, 2)
                                            * (value_real * wavenumber_real * dist_value
                                                + (-one - wavenumber_imag * dist_value)
                                                    * value_imag);
                                    },
                                )
                        },
                    );

                Zip::from(result_real.axis_iter_mut(Axis(0)))
                    .and(result_imag.axis_iter_mut(Axis(0)))
                    .for_each(|real_row, imag_row| {
                        Zip::from(dist.view()).and(real_row).and(imag_row).for_each(
                            |dist_elem, real_elem, imag_elem| {
                                if *dist_elem == zero {
                                    *real_elem = zero;
                                    *imag_elem = zero;
                                }
                            },
                        )
                    });
            }

            fn assemble_in_place_helmholtz(
                sources: ArrayView2<<$scalar as Scalar>::Real>,
                targets: ArrayView2<<$scalar as Scalar>::Real>,
                mut result: ArrayViewMut2<$scalar>,
                wavenumber: c64,
                num_threads: usize,
            ) {
                use ndarray::Zip;

                type RealType = <$scalar as Scalar>::Real;

                let nsources = sources.len_of(Axis(1));

                create_pool(num_threads).install(|| {
                    Zip::from(targets.axis_iter(Axis(1)))
                        .and(result.axis_iter_mut(Axis(0)))
                        .par_for_each(|target, mut result_row| {
                            let mut tmp_real = Array2::<RealType>::zeros((1, nsources));
                            let mut tmp_imag = Array2::<RealType>::zeros((1, nsources));
                            Self::helmholtz_kernel(
                                target,
                                sources,
                                tmp_real.view_mut(),
                                tmp_imag.view_mut(),
                                wavenumber,
                                &EvalMode::Value,
                            );
                            Zip::from(result_row.view_mut())
                                .and(tmp_real.index_axis(Axis(0), 0))
                                .and(tmp_imag.index_axis(Axis(0), 0))
                                .for_each(|result_elem, &tmp_real_elem, &tmp_imag_elem| {
                                    result_elem.re = tmp_real_elem;
                                    result_elem.im = tmp_imag_elem;
                                });
                        });
                });
            }

            fn evaluate_in_place_helmholtz(
                sources: ArrayView2<<$scalar as Scalar>::Real>,
                targets: ArrayView2<<$scalar as Scalar>::Real>,
                charges: ArrayView2<$scalar>,
                mut result: ArrayViewMut3<$scalar>,
                wavenumber: c64,
                eval_mode: &EvalMode,
                num_threads: usize,
            ) {
                use ndarray::Zip;

                type RealType = <$scalar as Scalar>::Real;

                let nsources = sources.len_of(Axis(1));

                let charges_real = charges.map(|item| item.re);
                let charges_imag = charges.map(|item| item.im);

                let chunks = match eval_mode {
                    EvalMode::Value => 1,
                    EvalMode::ValueGrad => 4,
                };

                result.fill(num::traits::zero());

                create_pool(num_threads).install(|| {
                    Zip::from(targets.axis_iter(Axis(1)))
            .and(result.axis_iter_mut(Axis(1)))
            .par_for_each(|target, mut result_block| {
                let mut tmp_real = Array2::<RealType>::zeros((chunks, nsources));
                let mut tmp_imag = Array2::<RealType>::zeros((chunks, nsources));
                Self::helmholtz_kernel(
                    target,
                    sources,
                    tmp_real.view_mut(),
                    tmp_imag.view_mut(),
                    wavenumber,
                    eval_mode,
                );
                Zip::from(charges_real.axis_iter(Axis(0)))
                    .and(charges_imag.axis_iter(Axis(0)))
                    .and(result_block.axis_iter_mut(Axis(0)))
                    .for_each(|charge_vec_real, charge_vec_imag, result_row| {
                        Zip::from(tmp_real.axis_iter(Axis(0)))
                            .and(tmp_imag.axis_iter(Axis(0)))
                            .and(result_row)
                            .for_each(|tmp_real_row, tmp_imag_row, result_elem| {
                                Zip::from(tmp_real_row)
                                    .and(tmp_imag_row)
                                    .and(charge_vec_real)
                                    .and(charge_vec_imag)
                                    .for_each(
                                        |tmp_elem_real,
                                         tmp_elem_imag,
                                         charge_elem_real,
                                         charge_elem_imag| {
                                            result_elem.re += *tmp_elem_real * *charge_elem_real
                                                - *tmp_elem_imag * *charge_elem_imag;
                                            result_elem.im += *tmp_elem_real * *charge_elem_imag
                                                + *tmp_elem_imag * *charge_elem_real;
                                        },
                                    )
                            })
                    })
            })

                });
            }
        }
    };
}

// Default implementations for unsupported types.
impl HelmholtzEvaluator for f32 {}
impl HelmholtzEvaluator for f64 {}

// Actual implementations for supported types.
helmholtz_impl!(c64);
helmholtz_impl!(c32);
