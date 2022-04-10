use crate::create_pool;
use crate::EvalMode;
/// Implementation of assembler function for Laplace kernels.
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use ndarray_linalg::{c32, c64, Scalar};
use num::traits::FloatConst;

pub(crate) trait ModifiedHelmholtzEvaluator: Scalar {
    #[allow(unused_variables)]
    fn modified_helmholtz_kernel(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        omega: f64,
        eval_mode: &EvalMode,
    ) {
        panic!("Not implemented for this type.");
    }

    /// Implementation of the Laplace kernel without derivatives.
    #[allow(unused_variables)]
    fn modified_helmholtz_kernel_no_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        omega: f64,
    ) {
        panic!("Not implemented for this type");
    }

    // Implementation of the Laplace kernel with derivatives.
    #[allow(unused_variables)]
    fn modified_helmholtz_kernel_with_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        omega: f64,
    ) {
        panic!("Not implemented for this type.");
    }

    #[allow(unused_variables)]
    fn evaluate_in_place_modified_helmholtz(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        charges: ArrayView2<Self>,
        result: ArrayViewMut3<Self>,
        omega: f64,
        eval_mode: &EvalMode,
        num_threads: usize,
    ) {
        panic!("Not implemented for this type.");
    }

    #[allow(unused_variables)]
    fn assemble_in_place_modified_helmholtz(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        omega: f64,
        num_threads: usize,
    ) {
        panic!("Not implemented for this type.");
    }
}

macro_rules! modified_helmholtz_impl {
    ($scalar:ty) => {
        impl ModifiedHelmholtzEvaluator for $scalar {
            fn modified_helmholtz_kernel(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                result: ArrayViewMut2<$scalar>,
                omega: f64,
                eval_mode: &EvalMode,
            ) {
                {
                    match eval_mode {
                        EvalMode::Value => {
                            Self::modified_helmholtz_kernel_no_deriv(target, sources, result, omega)
                        }
                        EvalMode::ValueGrad => Self::modified_helmholtz_kernel_with_deriv(
                            target, sources, result, omega,
                        ),
                    };
                }
            }
            /// Implementation of the Laplace kernel without derivatives.
            fn modified_helmholtz_kernel_no_deriv(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                mut result: ArrayViewMut2<$scalar>,
                omega: f64,
            ) {
                use ndarray::Zip;

                let zero: $scalar = num::traits::zero();

                let m_inv_4pi: $scalar =
                    num::traits::cast::cast::<f64, $scalar>(0.25).unwrap() * <$scalar>::FRAC_1_PI();

                let omega: $scalar = num::traits::cast::cast::<f64, $scalar>(omega).unwrap();

                result.fill(zero);

                Zip::from(target).and(sources.axis_iter(Axis(0))).for_each(
                    |&target_value, source_row| {
                        Zip::from(source_row)
                            .and(result.index_axis_mut(Axis(0), 0))
                            .for_each(|&source_value, result_ref| {
                                *result_ref +=
                                    (target_value - source_value) * (target_value - source_value)
                            })
                    },
                );

                result
                    .index_axis_mut(Axis(0), 0)
                    .map_inplace(|item| *item = <$scalar>::sqrt(*item));

                result
                    .index_axis_mut(Axis(0), 0)
                    .map_inplace(|item| *item = <$scalar>::exp(-omega * *item) * m_inv_4pi / *item);
                result
                    .index_axis_mut(Axis(0), 0)
                    .iter_mut()
                    .filter(|item| !item.is_finite())
                    .for_each(|item| *item = zero);
            }

            /// Implementation of the Laplace kernel with derivatives.
            fn modified_helmholtz_kernel_with_deriv(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                mut result: ArrayViewMut2<$scalar>,
                omega: f64,
            ) {
                use ndarray::Zip;

                let zero: $scalar = num::traits::zero();
                let one: $scalar = num::traits::one();

                let m_inv_4pi: $scalar =
                    num::traits::cast::<f64, $scalar>(0.25).unwrap() * <$scalar>::FRAC_1_PI();

                let omega: $scalar = num::traits::cast::cast::<f64, $scalar>(omega).unwrap();

                result.fill(zero);

                let mut dist = ndarray::Array1::<$scalar>::zeros(sources.len_of(Axis(1)));

                // First compute the Green fct. values

                Zip::from(target).and(sources.axis_iter(Axis(0))).for_each(
                    |&target_value, source_row| {
                        Zip::from(source_row).and(dist.view_mut()).for_each(
                            |&source_value, dist_ref| {
                                *dist_ref +=
                                    (target_value - source_value) * (target_value - source_value)
                            },
                        )
                    },
                );

                dist.map_inplace(|item| *item = <$scalar>::sqrt(*item));

                Zip::from(result.index_axis_mut(Axis(0), 0))
                    .and(dist.view())
                    .for_each(|result_ref, &dist_value| {
                        *result_ref = <$scalar>::exp(-omega * dist_value) * m_inv_4pi / dist_value
                    });

                // Now compute the derivatives.

                let (values, mut derivs) = result.view_mut().split_at(Axis(0), 1);
                let values = values.index_axis(Axis(0), 0);

                Zip::from(derivs.axis_iter_mut(Axis(0)))
                    .and(target.view())
                    .and(sources.axis_iter(Axis(0)))
                    .for_each(|deriv_row, &target_value, source_row| {
                        Zip::from(deriv_row)
                            .and(source_row)
                            .and(values)
                            .and(dist.view())
                            .for_each(|deriv_value, &source_value, &value, &dist_value| {
                                *deriv_value = value * (target_value - source_value)
                                    / <$scalar>::powi(dist_value, 2)
                                    * (-omega * dist_value - one)
                            })
                    });

                result.view_mut().axis_iter_mut(Axis(0)).for_each(|row| {
                    Zip::from(row)
                        .and(dist.view())
                        .for_each(|elem, &dist_value| {
                            if dist_value == zero {
                                *elem = zero;
                            }
                        })
                });
            }

            fn assemble_in_place_modified_helmholtz(
                sources: ArrayView2<<Self as Scalar>::Real>,
                targets: ArrayView2<<Self as Scalar>::Real>,
                mut result: ArrayViewMut2<Self>,
                omega: f64,
                num_threads: usize,
            ) {
                use ndarray::Zip;

                let nsources = sources.len_of(Axis(1));

                create_pool(num_threads).install(|| {
                    Zip::from(targets.axis_iter(Axis(1)))
                        .and(result.axis_iter_mut(Axis(0)))
                        .par_for_each(|target, result_row| {
                            let tmp = result_row
                                .into_shape((1, nsources))
                                .expect("Cannot convert to 2-dimensional array.");
                            Self::modified_helmholtz_kernel(
                                target,
                                sources,
                                tmp,
                                omega,
                                &EvalMode::Value,
                            );
                        })
                });
            }

            fn evaluate_in_place_modified_helmholtz(
                sources: ArrayView2<$scalar>,
                targets: ArrayView2<$scalar>,
                charges: ArrayView2<$scalar>,
                mut result: ArrayViewMut3<$scalar>,
                omega: f64,
                eval_mode: &EvalMode,
                num_threads: usize,
            ) {
                use ndarray::Zip;

                let nsources = sources.len_of(Axis(1));

                let chunks = match eval_mode {
                    EvalMode::Value => 1,
                    EvalMode::ValueGrad => 4,
                };

                result.fill(num::traits::zero());

                create_pool(num_threads).install(|| {
                    Zip::from(targets.axis_iter(Axis(1)))
                        .and(result.axis_iter_mut(Axis(1)))
                        .par_for_each(|target, mut result_block| {
                            let mut tmp = Array2::<$scalar>::zeros((chunks, nsources));
                            Self::modified_helmholtz_kernel(
                                target,
                                sources,
                                tmp.view_mut(),
                                omega,
                                eval_mode,
                            );
                            Zip::from(charges.axis_iter(Axis(0)))
                                .and(result_block.axis_iter_mut(Axis(0)))
                                .for_each(|charge_vec, result_row| {
                                    Zip::from(tmp.axis_iter(Axis(0))).and(result_row).for_each(
                                        |tmp_row, result_elem| {
                                            Zip::from(tmp_row).and(charge_vec).for_each(
                                                |tmp_elem, charge_elem| {
                                                    *result_elem += *tmp_elem * *charge_elem
                                                },
                                            )
                                        },
                                    )
                                })
                        })
                });
            }
        }
    };
}

// Default implementations for unsupported types.
impl ModifiedHelmholtzEvaluator for c32 {}
impl ModifiedHelmholtzEvaluator for c64 {}

// Actual implementations for supported types.
modified_helmholtz_impl!(f64);
modified_helmholtz_impl!(f32);
