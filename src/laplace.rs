use crate::{EvalMode, ThreadingType};
/// Implementation of assembler function for Laplace kernels.
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use ndarray_linalg::Scalar;
use num::traits::FloatConst;

pub(crate) trait LaplaceEvaluator: Scalar {

    fn laplace_kernel(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
        eval_mode: &EvalMode,
    );
    /// Implementation of the Laplace kernel without derivatives.
    fn laplace_kernel_no_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
    );

    // Implementation of the Laplace kernel with derivatives.
    fn laplace_kernel_with_deriv(
        target: ArrayView1<<Self as Scalar>::Real>,
        sources: ArrayView2<<Self as Scalar>::Real>,
        result: ArrayViewMut2<Self>,
    );
    
    #[allow(unused_variables)]
    fn evaluate_in_place_laplace(
        sources: ArrayView2<<Self as Scalar>::Real>,
        targets: ArrayView2<<Self as Scalar>::Real>,
        charges: ArrayView2<Self>,
        result: ArrayViewMut3<Self>,
        eval_mode: &EvalMode,
        threading_type: &ThreadingType,
    ) {

    panic!("Not implemented for this type.");

    }
}

macro_rules! laplace_impl {
    ($scalar:ty) => {
        impl LaplaceEvaluator for $scalar {

            fn laplace_kernel(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                result: ArrayViewMut2<$scalar>,
                eval_mode: &EvalMode,
            ) {
                {
                    match eval_mode {
                        EvalMode::Value => Self::laplace_kernel_no_deriv(target, sources, result),
                        EvalMode::ValueGrad => {
                            Self::laplace_kernel_with_deriv(target, sources, result)
                        }
                    };
                }
            }
            /// Implementation of the Laplace kernel without derivatives.
            fn laplace_kernel_no_deriv(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                mut result: ArrayViewMut2<$scalar>,
            ) {
                use ndarray::Zip;

                let zero: $scalar = num::traits::zero();

                let m_inv_4pi: $scalar =
                    num::traits::cast::cast::<f64, $scalar>(0.25).unwrap() * <$scalar>::FRAC_1_PI();

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
                    .mapv_inplace(|item| m_inv_4pi / <$scalar>::sqrt(item));
                result
                    .index_axis_mut(Axis(0), 0)
                    .iter_mut()
                    .filter(|item| item.is_infinite())
                    .for_each(|item| *item = zero);
            }

            /// Implementation of the Laplace kernel with derivatives.
            fn laplace_kernel_with_deriv(
                target: ArrayView1<$scalar>,
                sources: ArrayView2<$scalar>,
                mut result: ArrayViewMut2<$scalar>,
            ) {
                use ndarray::Zip;

                let zero: $scalar = num::traits::zero();

                let m_inv_4pi: $scalar =
                    num::traits::cast::<f64, $scalar>(0.25).unwrap() * <$scalar>::FRAC_1_PI();

                let m_4pi: $scalar =
                    num::traits::cast::<f64, $scalar>(4.0).unwrap() * <$scalar>::PI();

                result.fill(zero);

                // First compute the Green fct. values

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

                // Now compute the derivatives.

                result
                    .index_axis_mut(Axis(0), 0)
                    .mapv_inplace(|item| m_inv_4pi / <$scalar>::sqrt(item));
                result
                    .index_axis_mut(Axis(0), 0)
                    .iter_mut()
                    .filter(|item| item.is_infinite())
                    .for_each(|item| *item = zero);

                let (values, mut derivs) = result.split_at(Axis(0), 1);
                let values = values.index_axis(Axis(0), 0);

                Zip::from(derivs.axis_iter_mut(Axis(0)))
                    .and(target.view())
                    .and(sources.axis_iter(Axis(0)))
                    .for_each(|deriv_row, &target_value, source_row| {
                        Zip::from(deriv_row).and(source_row).and(values).for_each(
                            |deriv_value, &source_value, &value| {
                                *deriv_value = <$scalar>::powi(m_4pi * value, 3)
                                    * (source_value - target_value)
                                    * m_inv_4pi;
                            },
                        )
                    });
            }

            fn evaluate_in_place_laplace(
                sources: ArrayView2<$scalar>,
                targets: ArrayView2<$scalar>,
                charges: ArrayView2<$scalar>,
                mut result: ArrayViewMut3<$scalar>,
                eval_mode: &EvalMode,
                threading_type: &ThreadingType,
            ) {
                use ndarray::Zip;

                let nsources = sources.len_of(Axis(1));

                let chunks = match eval_mode {
                    EvalMode::Value => 1,
                    EvalMode::ValueGrad => 4,
                };

                result.fill(num::traits::zero());

                match threading_type {
                    ThreadingType::Parallel => Zip::from(targets.axis_iter(Axis(1)))
                        .and(result.axis_iter_mut(Axis(1)))
                        .par_for_each(|target, mut result_block| {
                            let mut tmp = Array2::<$scalar>::zeros((chunks, nsources));
                            Self::laplace_kernel(target, sources, tmp.view_mut(), eval_mode);
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
                        }),
                    ThreadingType::Serial => Zip::from(targets.axis_iter(Axis(1)))
                        .and(result.axis_iter_mut(Axis(1)))
                        .for_each(|target, mut result_block| {
                            let mut tmp = Array2::<$scalar>::zeros((chunks, nsources));
                            Self::laplace_kernel(target, sources, tmp.view_mut(), eval_mode);
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
                        }),
                }
            }
        }
    };
}

laplace_impl!(f64);
laplace_impl!(f32);

// ///Implementation of the assembler function for Laplace kernels.
// pub(crate) fn assemble_in_place_impl_laplace<T: RealType>(
//     sources: ArrayView2<T>,
//     targets: ArrayView2<T>,
//     mut result: ArrayViewMut2<T>,
//     threading_type: ThreadingType,
// ) {
//     use crate::kernels::laplace_kernel;
//     use ndarray::Zip;
//
//     let nsources = sources.len_of(Axis(1));
//
//     match threading_type {
//         ThreadingType::Parallel => Zip::from(targets.axis_iter(Axis(1)))
//             .and(result.axis_iter_mut(Axis(0)))
//             .par_apply(|target, result_row| {
//                 let tmp = result_row
//                     .into_shape((1, nsources))
//                     .expect("Cannot convert to 2-dimensional array.");
//                 laplace_kernel(target, sources, tmp, &EvalMode::Value);
//             }),
//         ThreadingType::Serial => Zip::from(targets.axis_iter(Axis(1)))
//             .and(result.axis_iter_mut(Axis(0)))
//             .apply(|target, result_row| {
//                 let tmp = result_row
//                     .into_shape((1, nsources))
//                     .expect("Cannot conver to 2-dimensional array.");
//                 laplace_kernel(target, sources, tmp, &EvalMode::Value);
//             }),
//     }
// }
//
// /// Implementation of the evaluator function for Laplace kernels.
// pub(crate) fn evaluate_in_place_impl_laplace<t: realtype>(
//     sources: arrayview2<t>,
//     targets: arrayview2<t>,
//     charges: arrayview2<t>,
//     mut result: arrayviewmut3<t>,
//     eval_mode: &evalmode,
//     threading_type: threadingtype,
// ) {
//     use crate::kernels::laplace_kernel;
//     use ndarray::Zip;
//
//     let nsources = sources.len_of(Axis(1));
//
//     let chunks = match eval_mode {
//         EvalMode::Value => 1,
//         EvalMode::ValueGrad => 4,
//     };
//
//     result.fill(num::traits::zero());
//
//     match threading_type {
//         ThreadingType::Parallel => Zip::from(targets.axis_iter(Axis(1)))
//             .and(result.axis_iter_mut(Axis(1)))
//             .par_apply(|target, mut result_block| {
//                 let mut tmp = Array2::<T>::zeros((chunks, nsources));
//                 laplace_kernel(target, sources, tmp.view_mut(), eval_mode);
//                 Zip::from(charges.axis_iter(Axis(0)))
//                     .and(result_block.axis_iter_mut(Axis(0)))
//                     .apply(|charge_vec, result_row| {
//                         Zip::from(tmp.axis_iter(Axis(0)))
//                             .and(result_row)
//                             .apply(|tmp_row, result_elem| {
//                                 Zip::from(tmp_row).and(charge_vec).apply(
//                                     |tmp_elem, charge_elem| {
//                                         *result_elem += *tmp_elem * *charge_elem
//                                     },
//                                 )
//                             })
//                     })
//             }),
//         ThreadingType::Serial => Zip::from(targets.axis_iter(Axis(1)))
//             .and(result.axis_iter_mut(Axis(1)))
//             .apply(|target, mut result_block| {
//                 let mut tmp = Array2::<T>::zeros((chunks, nsources));
//                 laplace_kernel(target, sources, tmp.view_mut(), eval_mode);
//                 Zip::from(charges.axis_iter(Axis(0)))
//                     .and(result_block.axis_iter_mut(Axis(0)))
//                     .apply(|charge_vec, result_row| {
//                         Zip::from(tmp.axis_iter(Axis(0)))
//                             .and(result_row)
//                             .apply(|tmp_row, result_elem| {
//                                 Zip::from(tmp_row).and(charge_vec).apply(
//                                     |tmp_elem, charge_elem| {
//                                         *result_elem += *tmp_elem * *charge_elem
//                                     },
//                                 )
//                             })
//                     })
//             }),
//     }
// }
