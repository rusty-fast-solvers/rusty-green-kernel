use crate::kernels::EvalMode;
/// Implementation of assembler function for modified Helmholtz kernels.
use ndarray::{Array2, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use rusty_kernel_tools::{RealType, ThreadingType};

///Implementation of the assembler function for modified Helmholtz kernels.
pub(crate) fn assemble_in_place_impl_modified_helmholtz<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    mut result: ArrayViewMut2<T>,
    omega: f64,
    threading_type: ThreadingType,
) {
    use crate::kernels::modified_helmholtz_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, result_row| {
                let tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot convert to 2-dimensional array.");
                modified_helmholtz_kernel(target, sources, tmp, omega, &EvalMode::Value);
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, result_row| {
                let tmp = result_row
                    .into_shape((1, nsources))
                    .expect("Cannot conver to 2-dimensional array.");
                modified_helmholtz_kernel(target, sources, tmp, omega, &EvalMode::Value);
            }),
    }
}

/// Implementation of the evaluator function for modified Helmholtz kernels.
pub(crate) fn evaluate_in_place_impl_modified_helmholtz<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    charges: ArrayView2<T>,
    mut result: ArrayViewMut3<T>,
    omega: f64,
    eval_mode: &EvalMode,
    threading_type: ThreadingType,
) {
    use crate::kernels::modified_helmholtz_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    let chunks = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    result.fill(num::traits::zero());

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.axis_iter_mut(Axis(1)))
            .par_for_each(|target, mut result_block| {
                let mut tmp = Array2::<T>::zeros((chunks, nsources));
                modified_helmholtz_kernel(target, sources, tmp.view_mut(), omega, eval_mode);
                Zip::from(charges.rows())
                    .and(result_block.rows_mut())
                    .for_each(|charge_vec, result_row| {
                        Zip::from(tmp.rows())
                            .and(result_row)
                            .for_each(|tmp_row, result_elem| {
                                Zip::from(tmp_row).and(charge_vec).for_each(
                                    |tmp_elem, charge_elem| {
                                        *result_elem += *tmp_elem * *charge_elem
                                    },
                                )
                            })
                    })
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.axis_iter_mut(Axis(1)))
            .for_each(|target, mut result_block| {
                let mut tmp = Array2::<T>::zeros((chunks, nsources));
                modified_helmholtz_kernel(target, sources, tmp.view_mut(), omega, eval_mode);
                Zip::from(charges.rows())
                    .and(result_block.rows_mut())
                    .for_each(|charge_vec, result_row| {
                        Zip::from(tmp.rows())
                            .and(result_row)
                            .for_each(|tmp_row, result_elem| {
                                Zip::from(tmp_row).and(charge_vec).for_each(
                                    |tmp_elem, charge_elem| {
                                        *result_elem += *tmp_elem * *charge_elem
                                    },
                                )
                            })
                    })
            }),
    }
}
