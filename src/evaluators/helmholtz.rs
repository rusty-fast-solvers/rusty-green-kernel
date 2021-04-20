use crate::kernels::EvalMode;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis};
use num::complex::Complex;
use rusty_kernel_tools::{RealType, ThreadingType};

/// Implementation of assembler function for Helmholtz.
pub(crate) fn assemble_in_place_impl_helmholtz<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    mut result: ArrayViewMut2<num::complex::Complex<T>>,
    wavenumber: num::complex::Complex<f64>,
    threading_type: ThreadingType,
) {
    use crate::kernels::helmholtz_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, mut result_row| {
                let mut tmp_real = Array2::<T>::zeros((1, nsources));
                let mut tmp_imag = Array2::<T>::zeros((1, nsources));
                helmholtz_kernel(
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
            }),
        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, mut result_row| {
                let mut tmp_real = Array2::<T>::zeros((1, nsources));
                let mut tmp_imag = Array2::<T>::zeros((1, nsources));
                helmholtz_kernel(
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
            }),
    }
}

/// Implementation of the evaluator function for Laplace kernels.
pub(crate) fn evaluate_in_place_impl_helmholtz<T: RealType>(
    sources: ArrayView2<T>,
    targets: ArrayView2<T>,
    charges: ArrayView2<Complex<T>>,
    mut result: ArrayViewMut3<Complex<T>>,
    wavenumber: Complex<f64>,
    eval_mode: &EvalMode,
    threading_type: ThreadingType,
) {
    use crate::kernels::helmholtz_kernel;
    use ndarray::Zip;

    let nsources = sources.len_of(Axis(1));

    let chunks = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    let charges_real = charges.map(|item| item.re);
    let charges_imag = charges.map(|item| item.im);

    result.fill(num::traits::zero());

    match threading_type {
        ThreadingType::Parallel => Zip::from(targets.columns())
            .and(result.axis_iter_mut(Axis(1)))
            .par_for_each(|target, mut result_block| {
                let mut tmp_real = Array2::<T>::zeros((chunks, nsources));
                let mut tmp_imag = Array2::<T>::zeros((chunks, nsources));
                helmholtz_kernel(
                    target,
                    sources,
                    tmp_real.view_mut(),
                    tmp_imag.view_mut(),
                    wavenumber,
                    eval_mode,
                );
                Zip::from(charges_real.rows())
                    .and(charges_imag.rows())
                    .and(result_block.rows_mut())
                    .for_each(|charge_vec_real, charge_vec_imag, result_row| {
                        Zip::from(tmp_real.rows())
                            .and(tmp_imag.rows())
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
            }),

        ThreadingType::Serial => Zip::from(targets.columns())
            .and(result.axis_iter_mut(Axis(1)))
            .for_each(|target, mut result_block| {
                let mut tmp_real = Array2::<T>::zeros((chunks, nsources));
                let mut tmp_imag = Array2::<T>::zeros((chunks, nsources));
                helmholtz_kernel(
                    target,
                    sources,
                    tmp_real.view_mut(),
                    tmp_imag.view_mut(),
                    wavenumber,
                    eval_mode,
                );
                Zip::from(charges_real.rows())
                    .and(charges_imag.rows())
                    .and(result_block.rows_mut())
                    .for_each(|charge_vec_real, charge_vec_imag, result_row| {
                        Zip::from(tmp_real.rows())
                            .and(tmp_imag.rows())
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
            }),
    }
}
