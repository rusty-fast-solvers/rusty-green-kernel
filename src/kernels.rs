//! Definitions of the supported Greens function kernels.
use rusty_kernel_tools::RealType;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use num;

pub enum EvalMode {
    Value,
    ValueGrad,
}

pub fn laplace_kernel<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    result: ArrayViewMut2<T>,
    eval_mode: &EvalMode,
) {
    match eval_mode {
        EvalMode::Value => laplace_kernel_impl_no_deriv(target, sources, result),
        EvalMode::ValueGrad => laplace_kernel_impl_deriv(target, sources, result),
    };
}

/// Implementation of the Laplace kernel without derivatives
pub fn laplace_kernel_impl_no_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    mut result: ArrayViewMut2<T>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();

    let m_inv_4pi: T =
        num::traits::cast::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    result.fill(zero);

    Zip::from(target)
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .for_each(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    result
        .index_axis_mut(Axis(0), 0)
        .mapv_inplace(|item| m_inv_4pi / item.sqrt());
    result
        .index_axis_mut(Axis(0), 0)
        .iter_mut()
        .filter(|item| item.is_infinite())
        .for_each(|item| *item = zero);
}

/// Implementation of the Laplace kernel with derivatives
pub fn laplace_kernel_impl_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    mut result: ArrayViewMut2<T>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();

    let m_inv_4pi: T =
        num::traits::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let m_4pi: T = num::traits::cast::<f64, T>(4.0).unwrap() * num::traits::FloatConst::PI();

    result.fill(zero);

    // First compute the Green fct. values

    Zip::from(target)
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .for_each(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    // Now compute the derivatives.

    result
        .index_axis_mut(Axis(0), 0)
        .mapv_inplace(|item| m_inv_4pi / item.sqrt());
    result
        .index_axis_mut(Axis(0), 0)
        .iter_mut()
        .filter(|item| item.is_infinite())
        .for_each(|item| *item = zero);

    let (values, mut derivs) = result.split_at(Axis(0), 1);
    let values = values.index_axis(Axis(0), 0);

    Zip::from(derivs.rows_mut())
        .and(target.view())
        .and(sources.rows())
        .for_each(|deriv_row, &target_value, source_row| {
            Zip::from(deriv_row).and(source_row).and(values).for_each(
                |deriv_value, &source_value, &value| {
                    *deriv_value =
                        (m_4pi * value).powi(3) * (source_value - target_value) * m_inv_4pi;
                },
            )
        });
}

/// Implementation of the Helmholtz kernel
pub fn helmholtz_kernel<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    result_real: ArrayViewMut2<T>,
    result_imag: ArrayViewMut2<T>,
    wavenumber: num::complex::Complex<f64>,
    eval_mode: &EvalMode,
) {
    match eval_mode {
        EvalMode::Value => {
            helmholtz_kernel_impl_no_deriv(target, sources, result_real, result_imag, wavenumber)
        }
        EvalMode::ValueGrad => {
            helmholtz_kernel_impl_deriv(target, sources, result_real, result_imag, wavenumber)
        }
    };
}

pub fn helmholtz_kernel_impl_no_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    mut result_real: ArrayViewMut2<T>,
    mut result_imag: ArrayViewMut2<T>,
    wavenumber: num::complex::Complex<f64>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();

    let m_inv_4pi: T =
        num::traits::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let wavenumber_real: T = num::traits::cast::<f64, T>(wavenumber.re).unwrap();
    let wavenumber_imag: T = num::traits::cast::<f64, T>(wavenumber.im).unwrap();

    let mut dist = Array1::<T>::zeros(sources.len_of(Axis(1)));
    let exp_im = (-wavenumber_imag).exp();

    result_real.fill(zero);
    result_imag.fill(zero);

    Zip::from(target)
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(dist.view_mut())
                .for_each(|&source_value, dist_ref| {
                    *dist_ref += (source_value - target_value).powi(2)
                })
        });

    dist.mapv_inplace(|item| item.sqrt());

    Zip::from(dist.view())
        .and(result_real.index_axis_mut(Axis(0), 0))
        .and(result_imag.index_axis_mut(Axis(0), 0))
        .for_each(|&dist_val, result_real_val, result_imag_val| {
            *result_real_val = exp_im * (wavenumber_real * dist_val).cos() * m_inv_4pi / dist_val;
            *result_imag_val = exp_im * (wavenumber_real * dist_val).sin() * m_inv_4pi / dist_val;
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

pub fn helmholtz_kernel_impl_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    mut result_real: ArrayViewMut2<T>,
    mut result_imag: ArrayViewMut2<T>,
    wavenumber: num::complex::Complex<f64>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();
    let one: T = num::traits::one();

    let m_inv_4pi: T =
        num::traits::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let wavenumber_real: T = num::traits::cast::<f64, T>(wavenumber.re).unwrap();
    let wavenumber_imag: T = num::traits::cast::<f64, T>(wavenumber.im).unwrap();

    let mut dist = Array1::<T>::zeros(sources.len_of(Axis(1)));
    let exp_im = (-wavenumber_imag).exp();

    result_real.fill(zero);
    result_imag.fill(zero);

    Zip::from(target)
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(dist.view_mut())
                .for_each(|&source_value, dist_ref| {
                    *dist_ref += (source_value - target_value).powi(2)
                })
        });

    dist.mapv_inplace(|item| item.sqrt());

    Zip::from(dist.view())
        .and(result_real.index_axis_mut(Axis(0), 0))
        .and(result_imag.index_axis_mut(Axis(0), 0))
        .for_each(|&dist_val, result_real_val, result_imag_val| {
            *result_real_val = exp_im * (wavenumber_real * dist_val).cos() * m_inv_4pi / dist_val;
            *result_imag_val = exp_im * (wavenumber_real * dist_val).sin() * m_inv_4pi / dist_val;
        });

    // Now do the derivative term

    let (values_real, mut derivs_real) = result_real.view_mut().split_at(Axis(0), 1);
    let (values_imag, mut derivs_imag) = result_imag.view_mut().split_at(Axis(0), 1);

    let values_real = values_real.index_axis(Axis(0), 0);
    let values_imag = values_imag.index_axis(Axis(0), 0);

    Zip::from(derivs_real.rows_mut())
        .and(derivs_imag.rows_mut())
        .and(target.view())
        .and(sources.rows())
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
                            *deriv_real_value = (target_value - source_value) / dist_value.powi(2)
                                * ((-one - wavenumber_imag * dist_value) * value_real
                                    - wavenumber_real * dist_value * value_imag);
                            *deriv_imag_value = (target_value - source_value) / dist_value.powi(2)
                                * (value_real * wavenumber_real * dist_value
                                    + (-one - wavenumber_imag * dist_value) * value_imag);
                        },
                    )
            },
        );

    Zip::from(result_real.rows_mut())
        .and(result_imag.rows_mut())
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
