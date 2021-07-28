//! Definitions of the supported Greens function kernels.
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use num;
use rusty_base::{EvalMode, RealType};


/// Evaluation of the Laplace kernel for
/// a single target and many sources. 
/// 
/// The type T is either f32 or f64.
/// 
/// # Arguments
/// 
/// * `target` - An array with 3 elements containing the target point.
/// * `sources` - An array of shape (3, nsources) cotaining the source points.
/// * `result` - If eval_mode is equal to `Value` an array of shape (1, nsources)
///              that contains the values of the Green's function between the target
///              and the sources.
/// * `eval_mode` - The Evaluation Mode. Either `Value` if only the values of the Green's
///                 function are requested, or `ValueGrad` if both the value and derivatives
///                 are requested.
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

/// Implementation of the Laplace kernel without derivatives.
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
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .apply(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    result
        .index_axis_mut(Axis(0), 0)
        .mapv_inplace(|item| m_inv_4pi / <T as num::Float>::sqrt(item));
    result
        .index_axis_mut(Axis(0), 0)
        .iter_mut()
        .filter(|item| item.is_infinite())
        .for_each(|item| *item = zero);
}

/// Implementation of the Laplace kernel with derivatives.
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
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .apply(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    // Now compute the derivatives.

    result
        .index_axis_mut(Axis(0), 0)
        .mapv_inplace(|item| m_inv_4pi / <T as num::Float>::sqrt(item));
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
        .apply(|deriv_row, &target_value, source_row| {
            Zip::from(deriv_row).and(source_row).and(values).apply(
                |deriv_value, &source_value, &value| {
                    *deriv_value =
                        <T as num::Float>::powi(m_4pi * value, 3) * (source_value - target_value) * m_inv_4pi;
                },
            )
        });
}

/// Evaluation of the Helmholtz kernel for
/// a single target and many sources. 
/// 
/// The type T is either f32 or f64.
/// 
/// # Arguments
/// 
/// * `target` - An array with 3 elements containing the target point.
/// * `sources` - An array of shape (3, nsources) cotaining the source points.
/// * `result_real` - If eval_mode is equal to `Value` an array of shape (1, nsources)
///                   that contains the real part of the Green's function values between the target
///                   and the sources.
/// * `result__imag` - If eval_mode is equal to `Value` an array of shape (1, nsources)
///                   that contains the imaginary part of the Green's function values between the target
///                   and the sources.
/// * `wavenumber`   - The wavenumber k of the Helmholtz kernel.
/// * `eval_mode` - The Evaluation Mode. Either `Value` if only the values of the Green's
///                 function are requested, or `ValueGrad` if both the value and derivatives
///                 are requested.
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

/// Implementation of the Helmholtz kernel with derivatives.
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

    result_real.fill(zero);
    result_imag.fill(zero);

    Zip::from(target)
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(dist.view_mut())
                .apply(|&source_value, dist_ref| {
                    *dist_ref += <T as num::Float>::powi(source_value - target_value, 2)
                })
        });

    dist.mapv_inplace(|item| <T as num::Float>::sqrt(item));

    Zip::from(dist.view())
        .and(result_real.index_axis_mut(Axis(0), 0))
        .and(result_imag.index_axis_mut(Axis(0), 0))
        .apply(|&dist_val, result_real_val, result_imag_val| {
            let exp_val = <T as num::Float>::exp(-wavenumber_imag * dist_val);
            *result_real_val = exp_val * <T as num::Float>::cos(wavenumber_real * dist_val) * m_inv_4pi / dist_val;
            *result_imag_val = exp_val * <T as num::Float>::sin(wavenumber_real * dist_val) * m_inv_4pi / dist_val;
        });

    Zip::from(dist.view())
        .and(result_real.index_axis_mut(Axis(0), 0))
        .and(result_imag.index_axis_mut(Axis(0), 0))
        .apply(|&dist_val, result_real_val, result_imag_val| {
            if dist_val == zero {
                *result_real_val = zero;
                *result_imag_val = zero;
            }
        });
}

/// Implementation of the Helmholtz kernel with derivatives.
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

    result_real.fill(zero);
    result_imag.fill(zero);

    Zip::from(target)
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(dist.view_mut())
                .apply(|&source_value, dist_ref| {
                    *dist_ref += <T as num::Float>::powi(source_value - target_value, 2)
                })
        });

    dist.mapv_inplace(|item| <T as num::Float>::sqrt(item));

    Zip::from(dist.view())
        .and(result_real.index_axis_mut(Axis(0), 0))
        .and(result_imag.index_axis_mut(Axis(0), 0))
        .apply(|&dist_val, result_real_val, result_imag_val| {
            let exp_val = <T as num::Float>::exp(-wavenumber_imag * dist_val);
            *result_real_val = exp_val * <T as num::Float>::cos(wavenumber_real * dist_val) * m_inv_4pi / dist_val;
            *result_imag_val = exp_val * <T as num::Float>::sin(wavenumber_real * dist_val) * m_inv_4pi / dist_val;
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
        .apply(
            |deriv_real_row, deriv_imag_row, &target_value, source_row| {
                Zip::from(deriv_real_row)
                    .and(deriv_imag_row)
                    .and(source_row)
                    .and(values_real)
                    .and(values_imag)
                    .and(dist.view())
                    .apply(
                        |deriv_real_value,
                         deriv_imag_value,
                         &source_value,
                         &value_real,
                         &value_imag,
                         &dist_value| {
                            *deriv_real_value = (target_value - source_value) / <T as num::Float>::powi(dist_value, 2)
                                * ((-one - wavenumber_imag * dist_value) * value_real
                                    - wavenumber_real * dist_value * value_imag);
                            *deriv_imag_value = (target_value - source_value) / <T as num::Float>::powi(dist_value, 2)
                                * (value_real * wavenumber_real * dist_value
                                    + (-one - wavenumber_imag * dist_value) * value_imag);
                        },
                    )
            },
        );

    Zip::from(result_real.axis_iter_mut(Axis(0)))
        .and(result_imag.axis_iter_mut(Axis(0)))
        .apply(|real_row, imag_row| {
            Zip::from(dist.view()).and(real_row).and(imag_row).apply(
                |dist_elem, real_elem, imag_elem| {
                    if *dist_elem == zero {
                        *real_elem = zero;
                        *imag_elem = zero;
                    }
                },
            )
        });
}

/// Evaluation of the modified Helmholtz kernel for
/// a single target and many sources. 
/// 
/// The type T is either f32 or f64.
/// 
/// # Arguments
/// 
/// * `target` - An array with 3 elements containing the target point.
/// * `sources` - An array of shape (3, nsources) cotaining the source points.
/// * `result` - If eval_mode is equal to `Value` an array of shape (1, nsources)
///              that contains the values of the Green's function between the target
///              and the sources.
/// * `omega` - The omega parameter of the modified Helmholtz kernel.
/// * `eval_mode` - The Evaluation Mode. Either `Value` if only the values of the Green's
///                 function are requested, or `ValueGrad` if both the value and derivatives
///                 are requested.
pub fn modified_helmholtz_kernel<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    result: ArrayViewMut2<T>,
    omega: f64,
    eval_mode: &EvalMode,
) {
    match eval_mode {
        EvalMode::Value => modified_helmholtz_kernel_impl_no_deriv(target, sources, omega, result),
        EvalMode::ValueGrad => modified_helmholtz_kernel_impl_deriv(target, sources, omega, result),
    };
}

/// Implementation of the modified Helmholtz kernel without derivatives.
pub fn modified_helmholtz_kernel_impl_no_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    omega: f64,
    mut result: ArrayViewMut2<T>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();

    let m_inv_4pi: T =
        num::traits::cast::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let omega: T = num::traits::cast::cast::<f64, T>(omega).unwrap();

    result.fill(zero);

    Zip::from(target)
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(result.index_axis_mut(Axis(0), 0))
                .apply(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    result
        .index_axis_mut(Axis(0), 0)
        .map_inplace(|item| *item = <T as num::Float>::sqrt(*item));

    result
        .index_axis_mut(Axis(0), 0)
        .map_inplace(|item| *item = <T as num::Float>::exp(-omega * *item) * m_inv_4pi / *item);
    result
        .index_axis_mut(Axis(0), 0)
        .iter_mut()
        .filter(|item| !item.is_finite())
        .for_each(|item| *item = zero);
}

/// Implementation of the modified Helmholtz kernel with derivatives.
pub fn modified_helmholtz_kernel_impl_deriv<T: RealType>(
    target: ArrayView1<T>,
    sources: ArrayView2<T>,
    omega: f64,
    mut result: ArrayViewMut2<T>,
) {
    use ndarray::Zip;

    let zero: T = num::traits::zero();
    let one: T = num::traits::one();

    let m_inv_4pi: T =
        num::traits::cast::<f64, T>(0.25).unwrap() * num::traits::FloatConst::FRAC_1_PI();

    let omega: T = num::traits::cast::cast::<f64, T>(omega).unwrap();

    result.fill(zero);

    let mut dist = ndarray::Array1::<T>::zeros(sources.len_of(Axis(1)));

    // First compute the Green fct. values

    Zip::from(target)
        .and(sources.axis_iter(Axis(0)))
        .apply(|&target_value, source_row| {
            Zip::from(source_row)
                .and(dist.view_mut())
                .apply(|&source_value, dist_ref| {
                    *dist_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    dist.map_inplace(|item| *item = <T as num::Float>::sqrt(*item));


    Zip::from(result.index_axis_mut(Axis(0), 0))
        .and(dist.view())
        .apply(|result_ref, &dist_value| {
            *result_ref = <T as num::Float>::exp(-omega * dist_value) * m_inv_4pi / dist_value
        });

        // Now compute the derivatives.

    let (values, mut derivs) = result.view_mut().split_at(Axis(0), 1);
    let values = values.index_axis(Axis(0), 0);

    Zip::from(derivs.axis_iter_mut(Axis(0)))
        .and(target.view())
        .and(sources.axis_iter(Axis(0)))
        .apply(|deriv_row, &target_value, source_row| {
            Zip::from(deriv_row)
                .and(source_row)
                .and(values)
                .and(dist.view())
                .apply(|deriv_value, &source_value, &value, &dist_value| {
                    *deriv_value = value * (target_value - source_value) / <T as num::Float>::powi(dist_value, 2)
                        * (-omega * dist_value - one)
                })
        });

    result.view_mut().axis_iter_mut(Axis(0)).for_each(|row|
        Zip::from(row)
        .and(dist.view())
        .apply(|elem, &dist_value|
            if dist_value == zero {
                *elem = zero;
            })
        );
}
