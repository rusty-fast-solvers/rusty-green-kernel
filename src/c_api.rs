use ndarray;
use num::complex::Complex;
use rusty_kernel_tools::ThreadingType;

use crate::RealDirectEvaluator;
use crate::ComplexDirectEvaluator;
use crate::make_laplace_evaluator;
use crate::make_helmholtz_evaluator;

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    make_laplace_evaluator(sources, targets).assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    make_laplace_evaluator(sources, targets).assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    charge_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges =
        unsafe { ndarray::ArrayView2::from_shape_ptr((ncharge_vecs, nsources), charge_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr((ncharge_vecs, ntargets, ncols), result_ptr)
    };

    make_laplace_evaluator(sources, targets).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    charge_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges =
        unsafe { ndarray::ArrayView2::from_shape_ptr((ncharge_vecs, nsources), charge_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr((ncharge_vecs, ntargets, ncols), result_ptr)
    };

    make_laplace_evaluator(sources, targets).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );
}

#[no_mangle]
pub extern "C" fn assemble_helmholtz_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    result_ptr: *mut f64,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut2::from_shape_ptr(
            (ntargets, nsources),
            result_ptr as *mut Complex<f64>,
        )
    };
    let wavenumber = Complex::<f64>::new(wavenumber_real, wavenumber_imag);

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    make_helmholtz_evaluator(sources, targets, wavenumber)
        .assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn assemble_helmholtz_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    result_ptr: *mut f32,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut2::from_shape_ptr(
            (ntargets, nsources),
            result_ptr as *mut Complex<f32>,
        )
    };
    let wavenumber = Complex::<f64>::new(wavenumber_real, wavenumber_imag);

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    make_helmholtz_evaluator(sources, targets, wavenumber)
        .assemble_in_place(result, threading_type);
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    charge_ptr: *const f64,
    result_ptr: *mut f64,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };
    let wavenumber = Complex::<f64>::new(wavenumber_real, wavenumber_imag);

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe {
        ndarray::ArrayView2::from_shape_ptr(
            (ncharge_vecs, nsources),
            charge_ptr as *mut Complex<f64>,
        )
    };

    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr(
            (ncharge_vecs, ntargets, ncols),
            result_ptr as *mut Complex<f64>,
        )
    };

    make_helmholtz_evaluator(sources, targets, wavenumber).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );
}

#[no_mangle]
pub extern "C" fn evaluate_helmholtz_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    charge_ptr: *const f32,
    result_ptr: *mut f32,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    parallel: bool,
) {
    use crate::kernels::EvalMode;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let threading_type = match parallel {
        true => ThreadingType::Parallel,
        false => ThreadingType::Serial,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };
    let wavenumber = Complex::<f64>::new(wavenumber_real, wavenumber_imag);

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe {
        ndarray::ArrayView2::from_shape_ptr(
            (ncharge_vecs, nsources),
            charge_ptr as *mut Complex<f32>,
        )
    };
    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr(
            (ncharge_vecs, ntargets, ncols),
            result_ptr as *mut Complex<f32>,
        )
    };

    make_helmholtz_evaluator(sources, targets, wavenumber).evaluate_in_place(
        charges,
        result,
        &eval_mode,
        threading_type,
    );
}
