//! This module defines C API function to access all assembly and evaluation routines.

use crate::*;
use ndarray;
use ndarray_linalg::{c32, c64};

/// Assemble the Laplace kernel (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets, nsources)` array that stores the result.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `num_threads`   - Number of threads to use
#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {
    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    f64::assemble_kernel_in_place(sources, targets, result, KernelType::Laplace, num_threads);
}

/// Assemble the Laplace kernel (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets, nsources)` array that stores the result.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `num_threads`   - Number of threads to use
#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {
    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    f32::assemble_kernel_in_place(sources, targets, result, KernelType::Laplace, num_threads);
}

/// Evaluate the Laplace potential sum (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 1)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 4) array (if `return_gradients is true) that stores the result.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
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
    num_threads: usize,
) {
    let kernel_type = KernelType::Laplace;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
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

    f64::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}

/// Evaluate the Laplace potential sum (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 1)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 4) array (if `return_gradients is true) that stores the result.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
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
    num_threads: usize,
) {
    let kernel_type = KernelType::Laplace;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
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

    f32::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}

/// Assemble the Helmholtz kernel (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets,  2 * nsources)` array that stores the result
///                  using a complex number memory layout.
/// * `wavenumber_real` - Real part of the wavenumber parameter.
/// * `wavenumber_imag` - Imaginary part of the wavenumber parameter.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn assemble_helmholtz_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    result_ptr: *mut f64,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut2::from_shape_ptr(
            (ntargets, nsources),
            result_ptr as *mut c64,
        )
    };
    let wavenumber = c64::new(wavenumber_real, wavenumber_imag);
    let kernel_type = KernelType::Helmholtz(wavenumber);


    c64::assemble_kernel_in_place(sources, targets, result, kernel_type, num_threads);

    
}

/// Assemble the Helmholtz kernel (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets, 2 * nsources)` array that stores the result
///                  using a complex number memory layout.
/// * `wavenumber_real` - Real part of the wavenumber parameter.
/// * `wavenumber_imag` - Imaginary part of the wavenumber parameter.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn assemble_helmholtz_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    result_ptr: *mut f32,
    wavenumber_real: f64,
    wavenumber_imag: f64,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result = unsafe {
        ndarray::ArrayViewMut2::from_shape_ptr(
            (ntargets, nsources),
            result_ptr as *mut c32,
        )
    };
    let wavenumber = c64::new(wavenumber_real, wavenumber_imag);
    let kernel_type = KernelType::Helmholtz(wavenumber);


    c32::assemble_kernel_in_place(sources, targets, result, kernel_type, num_threads);
}

/// Evaluate the Helmholtz potential sum (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 2)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 2 * 4) array (if `return_gradients is true) that stores the result
///                  using a complex number memory layout.
/// * `wavenumber_real` - Real part of the wavenumber parameter.
/// * `wavenumber_imag` - Imaginary part of the wavenumber parameter.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
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
    num_threads: usize,
) {

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };
        
    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };
    let wavenumber = c64::new(wavenumber_real, wavenumber_imag);

    let kernel_type = KernelType::Helmholtz(wavenumber);
    
    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe {
        ndarray::ArrayView2::from_shape_ptr(
            (ncharge_vecs, nsources),
            charge_ptr as *mut c64,
        )
    };
    
    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr(
            (ncharge_vecs, ntargets, ncols),
            result_ptr as *mut c64,
        )
    };
    
    c64::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}

/// Evaluate the Helmholtz potential sum (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 2)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 2 * 4) array (if `return_gradients is true) that stores the result
///                  using a complex number memory layout.
/// * `wavenumber_real` - Real part of the wavenumber parameter.
/// * `wavenumber_imag` - Imaginary part of the wavenumber parameter.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
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
    num_threads: usize,
) {

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
    };

    let ncols: usize = match eval_mode {
        EvalMode::Value => 1,
        EvalMode::ValueGrad => 4,
    };
    let wavenumber = c64::new(wavenumber_real, wavenumber_imag);

    let kernel_type = KernelType::Helmholtz(wavenumber);
    
    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe {
        ndarray::ArrayView2::from_shape_ptr(
            (ncharge_vecs, nsources),
            charge_ptr as *mut c32,
        )
    };
    
    let result = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr(
            (ncharge_vecs, ntargets, ncols),
            result_ptr as *mut c32,
        )
    };
    
    c32::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}

/// Assemble the modified Helmholtz kernel (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets, nsources)` array that stores the result.
/// * `omega`      - The omega parameter of the kernel.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn assemble_modified_helmholtz_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    result_ptr: *mut f64,
    omega: f64,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    f64::assemble_kernel_in_place(sources, targets, result, KernelType::ModifiedHelmholtz(omega), num_threads);
}

/// Assemble the modified Helmholtz kernel (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `result_ptr` - Pointer to an existing `(ntargets, nsources)` array that stores the result.
/// * `omega`      - The omega parameter of the kernel.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn assemble_modified_helmholtz_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    result_ptr: *mut f32,
    omega: f64,
    nsources: usize,
    ntargets: usize,
    num_threads: usize,
) {

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let result =
        unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    f32::assemble_kernel_in_place(sources, targets, result, KernelType::ModifiedHelmholtz(omega), num_threads);
}

/// Evaluate the modified Helmholtz potential sum (double precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 1)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 4) array (if `return_gradients is true) that stores the result.
/// * `omega`      - The omega parameter of the kernel.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn evaluate_modified_helmholtz_kernel_f64(
    source_ptr: *const f64,
    target_ptr: *const f64,
    charge_ptr: *const f64,
    result_ptr: *mut f64,
    omega: f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    num_threads: usize,
) {

    let kernel_type = KernelType::Laplace;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
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

    f64::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}

/// Evaluate the modified Helmholtz potential sum (single precision version).
///
///
/// # Arguments
///
/// * `source_ptr` - Pointer to a `(3, nsources)` array of sources.
/// * `target_ptr` - Pointer to a `(3, ntargets)` array of targets.
/// * `charge_ptr` - Pointer to a `(ncharge_vecs, nsources)` array of `ncharge_vecs` charge vectors.
/// * `result_ptr` - Pointer to an existing `(ncharge_vecs, ntargets, 1)` array (if `return_gradients is false)
///                  or to an `(ncharge_vecs, ntargets, 4) array (if `return_gradients is true) that stores the result.
/// * `omega`      - The omega parameter of the kernel.
/// * `nsources`   - Number of sources.
/// * `ntargets`   - Number of targets.
/// * `ncharge_vecs` - Number of charge vectors.
/// * `return_gradients` - If true return also the gradients.
/// * `parallel`   - If true, assemble multithreaded, otherwise single threaded.
#[no_mangle]
pub extern "C" fn evaluate_modified_helmholtz_kernel_f32(
    source_ptr: *const f32,
    target_ptr: *const f32,
    charge_ptr: *const f32,
    result_ptr: *mut f32,
    omega: f64,
    nsources: usize,
    ntargets: usize,
    ncharge_vecs: usize,
    return_gradients: bool,
    num_threads: usize,
) {

    let kernel_type = KernelType::Laplace;

    let eval_mode = match return_gradients {
        true => EvalMode::ValueGrad,
        false => EvalMode::Value,
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

    f32::evaluate_kernel_in_place(
        sources,
        targets,
        charges,
        result,
        kernel_type,
        eval_mode,
        num_threads,
    );
}
