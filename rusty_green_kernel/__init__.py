import numpy as np
from .rusty_green_kernel import ffi, lib

def as_double_ptr(arr):
    """Turn to a double ptr."""
    return ffi.cast("double*", arr.ctypes.data)


def as_float_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast("float*", arr.ctypes.data)


def as_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)

def as_double(num):
    """Cast number to double."""
    return ffi.cast("double", num)

def as_float(num):
    """Cast number to float."""
    return ffi.cat("float", num)

def align_data(arr, dtype=None):
    """Make sure that an array has the right properties."""

    if dtype is None:
        dtype = arr.dtype

    return np.require(arr, dtype=dtype, requirements=["C", "A"])


def assemble_laplace_kernel(sources, targets, dtype=np.float64, parallel=True):
    """Assemble the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    targets = align_data(targets, dtype=dtype)
    sources = align_data(sources, dtype=dtype)

    result = np.empty((ntargets, nsources), dtype=dtype)

    if dtype == np.float32:
        lib.assemble_laplace_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(result),
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    elif dtype == np.float64:
        lib.assemble_laplace_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(result),
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    else:
        raise NotImplementedError

    return result


def evaluate_laplace_kernel(
    sources, targets, charges, dtype=np.float64, parallel=True, return_gradients=False
):
    """Evaluate the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            f"charges must be a 1- or 2-dim array of shape (...,nsources), current shape: {charges.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if return_gradients:
        ncols = 4
    else:
        ncols = 1

    if charges.ndim == 1:
        ncharge_vecs = 1
        result = np.empty((ntargets, ncols), dtype=dtype)

    else:
        ncharge_vecs = charges.shape[0]
        result = np.empty((ncharge_vecs, ntargets, ncols), dtype=dtype)

    targets = align_data(targets, dtype=dtype)
    sources = align_data(sources, dtype=dtype)
    charges = align_data(charges, dtype=dtype)

    if dtype == np.float32:
        lib.evaluate_laplace_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(charges),
            as_float_ptr(result),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    elif dtype == np.float64:
        lib.evaluate_laplace_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(charges),
            as_double_ptr(result),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    else:
        raise NotImplementedError

    return result


def assemble_helmholtz_kernel(
    sources, targets, wavenumber, dtype=np.complex128, parallel=True
):
    """Assemble the Helmholtz kernel matrix for many targets and sources."""

    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(
            f"dtype must be one of [np.complex128, np.complex64], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    targets = align_data(targets, dtype=real_type)
    sources = align_data(sources, dtype=real_type)

    buffer = np.empty(2 * nsources * ntargets, dtype=real_type)

    if real_type == np.float32:
        lib.assemble_helmholtz_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(buffer),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    elif real_type == np.float64:
        lib.assemble_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(buffer),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    else:
        raise NotImplementedError

    result = np.frombuffer(buffer, dtype=dtype).reshape(ntargets, nsources)

    return result


def evaluate_helmholtz_kernel(
    sources,
    targets,
    charges,
    wavenumber,
    dtype=np.complex128,
    parallel=True,
    return_gradients=False,
):
    """Evaluate the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(
            f"dtype must be one of [np.complex128, np.complex64], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            f"charges must be a 1- or 2-dim array of shape (...,nsources), current shape: {charges.shape}."
        )

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")


    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if return_gradients:
        ncols = 4
    else:
        ncols = 1

    if charges.ndim == 1:
        ncharge_vecs = 1
        result_buffer = np.empty(2 * ntargets * ncols, dtype=real_type)

    else:
        ncharge_vecs = charges.shape[0]
        result_buffer = np.empty(2 * ntargets * ncols * ncharge_vecs, dtype=real_type)

    targets = align_data(targets, dtype=real_type)
    sources = align_data(sources, dtype=real_type)
    charges = align_data(charges, dtype=dtype)

    charge_buffer = np.frombuffer(charges, dtype=real_type)

    if dtype == np.complex64:
        lib.evaluate_helmholtz_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(charge_buffer),
            as_float_ptr(result_buffer),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    elif dtype == np.complex128:
        lib.evaluate_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(charge_buffer),
            as_double_ptr(result_buffer),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    else:
        raise NotImplementedError

    if charges.ndim == 1:
        return np.frombuffer(result_buffer, dtype=dtype).reshape(ntargets, ncols)
    else:
        return np.frombuffer(result_buffer, dtype=dtype).reshape(
            ncharge_vecs, ntargets, ncols
        )

def assemble_modified_helmholtz_kernel(sources, targets, omega, dtype=np.float64, parallel=True):
    """Assemble the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    targets = align_data(targets, dtype=dtype)
    sources = align_data(sources, dtype=dtype)

    result = np.empty((ntargets, nsources), dtype=dtype)

    if dtype == np.float32:
        lib.assemble_modified_helmholtz_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(result),
            omega,
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    elif dtype == np.float64:
        lib.assemble_modified_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(result),
            omega,
            as_usize(nsources),
            as_usize(ntargets),
            parallel,
        )
    else:
        raise NotImplementedError

    return result


def evaluate_modified_helmholtz_kernel(
    sources, targets, charges, omega, dtype=np.float64, parallel=True, return_gradients=False
):
    """Evaluate the Laplace kernel matrix for many targets and sources."""

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            f"target must be a 2-dim array of shape (3, ntargets), current shape: {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            f"sources must be a 2-dim array of shape (3, nsources), current shape: {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            f"charges must be a 1- or 2-dim array of shape (...,nsources), current shape: {charges.shape}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if return_gradients:
        ncols = 4
    else:
        ncols = 1

    if charges.ndim == 1:
        ncharge_vecs = 1
        result = np.empty((ntargets, ncols), dtype=dtype)

    else:
        ncharge_vecs = charges.shape[0]
        result = np.empty((ncharge_vecs, ntargets, ncols), dtype=dtype)

    targets = align_data(targets, dtype=dtype)
    sources = align_data(sources, dtype=dtype)
    charges = align_data(charges, dtype=dtype)

    if dtype == np.float32:
        lib.evaluate_modified_helmholtz_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(charges),
            as_float_ptr(result),
            omega,
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    elif dtype == np.float64:
        lib.evaluate_modified_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(charges),
            as_double_ptr(result),
            omega,
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            parallel,
        )
    else:
        raise NotImplementedError

    return result
