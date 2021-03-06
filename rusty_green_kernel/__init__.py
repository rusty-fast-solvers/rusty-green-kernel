"""
Optimized evaluation of direct Greens functions interactions.

This module provides routines to evaluate the Greens functions
for Laplace, Helmholtz, and modified Helmholtz problmes, given
vectors of sources and targets. The underlying routines are
implemented in Rust, multithreaded, and take advantage of SIMD
acceleration.

The implemented kernels are:

Laplace Kernel: g(x, y) = 1 / (4 pi * | x - y |)
Helmhotz Kernel: g(x, y) = exp(1j * k * | x - y |) / (4 pi * | x- y|)
Modified Helmholtz Kernel:
    g(x, y) = exp(-omega * | x - y | ) / (4 pi * | x - y |)

For x = y all kernels are defined to be zero.

The routines starting with `assemble...` evaluate the matrix if interactions
A_ij = g(x_i, y_j), where the x are targets and the y are sources.

The routines starting with `evaluate...` evaluate potential sums of the forms

f(x_i) = sum_j g(x_i, y_j) c_j

for charges c_j. Optionally, also the gradients with respect to the targets x_i
can be evaluated.

"""


import numpy as np
from .rusty_green_kernel import ffi, lib
import os

__all__ = [
    "assemble_laplace_kernel",
    "evaluate_laplace_kernel",
    "assemble_helmholtz_kernel",
    "evaluate_helmholtz_kernel",
    "assemble_modified_helmholtz_kernel",
    "evaluate_modified_helmholtz_kernel",
]

CPU_COUNT = os.cpu_count()


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


def assemble_laplace_kernel(sources, targets, dtype=np.float64, num_threads=CPU_COUNT):
    """
    Assemble the Laplace kernel matrix for many targets and sources.

    Returns the Laplace kernel matrix for the Green's function

    g(x, y) = 1 / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    dtype: dtype object
        Allowed types are np.float64 and np.float32. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.float64 (default)
        or np.float32.
    num_threads: int
        Number of CPU threads to use.

    Outputs
    -------
    A kernel matrix A, such that A[i, j] is the interaction of the
    jth source point with the ith target point using the Laplace Green's
    function.

    """

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
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
            as_usize(num_threads),
        )
    elif dtype == np.float64:
        lib.assemble_laplace_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(result),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(num_threads),
        )
    else:
        raise NotImplementedError

    return result


def evaluate_laplace_kernel(
    sources,
    targets,
    charges,
    dtype=np.float64,
    return_gradients=False,
    num_threads=CPU_COUNT,
):
    """
    Evaluate a potential sum for the Laplace kernel.

    Returns the potential sum

    f(x_i) = sum_j g(x_i, y_j) c_j

    for the Laplace kernel

    g(x, y) = 1 / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    charges: ndarray
        A (m, nsources) array of m charge vectors. The charges
        must be real numbers.
    dtype: dtype object
        Allowed types are np.float64 and np.float32. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.float64 (default)
        or np.float32.
    num_threads: int
        Number of CPU threads to use.
    return_gradients: bool
        If True, return also the gradient of f with respect to the target
        point x_i.

    Outputs
    -------
    An array of A dimension (m, ntargets, 1) if gradients are not requested
    and of dimension (m, ntargets, 4) if gradients are requested. The value
    A[i, j, 0] is the value of the charge potential sum for the ith charge
    vector at the jth target. The values A[i, j, 1:] is the associated
    gradient with respect to the target.

    """

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            "charges must be a 1- or 2-dim array of shape (...,nsources), current"
            f" shape: {charges.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if return_gradients:
        ncols = 4
    else:
        ncols = 1

    if charges.ndim == 1:
        ncharge_vecs = 1
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
            as_usize(num_threads),
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
            as_usize(num_threads),
        )
    else:
        raise NotImplementedError

    return result


def assemble_helmholtz_kernel(
    sources, targets, wavenumber, dtype=np.complex128, num_threads=CPU_COUNT
):
    """
    Assemble the Helmholtz kernel matrix for many targets and sources.

    Returns the Helmholtz kernel matrix for the Green's function

    g(x, y) = exp(1j * k * | x- y| ) / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    wavenumber: complex number
        The wavenumber k. Complex numbers are allowed.
    dtype: dtype object
        Allowed types are np.complex128 and np.complex64. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.complex128 (default)
        or np.complex64.
    num_threads: int
        Number of CPU threads to use.

    Outputs
    -------
    A kernel matrix A, such that A[i, j] is the interaction of the
    jth source point with the ith target point using the Helmholtz Green's
    function.

    """

    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(
            "dtype must be one of [np.complex128, np.complex64], current value:"
            f" {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
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
            as_usize(num_threads),
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
            as_usize(num_threads),
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
    return_gradients=False,
    num_threads=CPU_COUNT,
):
    """
    Evaluate a potential sum for the Helmholtz kernel.

    Returns the potential sum

    f(x_i) = sum_j g(x_i, y_j) c_j

    for the Helmholtz kernel

    g(x, y) = exp( 1j * k * | x - y | ) / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    charges: ndarray
        A (m, nsources) array of m charge vectors. The charges
        are allowed to be complex.
    wavenumber: complex number
        The wavenumber k. Complex numbers are allowed.
    dtype: dtype object
        Allowed types are np.complex128 and np.complex64. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.complex128 (default)
        or np.complex64.
    num_threads: int
        Number of CPU threads to use.
    return_gradients: bool
        If True, return also the gradient of f with respect to the target
        point x_i.

    Outputs
    -------
    An array of A dimension (m, ntargets, 1) if gradients are not requested
    and of dimension (m, ntargets, 4) if gradients are requested. The value
    A[i, j, 0] is the value of the charge potential sum for the ith charge
    vector at the jth target. The values A[i, j, 1:] is the associated
    gradient with respect to the target.

    """

    if dtype not in [np.complex128, np.complex64]:
        raise ValueError(
            "dtype must be one of [np.complex128, np.complex64], current value:"
            f" {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            "charges must be a 1- or 2-dim array of shape (...,nsources), current"
            f" shape: {charges.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
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
    else:
        ncharge_vecs = charges.shape[0]

    result = np.empty((ncharge_vecs, ntargets, ncols), dtype=dtype)

    targets = align_data(targets, dtype=real_type)
    sources = align_data(sources, dtype=real_type)
    charges = align_data(charges, dtype=dtype)

    charge_buffer = np.frombuffer(charges, dtype=real_type)

    if dtype == np.complex64:
        lib.evaluate_helmholtz_kernel_f32(
            as_float_ptr(sources),
            as_float_ptr(targets),
            as_float_ptr(charge_buffer),
            as_float_ptr(result),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            as_usize(num_threads),
        )
    elif dtype == np.complex128:
        lib.evaluate_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(charge_buffer),
            as_double_ptr(result),
            as_double(np.real(wavenumber)),
            as_double(np.imag(wavenumber)),
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(ncharge_vecs),
            return_gradients,
            as_usize(num_threads),
        )
    else:
        raise NotImplementedError

    return np.frombuffer(result, dtype=dtype).reshape(ncharge_vecs, ntargets, ncols)


def assemble_modified_helmholtz_kernel(
    sources, targets, omega, dtype=np.float64, num_threads=CPU_COUNT
):
    """
    Assemble the modified Helmholtz kernel matrix for many targets and sources.

    Returns the modified Helmholtz kernel matrix for the Green's function

    g(x, y) = exp( -omega * | x- y | ) / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    wavenumber: real number
        The wavenumber k. Only real numbers are allowed.
    dtype: dtype object
        Allowed types are np.float64 and np.float32. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.float64 (default)
        or np.float32.
    num_threads: int
        Number of CPU threads to use.

    Outputs
    -------
    A kernel matrix A, such that A[i, j] is the interaction of the
    jth source point with the ith target point using the modified Helmholtz Green's
    function.

    """

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
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
            as_usize(num_threads),
        )
    elif dtype == np.float64:
        lib.assemble_modified_helmholtz_kernel_f64(
            as_double_ptr(sources),
            as_double_ptr(targets),
            as_double_ptr(result),
            omega,
            as_usize(nsources),
            as_usize(ntargets),
            as_usize(num_threads),
        )
    else:
        raise NotImplementedError

    return result


def evaluate_modified_helmholtz_kernel(
    sources,
    targets,
    charges,
    omega,
    dtype=np.float64,
    num_threads=CPU_COUNT,
    return_gradients=False,
):
    """
    Evaluate a potential sum for the modified Helmholtz kernel.

    Returns the potential sum

    f(x_i) = sum_j g(x_i, y_j) c_j

    for the modified Helmholtz kernel

    g(x, y) = exp( -omega * | x - y | ) / (4 pi * | x - y |)

    Parameters
    ----------

    sources: ndarray
        A (3, nsources) array definining nsources sources points.
    targets: ndarray
        A (3, ntargets) array defining ntargets target points.
    charges: ndarray
        A (m, nsources) array of m charge vectors. The charges
        are allowed to be complex.
    wavenumber: real number
        The parameter omega. Only real numbers are allowed.
    dtype: dtype object
        Allowed types are np.float64 and np.float32. If the input
        parameters are not given in the specified type, they are
        converted to it. The output is returned as np.float64 (default)
        or np.float32.
    num_threads: int
        Number of CPU threads to use.
    return_gradients: bool
        If True, return also the gradient of f with respect to the target
        point x_i.

    Outputs
    -------
    An array of A dimension (m, ntargets, 1) if gradients are not requested
    and of dimension (m, ntargets, 4) if gradients are requested. The value
    A[i, j, 0] is the value of the charge potential sum for the ith charge
    vector at the jth target. The values A[i, j, 1:] is the associated
    gradient with respect to the target.

    """

    if dtype not in [np.float64, np.float32]:
        raise ValueError(
            f"dtype must be one of [np.float64, np.float32], current value: {dtype}."
        )

    if targets.ndim != 2 or targets.shape[0] != 3:
        raise ValueError(
            "target must be a 2-dim array of shape (3, ntargets), current shape:"
            f" {targets.shape}."
        )

    if sources.ndim != 2 or sources.shape[0] != 3:
        raise ValueError(
            "sources must be a 2-dim array of shape (3, nsources), current shape:"
            f" {sources.shape}."
        )

    if charges.shape[-1] != sources.shape[1] or charges.ndim > 2:
        raise ValueError(
            "charges must be a 1- or 2-dim array of shape (...,nsources), current"
            f" shape: {charges.shape}."
        )

    if num_threads == 0:
        raise ValueError(
            "number of threads must be larger than zero, current number of threads:"
            f" {num_threads}."
        )

    nsources = sources.shape[1]
    ntargets = targets.shape[1]

    if return_gradients:
        ncols = 4
    else:
        ncols = 1

    if charges.ndim == 1:
        ncharge_vecs = 1
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
            as_usize(num_threads),
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
            as_usize(num_threads),
        )
    else:
        raise NotImplementedError

    return result
