"""Unit tests for direct assembly and evaluation of kernels."""
import numpy as np
import pytest


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_assemble(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from rusty_green_kernel import assemble_laplace_kernel

    nsources = 10
    ntargets = 20

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target

    actual = assemble_laplace_kernel(sources, targets, dtype=dtype, parallel=parallel)

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((ntargets, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        expected[index, :] = 1.0 / (
            4 * np.pi * np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        )

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_only_values(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from rusty_green_kernel import evaluate_laplace_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_laplace_kernel(
        sources, targets, charges, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((nsources, ntargets), dtype=dtype)

    for index, target in enumerate(targets.T):
        expected[:, index] = 1.0 / (
            4 * np.pi * np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        )

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    expected = np.expand_dims(charges @ expected, -1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_laplace_evaluate_values_and_deriv(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from rusty_green_kernel import evaluate_laplace_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_laplace_kernel(
        sources, targets, charges, dtype=dtype, return_gradients=True, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets, 4), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = sources - target.reshape(3, 1)
        dist = np.linalg.norm(diff, axis=0)
        expected[:, index, 0] = 1.0 / (4 * np.pi * dist)
        expected[:, index, 1:] = diff.T / (4 * np.pi * dist.reshape(nsources, 1) ** 3)
        expected[dist == 0, index, :] = 0

    # Reset the warnings
    np.seterr(**old_params)

    expected = np.tensordot(charges, expected, 1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_assemble(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from rusty_green_kernel import assemble_helmholtz_kernel

    wavenumber = 2.5

    nsources = 10
    ntargets = 20

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target

    actual = assemble_helmholtz_kernel(
        sources, targets, wavenumber, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((ntargets, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[index, :] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[index, dist == 0] = 0

    # Reset the warnings
    np.seterr(**old_params)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_evaluate_only_values(dtype, rtol):
    """Test the Laplace kernel."""
    from rusty_green_kernel import evaluate_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 2.5 + 1.3j

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=real_type) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=real_type
    )

    actual = evaluate_helmholtz_kernel(
        sources, targets, charges, wavenumber, dtype=dtype, parallel=False
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[:, index] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[dist == 0, index] = 0

    # Reset the warnings
    np.seterr(**old_param)

    expected = np.expand_dims(np.tensordot(charges, expected, 1), -1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.complex128, 1e-14), (np.complex64, 5e-6)])
def test_helmholtz_evaluate_values_and_deriv(dtype, rtol, parallel):
    """Test the Laplace kernel."""
    from rusty_green_kernel import evaluate_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 2.5 + 1.3j

    if dtype == np.complex128:
        real_type = np.float64
    elif dtype == np.complex64:
        real_type = np.float32
    else:
        raise ValueError(f"Unsupported type: {dtype}.")

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=real_type)
    sources = rng.random((3, nsources), dtype=real_type)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=real_type) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=real_type
    )

    actual = evaluate_helmholtz_kernel(
        sources,
        targets,
        charges,
        wavenumber,
        dtype=dtype,
        return_gradients=True,
        parallel=parallel,
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets, 4), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = target.reshape(3, 1) - sources
        dist = np.linalg.norm(diff, axis=0)
        expected[:, index, 0] = np.exp(1j * wavenumber * dist) / (4 * np.pi * dist)
        expected[:, index, 1:] = (
            diff.T
            * expected[:, index, 0].reshape(nsources, 1)
            / dist.reshape(nsources, 1) ** 2
            * (1j * wavenumber * dist.reshape(nsources, 1) - 1)
        )
        expected[dist == 0, index, :] = 0

    # Reset the warnings
    np.seterr(**old_params)

    expected = np.tensordot(charges, expected, 1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_modified_helmholtz_assemble(dtype, rtol, parallel):
    """Test the modified Helmholtz kernel."""
    from rusty_green_kernel import assemble_modified_helmholtz_kernel

    nsources = 10
    ntargets = 20

    omega = 2.5

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target

    actual = assemble_modified_helmholtz_kernel(
        sources, targets, omega, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((ntargets, nsources), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[index, :] = np.exp(-omega * dist) / (4 * np.pi * dist)

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_modified_helmholtz_evaluate_only_values(dtype, rtol, parallel):
    """Test the modified Helmholtz kernel."""
    from rusty_green_kernel import evaluate_modified_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    omega = 2.5

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_modified_helmholtz_kernel(
        sources, targets, charges, omega, dtype=dtype, parallel=parallel
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_param = np.geterr()["divide"]
    np.seterr(divide="ignore")

    expected = np.empty((nsources, ntargets), dtype=dtype)

    for index, target in enumerate(targets.T):
        dist = np.linalg.norm(sources - target.reshape(3, 1), axis=0)
        expected[:, index] = np.exp(-omega * dist) / (4 * np.pi * dist)

    # Reset the warnings
    np.seterr(divide=old_param)

    expected[0, 0] = 0  # First source and target are identical.

    expected = np.expand_dims(charges @ expected, -1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype,rtol", [(np.float64, 1e-14), (np.float32, 5e-6)])
def test_modified_helmholtz_evaluate_values_and_deriv(dtype, rtol, parallel):
    """Test the modified Helmholtz kernel."""
    from rusty_green_kernel import evaluate_modified_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    omega = 2.5

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype)

    actual = evaluate_modified_helmholtz_kernel(
        sources,
        targets,
        charges,
        omega,
        dtype=dtype,
        return_gradients=True,
        parallel=parallel,
    )

    # Calculate expected result

    # A divide by zero error is expected to happen here.
    # So just ignore the warning.
    old_params = np.geterr()
    np.seterr(all="ignore")

    expected = np.empty((nsources, ntargets, 4), dtype=dtype)

    for index, target in enumerate(targets.T):
        diff = target.reshape(3, 1) - sources
        dist = np.linalg.norm(diff, axis=0)
        expected[:, index, 0] = np.exp(-omega * dist) / (4 * np.pi * dist)
        expected[:, index, 1:] = (
            diff.T
            / (4 * np.pi * dist.reshape(nsources, 1) ** 3)
            * np.exp(-omega * dist.reshape(nsources, 1))
            * (-omega * dist.reshape(nsources, 1) - 1)
        )
        expected[dist == 0, index, :] = 0

    # Reset the warnings
    np.seterr(**old_params)

    expected = np.tensordot(charges, expected, 1)

    np.testing.assert_allclose(actual, expected, rtol=rtol)


def test_laplace_derivative_is_correct():
    """Test that the Gradient of the Laplace kernel is correct."""
    from rusty_green_kernel import evaluate_laplace_kernel

    nsources = 10

    eps = 1e-10

    dtype = np.float64

    targets = np.array(
        [
            [1.1, 1.5, 2.3],
            [1.1 + eps, 1.5, 2.3],
            [1.1 - eps, 1.5, 2.3],
            [1.1, 1.5 + eps, 2.3],
            [1.1, 1.5 - eps, 2.3],
            [1.1, 1.5, 2.3 + eps],
            [1.1, 1.5, 2.3 - eps],
        ]
    ).T

    rng = np.random.default_rng(seed=0)

    sources = rng.random((3, nsources), dtype=dtype)
    charges = rng.random((1, nsources), dtype=dtype)

    # Evalute derivative approximately.

    values = evaluate_laplace_kernel(sources, targets, charges)

    x_deriv = (values[0, 1, 0] - values[0, 2, 0]) / (2 * eps)
    y_deriv = (values[0, 3, 0] - values[0, 4, 0]) / (2 * eps)
    z_deriv = (values[0, 5, 0] - values[0, 6, 0]) / (2 * eps)

    expected = np.array([x_deriv, y_deriv, z_deriv])

    actual = evaluate_laplace_kernel(sources, targets, charges, return_gradients=True)[
        0, 0, 1:
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_helmholtz_derivative_is_correct():
    """Test that the Gradient of the Helmholtz kernel is correct."""
    from rusty_green_kernel import evaluate_helmholtz_kernel

    nsources = 10

    wavenumber = 2.5 + 1.3j

    eps = 1e-10

    dtype = np.float64

    targets = np.array(
        [
            [1.1, 1.5, 2.3],
            [1.1 + eps, 1.5, 2.3],
            [1.1 - eps, 1.5, 2.3],
            [1.1, 1.5 + eps, 2.3],
            [1.1, 1.5 - eps, 2.3],
            [1.1, 1.5, 2.3 + eps],
            [1.1, 1.5, 2.3 - eps],
        ]
    ).T

    rng = np.random.default_rng(seed=0)

    sources = rng.random((3, nsources), dtype=dtype)
    charges = rng.random((1, nsources), dtype=dtype)

    # Evalute derivative approximately.

    values = evaluate_helmholtz_kernel(sources, targets, charges, wavenumber)

    x_deriv = (values[0, 1, 0] - values[0, 2, 0]) / (2 * eps)
    y_deriv = (values[0, 3, 0] - values[0, 4, 0]) / (2 * eps)
    z_deriv = (values[0, 5, 0] - values[0, 6, 0]) / (2 * eps)

    expected = np.array([x_deriv, y_deriv, z_deriv])

    actual = evaluate_helmholtz_kernel(
        sources, targets, charges, wavenumber, return_gradients=True
    )[0, 0, 1:]

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_modified_helmholtz_derivative_is_correct():
    """Test that the Gradient of the Helmholtz kernel is correct."""
    from rusty_green_kernel import evaluate_modified_helmholtz_kernel

    nsources = 10

    omega = 1.3

    eps = 1e-10

    dtype = np.float64

    targets = np.array(
        [
            [1.1, 1.5, 2.3],
            [1.1 + eps, 1.5, 2.3],
            [1.1 - eps, 1.5, 2.3],
            [1.1, 1.5 + eps, 2.3],
            [1.1, 1.5 - eps, 2.3],
            [1.1, 1.5, 2.3 + eps],
            [1.1, 1.5, 2.3 - eps],
        ]
    ).T

    rng = np.random.default_rng(seed=0)

    sources = rng.random((3, nsources), dtype=dtype)
    charges = rng.random((1, nsources), dtype=dtype)

    # Evalute derivative approximately.

    values = evaluate_modified_helmholtz_kernel(sources, targets, charges, omega)

    x_deriv = (values[0, 1, 0] - values[0, 2, 0]) / (2 * eps)
    y_deriv = (values[0, 3, 0] - values[0, 4, 0]) / (2 * eps)
    z_deriv = (values[0, 5, 0] - values[0, 6, 0]) / (2 * eps)

    expected = np.array([x_deriv, y_deriv, z_deriv])

    actual = evaluate_modified_helmholtz_kernel(
        sources, targets, charges, omega, return_gradients=True
    )[0, 0, 1:]

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_helmholtz_at_zero_agrees_with_laplace():
    """Test if Helmholtz with wavenumber 0 agrees with Laplace."""
    from rusty_green_kernel import evaluate_helmholtz_kernel
    from rusty_green_kernel import evaluate_laplace_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 0

    dtype = np.float64

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=dtype
    )

    values_helmholtz = evaluate_helmholtz_kernel(
        sources, targets, charges, wavenumber, return_gradients=True
    )
    values_laplace = evaluate_laplace_kernel(
        sources, targets, np.real(charges), return_gradients=True
    ) + 1j * evaluate_laplace_kernel(
        sources, targets, np.imag(charges), return_gradients=True
    )

    np.testing.assert_allclose(values_helmholtz, values_laplace, rtol=1E-14)

def test_helmholtz_imaginary_wavenumber_agrees_with_modified_helmholtz():
    """Test if Helmholtz with wavenumber 0 agrees with Laplace."""
    from rusty_green_kernel import evaluate_helmholtz_kernel
    from rusty_green_kernel import evaluate_modified_helmholtz_kernel

    nsources = 10
    ntargets = 20
    ncharge_vecs = 2

    wavenumber = 1.3j

    dtype = np.float64

    rng = np.random.default_rng(seed=0)
    # Construct target and sources so that they do not overlap
    # apart from the first point.

    targets = 1.5 + rng.random((3, ntargets), dtype=dtype)
    sources = rng.random((3, nsources), dtype=dtype)
    sources[:, 0] = targets[:, 0]  # Test what happens if source = target
    charges = rng.random((ncharge_vecs, nsources), dtype=dtype) + 1j * rng.random(
        (ncharge_vecs, nsources), dtype=dtype
    )

    values_helmholtz = evaluate_helmholtz_kernel(
        sources, targets, charges, wavenumber, return_gradients=True
    )
    values_modified_helmholtz = evaluate_modified_helmholtz_kernel(
        sources, targets, np.real(charges), np.imag(wavenumber), return_gradients=True
    ) + 1j * evaluate_modified_helmholtz_kernel(
        sources, targets, np.imag(charges), np.imag(wavenumber), return_gradients=True
    )

    np.testing.assert_allclose(values_helmholtz, values_modified_helmholtz, rtol=1E-14)