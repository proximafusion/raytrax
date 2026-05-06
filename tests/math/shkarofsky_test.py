import jax
import numpy as np

from raytrax.math import shkarofsky

jax.config.update("jax_enable_x64", True)

_MACHINE_PRECISION = float(np.finfo(float).eps)

# Test data generated using https://github.com/LeiShi/Synthetic-Diagnostics-Platform/
# phi, psi, F_{q+1/2}(phi, psi)

TEST_DATA_F12 = [
    (-10, 0.0, (-0.010050769437519707 + 6.593662989359227e-45j)),
    (-10, -10, (-0.002503136792640365 + 0.0886226925452758j)),
    (-10, -1, (-0.010154979888846649 + 5.884260710583213e-37j)),
    (-10, 10, (-0.002503136792640365 + 0.0886226925452758j)),
    (-10, 1, (-0.010154979888846649 + 5.884260710583213e-37j)),
    (-1, 0.0, (-1.0761590138255368 + 0.6520493321732922j)),
    (-1, -10, (0.010260294560234431 + 5.884260710583213e-36j)),
    (-1, -1, (-0.30134038892379195 + 0.9024587377928238j)),
    (-1, 10, (0.010260294560234431 + 5.884260710583213e-36j)),
    (-1, 1, (-0.30134038892379195 + 0.9024587377928238j)),
    (10, 0.0, (-0.010050769437519707 - 6.593662989359227e-45j)),
    (10, -10, (-0.002503136792640365 - 0.0886226925452758j)),
    (10, -1, (-0.010154979888846649 - 5.884260710583213e-37j)),
    (10, 10, (-0.002503136792640365 - 0.0886226925452758j)),
    (10, 1, (-0.010154979888846649 - 5.884260710583213e-37j)),
    (1, 0.0, (-1.0761590138255368 - 0.6520493321732922j)),
    (1, -10, (0.010260294560234431 - 5.884260710583213e-36j)),
    (1, -1, (-0.30134038892379195 - 0.9024587377928238j)),
    (1, 10, (0.010260294560234431 - 5.884260710583213e-36j)),
    (1, 1, (-0.30134038892379195 - 0.9024587377928238j)),
    ((-0 - 10j), 0.0, (0.009950731878244697 - 0j)),
    ((-0 - 10j), -10, (0.005012405099081152 - 0j)),
    ((-0 - 10j), -1, (0.009854545575166438 - 0j)),
    ((-0 - 10j), 10, (0.005012405099081152 - 0j)),
    ((-0 - 10j), 1, (0.009854545575166438 - 0j)),
    ((-0 - 1j), 0.0, (0.757872156141312 - 0j)),
    ((-0 - 1j), -10, (0.010049711537118873 - 0j)),
    ((-0 - 1j), -1, (0.5401450401487555 - 0j)),
    ((-0 - 1j), 10, (0.010049711537118873 - 0j)),
    ((-0 - 1j), 1, (0.5401450401487555 - 0j)),
    (10j, 0.0, (-9.529127159394277e42 + 0j)),
    (10j, -10, (-0.16769112903689334 + 0j)),
    (10j, -1, (-1.430560222922888e42 + 0j)),
    (10j, 10, (-0.16769112903689334 + 0j)),
    (10j, 1, (-1.430560222922888e42 + 0j)),
    (1j, 0.0, (-8.878186033256132 + 0j)),
    (1j, -10, (0.010049711537118873 + 0j)),
    (1j, -1, (2.015347166109017 + 0j)),
    (1j, 10, (0.010049711537118873 + 0j)),
    (1j, 1, (2.015347166109017 + 0j)),
]

TEST_DATA_F32 = [
    (-10, 0.0, (-0.010153887503941306 + 1.3187325978718453e-42j)),
    (-10, -10, (0.002503136792640365 + 0.0886226925452758j)),
    (-10, -1, (-0.010260294560234431 + 5.884260710583213e-36j)),
    (-10, 10, (0.002503136792640365 + 0.0886226925452758j)),
    (-10, 1, (-0.010260294560234431 + 5.884260710583213e-36j)),
    (-1, 0.0, (-0.1523180276510736 + 1.3040986643465844j)),
    (-1, -10, (0.010154979888846649 + 5.884260710583213e-37j)),
    (-1, -1, (0.30134038892379195 + 0.8699951131126921j)),
    (-1, 10, (0.010154979888846649 + 5.884260710583213e-37j)),
    (-1, 1, (0.30134038892379195 + 0.8699951131126921j)),
    (10, 0.0, (-0.010153887503941306 - 1.3187325978718453e-42j)),
    (10, -10, (0.002503136792640365 - 0.0886226925452758j)),
    (10, -1, (-0.010260294560234431 - 5.884260710583213e-36j)),
    (10, 10, (0.002503136792640365 - 0.0886226925452758j)),
    (10, 1, (-0.010260294560234431 - 5.884260710583213e-36j)),
    (1, 0.0, (-0.1523180276510736 - 1.3040986643465844j)),
    (1, -10, (0.010154979888846649 - 5.884260710583213e-37j)),
    (1, -1, (0.30134038892379195 - 0.8699951131126921j)),
    (1, 10, (0.010154979888846649 - 5.884260710583213e-37j)),
    (1, 1, (0.30134038892379195 - 0.8699951131126921j)),
    ((-0 - 10j), 0.0, (0.009853624351060741 - 0j)),
    ((-0 - 10j), -10, (0.004987407441909125 + 0j)),
    ((-0 - 10j), -1, (0.0097592719135322 + 0j)),
    ((-0 - 10j), 10, (0.004987407441909125 - 0j)),
    ((-0 - 10j), 1, (0.0097592719135322 - 0j)),
    ((-0 - 1j), 0.0, (0.48425568771737604 - 0j)),
    ((-0 - 1j), -10, (0.009948720599021858 + 0j)),
    ((-0 - 1j), -1, (0.3690584588490665 + 0j)),
    ((-0 - 1j), 10, (0.009948720599021858 - 0j)),
    ((-0 - 1j), 1, (0.3690584588490665 - 0j)),
    (10j, 0.0, (1.9058254318788555e45 - 0j)),
    (10j, -10, (-0.3045884240445556 - 0j)),
    (10j, -1, (3.200393459084526e43 + 0j)),
    (10j, 10, (-0.3045884240445556 + 0j)),
    (10j, 1, (3.200393459084526e43 - 0j)),
    (1j, 0.0, (19.756372066512263 - 0j)),
    (1j, -10, (0.009948720599021858 + 0j)),
    (1j, -1, (3.5924339104403784 + 0j)),
    (1j, 10, (0.009948720599021858 - 0j)),
    (1j, 1, (3.5924339104403784 - 0j)),
]

TEST_DATA_F52 = [
    (-10, 0.0, (-0.010259166929420378 + 8.791550652478968e-41j)),
    (-10, -10, (0.007484347523396433 + 0.08817957908254943j)),
    (-10, -1, (-0.0103678416045477 + 5.590047675054051e-35j)),
    (-10, 10, (0.007484347523396433 + 0.08817957908254943j)),
    (-10, 1, (-0.0103678416045477 + 5.590047675054051e-35j)),
    (-1, 0.0, (0.5651213148992842 + 0.8693991095643896j)),
    (-1, -10, (0.01005182804615811 + 5.590047675054052e-38j)),
    (-1, -1, (0.547989416614312 + 0.4674611812364778j)),
    (-1, 10, (0.01005182804615811 + 5.590047675054052e-38j)),
    (-1, 1, (0.547989416614312 + 0.4674611812364778j)),
    (10, 0.0, (-0.010259166929420378 - 8.791550652478968e-41j)),
    (10, -10, (0.007484347523396433 - 0.08817957908254943j)),
    (10, -1, (-0.0103678416045477 - 5.590047675054051e-35j)),
    (10, 10, (0.007484347523396433 - 0.08817957908254943j)),
    (10, 1, (-0.0103678416045477 - 5.590047675054051e-35j)),
    (1, 0.0, (0.5651213148992842 - 0.8693991095643896j)),
    (1, -10, (0.01005182804615811 - 5.590047675054052e-38j)),
    (1, -1, (0.547989416614312 - 0.4674611812364778j)),
    (1, 10, (0.01005182804615811 - 5.590047675054052e-38j)),
    (1, 1, (0.547989416614312 - 0.4674611812364778j)),
    ((-0 - 10j), 0.0, (0.00975837659595058 + 0j)),
    ((-0 - 10j), -10, (0.0049626578637093025 + 0j)),
    ((-0 - 10j), -1, (0.009665806526590107 + 0j)),
    ((-0 - 10j), 10, (0.0049626578637093025 + 0j)),
    ((-0 - 10j), 1, (0.009665806526590107 + 0j)),
    ((-0 - 1j), 0.0, (0.34382954152174927 + 0j)),
    ((-0 - 1j), -10, (0.009849759281633702 + 0j)),
    ((-0 - 1j), -1, (0.2753257304267112 + 0j)),
    ((-0 - 1j), 10, (0.009849759281633702 + 0j)),
    ((-0 - 1j), 1, (0.2753257304267112 + 0j)),
    (10j, 0.0, (-1.2705502879192371e47 + 0j)),
    (10j, -10, (0.1792140711571161 + 0j)),
    (10j, -1, (1.2705405499686617e44 + 0j)),
    (10j, 10, (0.1792140711571161 + 0j)),
    (10j, 1, (1.2705405499686617e44 + 0j)),
    (1j, 0.0, (-12.504248044341509 + 0j)),
    (1j, -10, (0.009849759281633702 + 0j)),
    (1j, -1, (-2.811564121329206 + 0j)),
    (1j, 10, (0.009849759281633702 + 0j)),
    (1j, 1, (-2.811564121329206 + 0j)),
]


def _sequence_value(phi, psi, q):
    return shkarofsky._shkarofsky_sequence(psi, phi, q_max=q)[q]


def test_shkarofsky_sequence_f12():
    """Test the Shkarofsky function F_{1/2}."""
    for phi, psi, expected in TEST_DATA_F12:
        result = _sequence_value(phi=phi, psi=psi, q=0)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"F_{1 / 2} failed for phi={phi}, psi={psi}",
        )


def test_shkarofsky_sequence_f32():
    """Test the Shkarofsky function F_{3/2}."""
    for phi, psi, expected in TEST_DATA_F32:
        result = _sequence_value(phi=phi, psi=psi, q=1)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"F_{3 / 2} failed for phi={phi}, psi={psi}",
        )


def test_shkarofsky_sequence_f52():
    """Test the Shkarofsky function F_{5/2}."""
    for phi, psi, expected in TEST_DATA_F52:
        result = _sequence_value(phi=phi, psi=psi, q=2)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"F_{5 / 2} failed for phi={phi}, psi={psi}",
        )


def test_shkarofsky_sequence_f12_small_psi():
    """Check the continuity of F_{1/2} at its base-case threshold (_PSI_TOLERANCE_BASE).

    F_{1/2} uses _PSI_TOLERANCE_BASE (1e-6) to switch between the Z-function
    base-case formula and its psi->0 limit.  Both sides of this threshold must
    agree to machine precision.  The recurrence threshold (_PSI_TOLERANCE_RECUR
    = 0.05) does NOT affect F_{1/2}, so there is no discontinuity there.
    """
    base_limit = shkarofsky._PSI_TOLERANCE_BASE
    epsilon = 1e-10
    value_zero = shkarofsky._shkarofsky_sequence(0.0, 1.0, q_max=0)[0]
    value_epsilon = shkarofsky._shkarofsky_sequence(epsilon, 1.0, q_max=0)[0]
    value_below_limit = shkarofsky._shkarofsky_sequence(
        base_limit - epsilon, 1.0, q_max=0
    )[0]
    value_above_limit = shkarofsky._shkarofsky_sequence(
        base_limit + epsilon, 1.0, q_max=0
    )[0]
    # All four should agree to machine precision: F_{1/2} is smooth and the
    # nonzero branch is exact at psi = base_limit + epsilon.
    np.testing.assert_allclose(
        value_zero,
        value_epsilon,
        rtol=0,
        atol=_MACHINE_PRECISION,
    )
    np.testing.assert_allclose(
        value_zero,
        value_below_limit,
        rtol=0,
        atol=_MACHINE_PRECISION,
    )
    np.testing.assert_allclose(
        value_above_limit,
        value_zero,
        rtol=0,
        atol=1e-10,
    )


def test_shkarofsky_sequence_f72_zero_branch_accuracy():
    """F_{7/2} (q=3) zero-branch recurrence is accurate below _PSI_TOLERANCE_RECUR.

    Checks that:
    - At psi=1e-3 (well below the threshold) the result agrees with the exact
      psi=0 value to within O(psi^2) ~ 1e-4 relative error.
    - At psi=1e-4 (also below the threshold) the zero-branch gives the same
      accuracy, confirming it does not degrade for very small psi.
    """
    phi = complex(1.0, 0.0)

    value_zero = shkarofsky._shkarofsky_sequence(0.0, phi, q_max=3)[3]

    # Well below threshold (psi=1e-3): zero-branch recurrence, should agree
    # closely with the psi=0 result (O(psi^2) = O(1e-6) error).
    value_small_psi = shkarofsky._shkarofsky_sequence(1e-3, phi, q_max=3)[3]
    np.testing.assert_allclose(value_small_psi, value_zero, rtol=1e-4)

    # Also check at psi=1e-4 (even further below threshold): zero-branch still
    # accurate.
    tiny_psi = 1e-4
    value_tiny_zero_branch = shkarofsky._shkarofsky_sequence(tiny_psi, phi, q_max=3)[3]
    np.testing.assert_allclose(
        value_tiny_zero_branch,
        value_zero,
        rtol=1e-4,
        err_msg="Zero-branch recurrence should be accurate at psi << PSI_TOLERANCE_RECUR",
    )


# Reference values for F_{7/2}(phi, psi) computed with the fixed raytrax
# implementation (validated to match the external LeiShi/SDP reference for
# q=0..2 above).  Tuple layout: (phi, psi, expected F_{7/2}).
# The small-psi entries (|psi| = 1e-3, 1e-6) directly exercise the
# zero-branch recurrence path that was added to fix Bug 2; with the old
# naive recurrence they returned values ~10^8 times too large.
TEST_DATA_F72 = [
    (-1, 0.0, (0.62604853 + 0.34775964j)),
    (-1, -1, (0.47935626 + 0.16880334j)),
    (-1, 1, (0.47935626 + 0.16880334j)),
    (1, 0.0, (0.62604853 - 0.34775964j)),
    (1, -1, (0.47935626 - 0.16880334j)),
    (1, 1, (0.47935626 - 0.16880334j)),
    (-10, 0.0, (-0.01036668 + 3.5166e-39j)),
    (-10, -10, (0.01239087 + 0.0873j)),
    (10, 0.0, (-0.01036668 - 3.5166e-39j)),
    # Small psi: the critical regime for Bug 2.
    (-1, 1e-3, (0.62604872 + 0.34775953j)),
    (-1, 1e-6, (0.62604853 + 0.34775964j)),
    (1, 1e-3, (0.62604872 - 0.34775953j)),
    (1, 1e-6, (0.62604853 - 0.34775964j)),
]


def test_shkarofsky_sequence_f72_reference_values():
    """F_{7/2} (q=3) must match reference values, including in the small-psi regime.

    This is the primary regression test for Bug 2 (Shkarofsky instability).
    The small-psi entries (|psi| = 1e-3, 1e-6) directly exercise the
    zero-branch recurrence path that was added to fix the bug; they would
    return wildly wrong results (~10^8 off) with the old code.
    """
    for phi, psi, expected in TEST_DATA_F72:
        result = shkarofsky._shkarofsky_sequence(psi, phi, q_max=3)[3]
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-3,  # 0.1% — small-psi branch is O(psi^2) accurate
            atol=0.0,
            err_msg=f"F_{{7/2}} failed for phi={phi}, psi={psi}",
        )
