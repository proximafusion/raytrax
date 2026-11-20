"""Cold plasma dispersion relation (Appleton-Hartree) and dispersion tensor in Stix coordinates."""

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt

from raytrax.types import ScalarFloat


def dispersion_cold(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    frequency: ScalarFloat,
    cyclotron_frequency: ScalarFloat,
    plasma_frequency: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    """Computes the refractive index squared (N^2) from the dispersion relation of a
    cold magnetized plasma.

    Args:
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        frequency: Frequency of the wave (in Hertz!).
        cyclotron_frequency: Cyclotron frequency of electrons (in Hertz!).
        plasma_frequency: Plasma frequency of electrons (in Hertz!).
        mode: 'X' or 'O' for extraordinary or ordinary mode.
    """
    X = plasma_frequency**2 / frequency**2
    Y = cyclotron_frequency / frequency

    # Handle the case where refractive_index_para is 0 (perpendicular propagation)
    # When refractive_index_para is 0, the angle is 90 degrees, so sin2theta = 1
    safe_para = jnp.where(refractive_index_para == 0, 1.0, refractive_index_para)
    tantheta = refractive_index_perp / safe_para
    sin2theta = jnp.where(
        refractive_index_para == 0,
        1.0,  # sin²(90°) = 1
        tantheta**2 / (1 + tantheta**2),
    )
    return _dispersion_appleton_hartee(X=X, Y=Y, sin2theta=sin2theta, mode=mode)


def _dispersion_appleton_hartee(
    X: ScalarFloat,
    Y: ScalarFloat,
    sin2theta: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    """Computes the refractive index squared (N^2) using the Appleton-Hartree formula
    valid for electromagnetic wave propagation in a cold magnetized plasma.

    See e.g. https://en.wikipedia.org/wiki/Appleton%E2%80%93Hartree_equation

    Args:
        X: (ω_pe / ω)^2, normalized plasma frequency squared.
        Y: ω_ce / ω, normalized cyclotron frequency.
        sin2theta: sine squared of the angle between wave vector and magnetic field.
        mode: 'X' or 'O' for extraordinary or ordinary mode.

    Returns:
        The refractive index squared N^2.
    """
    cos2theta = 1 - sin2theta
    X1 = 1 - X
    D = Y**4 * sin2theta**2 + 4 * X1**2 * Y**2 * cos2theta
    a = 2 * X * X1
    b = 2 * X1 - Y**2 * sin2theta

    G = jnp.sqrt(D)
    sign = -1 if mode == "X" else +1
    # N^2 = 1 - a / d = 1 - X (b - sign * G) / (2 c) with d = b + sign * G, using
    # b^2 - G^2 = 4 (1 - X) c. The first form cancels where d -> 0 (O-mode at X = 1),
    # the second where c -> 0. Blending them with weights d^2 and (2 c)^2 is exact and
    # suppresses whichever one cancels.
    c = X1 - Y**2 * sin2theta - X1 * Y**2 * cos2theta
    d = b + sign * G
    N_squared = 1 - (a * d + 2 * X * c * (b - sign * G)) / (d**2 + 4 * c**2)
    return N_squared


def refractive_index_perp_squared_cold(
    refractive_index_para_squared: ScalarFloat,
    X: ScalarFloat,
    Y: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    r"""Computes $N_\perp^2$ of a cold-plasma mode at given $N_\parallel^2$.

    At fixed $N_\parallel$ the cold dispersion relation is a quadratic in
    $N_\perp^2$. With $P = 1 - X$ and $R$, $L$, $S$ multiplied by $1 - Y$, $1 + Y$
    and $1 - Y^2$ to clear their poles at $Y = 1$,

    $$\tilde R = 1 - Y - X, \quad \tilde L = 1 + Y - X, \quad \tilde S = 1 - Y^2 - X,$$

    it reads $\tilde S N_\perp^4 + b N_\perp^2 + c = 0$ with

    $$b = (\tilde S + (1 - Y^2) P) N_\parallel^2 - \tilde R \tilde L - P \tilde S,
    \quad c = P \left((1 - Y) N_\parallel^2 - \tilde R\right)
    \left((1 + Y) N_\parallel^2 - \tilde L\right).$$

    The discriminant factorises as

    $$b^2 - 4 \tilde S c = X^2 Y^2 \left(Y^2 (1 - N_\parallel^2)^2
    + 4 N_\parallel^2 (1 - X)\right),$$

    which is evaluated in this form: the difference $b^2 - 4 \tilde S c$ loses all
    digits at low density, where both terms are of order one and the discriminant
    of order $X^2$. The root with the $+$ sign of the square root is the O-mode,
    the other one the X-mode, for every $Y$: the ordering of the roots and the sign
    of $\tilde S$ both flip at $Y = 1$. Unlike the Appleton-Hartree formula, which
    parametrises the modes by the angle to the magnetic field, this form has no
    branch point at the O-mode cutoff $X = 1$, $N_\perp = 0$, where every reflected
    O-mode ray passes. The discriminant is non-negative for $X \le 1$; for $X > 1$
    there are regions of $N_\parallel$ where both roots are complex, no real wave
    exists, and the result is NaN.

    Args:
        refractive_index_para_squared: $N_\parallel^2$, squared refractive index
            along the magnetic field.
        X: (ω_pe / ω)^2, normalized plasma frequency squared.
        Y: ω_ce / ω, normalized cyclotron frequency.
        mode: 'X' or 'O' for extraordinary or ordinary mode.

    Returns:
        $N_\perp^2$ of the mode. Negative where the mode is evanescent.
    """
    P = 1 - X
    one_minus_y2 = (1 - Y) * (1 + Y)  # exact to rounding also close to Y = 1
    R = 1 - Y - X
    L = 1 + Y - X
    S = one_minus_y2 - X
    n_para2 = refractive_index_para_squared
    a = S
    b = (S + one_minus_y2 * P) * n_para2 - R * L - P * S
    c = P * ((1 - Y) * n_para2 - R) * ((1 + Y) * n_para2 - L)
    sign = 1.0 if mode == "O" else -1.0
    # sqrt(b^2 - 4 a c), with X, Y >= 0
    root = X * Y * jnp.sqrt(Y**2 * (1 - n_para2) ** 2 + 4 * n_para2 * P)
    # The root (-b + sign * root) / (2 a) equals 2 c / (-b - sign * root). The first
    # form cancels where sign * b > 0, the second where sign * b < 0 (and where
    # b = root = 0, in vacuum). Use the one in which nothing cancels. The two
    # jnp.where keep the derivative of the unused form from turning into NaN.
    use_second = sign * b >= 0
    numerator = jnp.where(use_second, 2 * c, -b + sign * root)
    denominator = jnp.where(use_second, -b - sign * root, 2 * a)
    denominator = jnp.where(denominator == 0, 1.0, denominator)
    return numerator / denominator


def dispersion_relation_cold(
    refractive_index_perp_squared: ScalarFloat,
    refractive_index_para_squared: ScalarFloat,
    X: ScalarFloat,
    Y: ScalarFloat,
    mode: Literal["X", "O"],
) -> ScalarFloat:
    r"""The cold-plasma ray Hamiltonian $\mathcal H_\sigma$: the wave determinant normalised by the launch mode.

    A plane wave satisfies $\mathsf M \mathbf E = 0$ with
    $\mathsf M = \boldsymbol\epsilon + \mathbf N \mathbf N^{\mathsf T} - N^2 \mathsf I$.
    At given $X$, $Y$ and $N_\parallel$ the determinant is a quadratic in $N_\perp^2$,
    whose two roots are the two values of $N_\perp^2$ that satisfy Maxwell's equations
    there. With $a = \tilde S = 1 - Y^2 - X$ and $b$, $c$ of
    `refractive_index_perp_squared_cold`,

    $$\tilde D = (1 - Y^2) \det \mathsf M = a N_\perp^4 + b N_\perp^2 + c
    = a\,(N_\perp^2 - n_{\perp,+}^2)(N_\perp^2 - n_{\perp,-}^2), \qquad
    n_{\perp,\pm}^2 = \frac{-b \pm X Y \sqrt g}{2a}, \qquad
    g = Y^2 (1 - N_\parallel^2)^2 + 4 N_\parallel^2 (1 - X).$$

    $\tilde D$ is regular everywhere, but it cannot start a ray: in vacuum it is
    $(1 - Y^2)(N^2 - 1)^2$, a double zero with no gradient on the ray. The
    Hamiltonian of one root is the determinant divided by the factor of the other,

    $$N_\perp^2 - n_{\perp,\sigma}^2
    = \frac{\tilde D}{a\,(N_\perp^2 - n_{\perp,-\sigma}^2)}
    = \frac{2 \tilde D}{2 a N_\perp^2 + b + \sigma X Y \sqrt g},$$

    with $\sigma = +1$ for the O and $-1$ for the X mode, for every $Y$. The division
    removes the extra vacuum factor and keeps the splitting of the modes, which is of
    first order in $X$. Its divisor, however, is $2 \sigma X Y \sqrt g$ on the ray: it
    vanishes at the fold $g = 0$, where the two roots coincide at finite $N_\perp$
    (beyond the cutoff $X = 1$; the X mode at $Y > 1$ gets there obliquely, turns in
    density and continues on the other root), and $\sqrt g$ is NaN beyond. Here only
    that square root is changed. It is taken at zero density, at the ray's
    $N_\parallel$ and the local $Y$:

    $$\mathcal H_\sigma = \frac{2 \tilde D}{2 a N_\perp^2 + b + \sigma X Y \sqrt{g_0}},
    \qquad g_0 = Y^2 (1 - N_\parallel^2)^2 + 4 N_\parallel^2 > 0.$$

    - At the edge $g_0 - g = 4 N_\parallel^2 X$ and the square root carries a factor
      $X$, so the divisor changes at order $X^2$: $\mathcal H_\sigma$ and
      $N_\perp^2 - n_{\perp,\sigma}^2$ agree to first order in the density, and the
      launch mode is the one of $\sigma$. At zero density $\mathcal H_\sigma = N^2 - 1$.
    - On the sheet of the launch mode $2 a N_\perp^2 + b = \sigma X Y \sqrt g$, the
      divisor is $\sigma X Y (\sqrt g + \sqrt{g_0})$, nonzero at every density, and
      $\mathcal H_\sigma = (N_\perp^2 - n_{\perp,\sigma}^2)\, 2 \sqrt g / (\sqrt g + \sqrt{g_0})$:
      the rays of the single-root Hamiltonian with a rescaled parameter.
    - At the fold $2 a N_\perp^2 + b = 0$ and the divisor is $\sigma X Y \sqrt{g_0}$.
      Locally $\mathcal H_\sigma$ has the characteristics of the determinant. It
      vanishes on both roots, on the incident and on the return leg; $\sigma$ labels
      the launch, not the root the ray is on.
    - Off the dispersion surface the divisor has a zero, at a distance of order $X^2$
      from the other root, where $\mathcal H_\sigma$ has a pole. $\mathcal H_\sigma$
      therefore singles out the launch mode only while $|\mathcal H_\sigma|$ on the
      ray is small against the separation of the roots, $X Y \sqrt g / |a|$, which
      vanishes at the plasma edge.

    The quotient is evaluated with $\eta = N^2 - 1$ and $m = Y^2 (1 - N_\parallel^2)$,

    $$\mathcal H_\sigma = \frac{a \eta^2 + X \eta\,(2 - 2X - m) + X^2 (1 - X)}
    {a \eta + X \left(1 - X - m/2 + \sigma Y \sqrt{g_0} / 2\right)}.$$

    Near the launch sheet $\eta = O(X)$ and every term of the numerator is of order
    $X^2$: nothing of order one is subtracted to obtain it. Nothing is divided by
    $a$, which vanishes at the upper-hybrid layer. The quotient is $0/0$ only in
    vacuum on the surface, where it is replaced by its limit $\eta$.

    Args:
        refractive_index_perp_squared: $N_\perp^2$ of the ray.
        refractive_index_para_squared: $N_\parallel^2$ of the ray.
        X: (ω_pe / ω)^2, normalized plasma frequency squared.
        Y: ω_ce / ω, normalized cyclotron frequency.
        mode: 'X' or 'O', the mode in which the ray is launched.

    Returns:
        $\mathcal H_\sigma$, zero on the cold dispersion surface and $N^2 - 1$ in
        vacuum.
    """
    sigma = 1.0 if mode == "O" else -1.0
    n_para2 = refractive_index_para_squared
    eta = refractive_index_perp_squared + n_para2 - 1.0
    a = (1 - Y) * (1 + Y) - X  # exact to rounding also close to Y = 1
    m = Y**2 * (1 - n_para2)
    root0 = Y * jnp.sqrt(Y**2 * (1 - n_para2) ** 2 + 4 * n_para2)
    numerator = (a * eta + X * (2 - 2 * X - m)) * eta + X**2 * (1 - X)
    divisor = a * eta + X * (1 - X - m / 2 + sigma * root0 / 2)
    # 0/0 only in vacuum on the surface, where the limit is eta. The two jnp.where
    # keep the derivative of the quotient from turning into NaN there.
    vacuum = divisor == 0
    return jnp.where(vacuum, eta, numerator / jnp.where(vacuum, 1.0, divisor))


def dispersion_tensor_stix(
    refractive_index_perp: ScalarFloat,
    refractive_index_para: ScalarFloat,
    dielectric_tensor: jt.Complex[jax.Array, "3 3"],
) -> jt.Complex[jax.Array, "3 3"]:
    r"""Computes the dispersion tensor in Stix coordinates.

    The dispersion tensor is defined as (Stix convention):

    .. math::
        D_{ij} = \epsilon_{ij} - N^2 \delta_{ij} + N_i N_j

    The refractive index has the following form in Stix coordinates:

    .. math::
        N = (N_\perp, 0, N_\parallel)

    where the magnetic field is aligned with the z-axis.

    Args:
        refractive_index_perp: Refractive index perpendicular to the magnetic field.
        refractive_index_para: Refractive index parallel to the magnetic field.
        dielectric_tensor: Dielectric tensor as a 3x3 complex array.

    Returns:
        Dispersion tensor as a 3x3 complex array.
    """
    n1 = refractive_index_perp
    n3 = refractive_index_para

    # N^2 delta_ij - Ni Nj  (note: we compute dielectric_tensor - nn = eps - N^2 I + NN)
    nn = jnp.array(
        [[n3**2, 0, -n1 * n3], [0, n1**2 + n3**2, 0], [-n1 * n3, 0, n1**2]],
        dtype=jnp.complex128,
    )
    return dielectric_tensor - nn
