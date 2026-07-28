# Physical model

The pivot is fixed in the world frame. A scalar-first Hamilton quaternion
{math}`q` maps body vectors to world vectors. Angular velocity {math}`\omega`
is expressed along the principal body axes:

```{math}
I\dot{\omega} =
\tau_g - B\omega - \omega \times (I\omega),
\qquad
\dot q = \frac{1}{2} q \otimes [0,\omega].
```

Gravity must depend on orientation:

```{math}
g_b = R(q)^T [0,0,-g]^T,
\qquad
\tau_g = r_\mathrm{cm} \times m g_b.
```

The symmetry or reference axis is configurable. Its world orientation gives
the tilt angle. Wobble is persistent rolling tilt amplitude; FTLE-based chaos
is evaluated independently.

Without damping, total mechanical energy is conserved. With no external
torque, world angular momentum is conserved. Under gravity about a fixed
pivot, its vertical component is conserved. These identities are executable
validation tests.

