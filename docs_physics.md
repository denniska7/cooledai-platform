# Physics Documentation: Heat Distribution in Data Centers

## 1. Governing Equation

The heat distribution in a data center is governed by the **3D transient heat equation**:

```
∂T/∂t = α∇²T + Q/(ρ·c_p)
```

Where:
- **T(x, y, z, t)**: Temperature field [°C or K]
- **t**: Time [s]
- **α**: Thermal diffusivity [m²/s] = k/(ρ·c_p)
- **∇²**: Laplacian operator = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
- **Q**: Volumetric heat generation rate [W/m³] (from servers)
- **ρ**: Air density [kg/m³]
- **c_p**: Specific heat capacity [J/(kg·K)]
- **k**: Thermal conductivity [W/(m·K)]

### Expanded Form

```
∂T/∂t = α(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²) + Q/(ρ·c_p)
```

## 2. Boundary Conditions

### 2.1 Inlet (Cooling Air Supply)
**Dirichlet Boundary Condition:**
```
T(x_inlet, y, z, t) = T_inlet = 18°C
```

### 2.2 Walls
**No-flux (Neumann) Boundary Condition:**
```
∂T/∂n = 0
```
Where n is the normal direction to the wall (assuming insulated walls).

### 2.3 Outlet
**Natural outlet condition:**
```
∂T/∂x = 0  (at outlet)
```

### 2.4 Floor and Ceiling
Options:
- **Neumann (adiabatic):** ∂T/∂z = 0
- **Dirichlet (fixed temp):** T = T_floor/ceiling

## 3. Initial Conditions

At t = 0:
```
T(x, y, z, 0) = T_initial = 22°C
```

## 4. Heat Source Term (Servers)

### 4.1 Point Source Approximation
For N server racks:
```
Q(x, y, z) = Σ P_i · δ(r - r_i)
```

Where:
- **P_i**: Power dissipation of rack i [W]
- **δ**: Dirac delta function
- **r_i**: Position of rack i

### 4.2 Volumetric Source
For distributed heat generation:
```
Q(x, y, z) = P_rack / V_rack  [W/m³]
```

Where V_rack is the volume of the server rack.

## 5. Dimensionless Form (Optional)

For numerical stability, we can non-dimensionalize:

```
T* = (T - T_ref) / ΔT
x* = x / L
t* = t · α / L²
```

This gives:
```
∂T*/∂t* = ∇*²T* + Q*
```

## 6. Steady-State Approximation

For steady-state (∂T/∂t = 0):
```
∇²T = -Q/(k)
```

This is the **Poisson equation** with heat sources.

## 7. Data Center Specific Considerations

### 7.1 Air Flow
In reality, data centers have complex air flow patterns:
- Cold aisle / Hot aisle configuration
- Forced convection from CRAC units
- Natural convection from hot equipment

For PINN v1.0, we simplify by focusing on heat diffusion. Future versions can incorporate:
```
∂T/∂t + v·∇T = α∇²T + Q/(ρ·c_p)
```
Where **v** is the velocity field (convection term).

### 7.2 Typical Values

| Parameter | Value | Units |
|-----------|-------|-------|
| T_inlet | 18-20 | °C |
| T_ambient | 22-25 | °C |
| P_rack | 3000-10000 | W |
| α (air) | 2.2×10⁻⁵ | m²/s |
| ρ (air) | 1.2 | kg/m³ |
| c_p (air) | 1005 | J/(kg·K) |

## 8. PINN Formulation

### 8.1 Network Input
```
Input: [x, y, z, t] ∈ ℝ⁴
```

### 8.2 Network Output
```
Output: T(x, y, z, t) ∈ ℝ
```

### 8.3 Physics Loss (PDE Residual)

Using automatic differentiation:
```
f = ∂T/∂t - α∇²T - Q/(ρ·c_p)

L_physics = (1/N) Σ |f(x_i, y_i, z_i, t_i)|²
```

### 8.4 Boundary Loss
```
L_boundary = (1/M) Σ |T(x_b) - T_bc|²
```

### 8.5 Initial Condition Loss
```
L_initial = (1/K) Σ |T(x_i, t=0) - T_initial|²
```

### 8.6 Total Loss
```
L_total = λ_1·L_physics + λ_2·L_boundary + λ_3·L_initial
```

Where λ_i are weighting hyperparameters.

## 9. Solution Strategy

1. **Sample collocation points** in the domain (x, y, z, t)
2. **Forward pass** through neural network to get T
3. **Compute derivatives** using automatic differentiation
4. **Calculate physics residual** f = ∂T/∂t - α∇²T - Q/(ρ·c_p)
5. **Evaluate loss** on physics + boundaries + initial conditions
6. **Backpropagate** and update network weights
7. **Repeat** until convergence

## 10. References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

- Cai, S., et al. (2021). Physics-Informed Neural Networks (PINNs) for Heat Transfer Problems. Journal of Heat Transfer.
