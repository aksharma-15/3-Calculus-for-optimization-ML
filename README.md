# 3-Calculus-for-optimization-ML
Calculus foundations for machine learning.
---

# Calculus Foundations for Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A first-principles, code-driven implementation of the mathematical optimization mechanics powering modern machine learning and deep learning models. Rather than relying on high-level autodiff engines (like PyTorch's `autograd` or TensorFlow's `GradientTape`), this notebook constructs objective functions, numerical differentiation, and gradient-based optimization loops entirely from scratch using vectorized **NumPy** operations and custom **Matplotlib** visual diagnostics.

---

## Core Concepts & Terminological Reference

To bridge intuitive geometric interpretations with formal machine learning engineering terminology, the notebook implements the following mathematical primitives:

| Notebook Term / Variable | Formal ML / Optimization Terminology | Mathematical Formulation | Engineering Context |
| :--- | :--- | :--- | :--- |
| **`w` ("the knob")** | **Model Parameter / Weight** (<i>w</i> ∈ ℝ) | <i>ŷ</i> = <i>w</i> · <i>x</i> | A learnable scalar weight governing a linear hypothesis function without a bias intercept (<i>b</i> = 0). |
| **`loss(w)`** | **Mean Squared Error (MSE) Objective / Cost Function** (<i>J(w)</i>) | <i>J(w)</i> = (1 / <i>N</i>) Σ (<i>w</i> · <i>x<sub>i</sub></i> − <i>y<sub>i</sub></i>)² | Measures the L2 empirical risk across the training dataset (`distance` vs. `fare`). |
| **`slope(f, x, h)`** | **Numerical Differentiation (Forward Finite Difference)** | <i>f'</i>(<i>x</i>) ≈ [<i>f</i>(<i>x</i> + <i>h</i>) − <i>f</i>(<i>x</i>)] / <i>h</i> | Approximates the first-order derivative using a perturbation step size <i>h</i> = 10⁻⁵. |
| **Tangent line slope** | **Central Finite Difference Approximation** | <i>f'</i>(<i>t</i>) ≈ [<i>f</i>(<i>t</i> + ε) − <i>f</i>(<i>t</i> − ε)] / (2ε) | Used inside `draw_curve` (ε = 10⁻⁵) for higher-order <i>O</i>(ε²) truncation accuracy when rendering tangent vectors. |
| **`bumpy(w)`** | **Non-Convex Quartic Loss Landscape** | <i>f</i>(<i>w</i>) = <i>w</i>⁴ − 8<i>w</i>³ + 18<i>w</i>² − 11<i>w</i> | A 4th-degree polynomial exhibiting multiple local extrema and inflection points, simulating non-convex deep learning loss surfaces. |
| **`learning_rate`** | **Step Size / Hyperparameter** (η or α) | <i>w</i><sub><i>t</i>+1</sub> ← <i>w<sub>t</sub></i> − η · ∇<sub><i>w</i></sub> <i>J</i>(<i>w<sub>t</sub></i>) | Controls the magnitude of parameter updates along the negative gradient direction. |


---

## Notebook Architecture & Execution Flow

1. **Visualization Diagnostic Suite (`_axes`, `draw_curve`, `draw_contour`, `draw_losses`, `draw_fit`)**
   * Encapsulates low-level Matplotlib state manipulation to render loss surfaces, first-order tangent hyperplanes, optimization trajectories (`paths`), 2D contour projections (`np.meshgrid`), and regression fits.
2. **Empirical Risk Formulation (`loss`, `predict`)**
   * Models a linear pricing relationship (`distance` $\rightarrow$ `fare`) and evaluates parameter candidates ($w \in \{5, 15, 16, 25\}$) across a convex parabolic MSE landscape.
3. **First-Order Sensitivity & Differentiation (`fare_rule`, `curve`, `slope`, `derivative`)**
   * Contrasts constant rates of change in linear systems ($\frac{\Delta y}{\Delta x} = 12$) against state-dependent instantaneous gradients on non-linear manifolds ($f(x) = (x - 3)^2$).
4. **Iterative Optimization & Hyperparameter Dynamics (`descend`)**
   * Implements batch Gradient Descent across both convex (`curve`) and non-convex (`bumpy`) objectives.
   * Demonstrates three canonical learning rate regimes:
     * **Under-damped / Slow Convergence ($\eta = 0.01$):** Fails to reach the global minimum $w^* = 3$ within the fixed 30-epoch budget.
     * **Optimal Convergence ($\eta = 0.10$):** Smoothly decays along the gradient to $w \approx 3.004$ ($J(w) \approx 1 \times 10^{-5}$).
     * **Divergent Oscillation ($\eta = 1.05$):** Overshoots the basin of attraction, causing parameter explosion (`runaway path`) beyond the plotted domain.

---

## Object-Oriented Programming (OOP) & Software Design Analysis

Even though this notebook is structured procedurally with standalone functions for pedagogical clarity, it relies heavily on **Object-Oriented Programming (OOP)** principles under the hood via Matplotlib's artist hierarchy and NumPy's `ndarray` protocol:

* **Encapsulation & State Management (`matplotlib.axes.Axes`):**
  * Instead of relying on Matplotlib's implicit, stateful global procedural interface (`plt.title()`, `plt.xlabel()`, `plt.gca()`), the helper `_axes()` instantiates explicit `Figure` and `Axes` objects via `fig, ax = plt.subplots(...)`.
  * All styling mutations (`ax.set_title`, `ax.grid`, `ax.spines["top"].set_visible(False)`) are invoked as methods directly on the encapsulated `ax` instance, preventing state leakage across sequential plots.
* **Composition & Delegation:**
  * High-level plotting functions (`draw_curve`, `draw_contour`, `draw_losses`, `draw_fit`) delegate axis initialization to the internal factory function `_axes()` (prefixed with a leading underscore `_` to signal private/internal module scope).
* **Polymorphism & Duck Typing (Higher-Order Callables):**
  * Functions like `draw_curve(f, ...)` and `slope(f, x)` accept any callable object `f` (whether a standard Python function, a `lambda`, or a custom class implementing `__call__`) that maps a scalar float to a scalar float, decoupling the optimization/visualization logic from specific mathematical definitions.
* **Operator Overloading (`numpy.ndarray`):**
  * Expressions such as `predicted = w * distance` and `np.mean((predicted - fare) ** 2)` leverage NumPy's C-backed operator overloading (`__mul__`, `__sub__`, `__pow__`) to execute element-wise SIMD vectorization without explicit Python `for` loops.

---

## Notable Code Comments & Engineering Annotations

Several inline comments within the notebook highlight deliberate engineering and numerical computing decisions worth noting when extending or debugging this codebase:

1. **`# === PICTURE HELPERS — run this cell, then ignore it ===============`**
   * *Purpose:* Establishes a clear separation of concerns between **visualization boilerplate** and **core mathematical logic**, allowing practitioners to treat the plotting functions as a black-box diagnostic API.
2. **`right = t > lo + 0.66 * (hi - lo) # keep the label inside the picture`**
   * *Purpose:* Implements dynamic bounding-box alignment (`ha="right"` vs. `ha="left"`) so tangent slope annotations do not clip against the right viewport boundary when evaluating points in the upper third of the domain.
3. **`# keep the window on the curve; a runaway path simply walks off the picture`**
   * *Purpose:* Locks the viewport bounds (`ax.set_xlim`, `ax.set_ylim`) strictly to the objective function's range rather than auto-scaling to the optimizer's trajectory. When a high learning rate ($\eta = 1.05$) causes gradient explosion, the trajectory visibly exits the frame rather than compressing the loss curve into an unreadable flat line.
4. **`w = start # 2, 0, 1, 6`**
   * *Purpose:* Documents candidate initialization states (`w_0`) for testing sensitivity to initial conditions—particularly critical when transitioning from convex quadratics to multi-modal non-convex polynomials like `bumpy(w)`.

---

## Quickstart & Local Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with NumPy and Matplotlib:

```bash
pip install numpy matplotlib jupyter
