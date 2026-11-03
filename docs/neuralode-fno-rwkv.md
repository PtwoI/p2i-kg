# Neural ODE, FNO and RWKV extension

This additive fragment adds three distinct architecture lines without changing
existing nodes or claims:

- **Neural ODE** replaces a discrete residual stack with a learned vector field
  evaluated by a differentiable ODE solver.
- **FNO** learns function-to-function maps through global Fourier multipliers.
- **RWKV** combines a recurrent weighted key-value state with parallelizable
  language-model training.

Each compact runtime exposes the defining module boundaries with random weights.
The `runtime_note` fields distinguish these educational examples from adaptive
adjoint solvers, production neural-operator stacks, and official RWKV kernels.
