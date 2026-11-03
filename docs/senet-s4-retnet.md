# SENet, S4 and RetNet extension

This additive fragment extends three distinct architecture lines without changing
existing nodes or claims:

- **SENet** adds input-dependent channel recalibration to residual CNNs.
- **S4** adds structured state-space sequence modeling and its parallel kernel view.
- **RetNet** adds multi-scale retention with equivalent parallel and recurrent forms.

The compact runtimes expose the architectural boundaries that matter for study.
They use random weights and fixed, small inputs; their `runtime_note` fields state
which production features and paper-scale configurations are intentionally omitted.
