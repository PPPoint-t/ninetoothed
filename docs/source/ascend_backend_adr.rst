Ascend Backend Architecture Decision
==========================================

Status
------

Accepted. The initial source-emission and source-only materialization
milestones are implemented. On Ascend 910B3 with CANN 9.0.0, the capability-
gated integration test verifies contiguous FP32 elementwise JIT launch,
tail-mask numerical output including NaN and signed-infinity add semantics,
source reload, dynamic grid, a non-default NPU stream, and one-dimensional
singleton broadcast (``N + 1 -> N``). Other devices, operations, dtypes,
layouts, and broadcast forms remain unverified.

Context
-------

NineToothed compiles application Python AST into target-neutral SSA before a
selected backend emits an artifact and a materializer makes that artifact
callable. The historical Ascend implementation predates this architecture. It
rewrites a copied Triton AST, embeds both CUDA and NPU source in one Python
module, and selects an NPU entry point at runtime.

The current environment provides a working Ascend 910B3 device, torch_npu,
triton.backends.ascend, and triton.language.extra.cann. It can execute NPU
tensor operations and import the Triton Ascend runtime APIs.

Decision
--------

The initial backend will use these boundaries:

* backend="ascend" is an explicit target. It is not inferred from
  torch.npu.is_available() and does not change the default Triton backend.
* The frontend remains target-neutral. It continues to lower applications to
  SSA and does not import Ascend APIs or apply Ascend AST rewrites.
* AscendBackend lowers SSA through an Ascend-specific pass bundle and emits
  one Triton Ascend Python source module per artifact. The current emitter is
  restricted to the verified FP32 elementwise operation set and emits CANN's
  ``triton.language.extra.cann`` import path.
* AscendMaterializer owns source caching, import/loading, NPU
  stream selection, runtime argument binding, and error reporting through the
  existing Artifact, BuiltArtifact, LaunchABI, and LaunchPlan contracts. Its
  initial AOT model is a reloadable Triton Python source artifact, not a CANN
  shared object.
* JIT is the first executable milestone. It must use the selected Ascend
  artifact only, without a CUDA source fallback or an NPU symbol convention.
* AOT must use the common BuiltArtifact manifest and reload contract. The exact
  Ascend binary or module packaging format remains deferred until it is
  verified against the installed Triton Ascend and CANN toolchain.

Consequences
------------

Target syntax differences such as imports, scalar calls, load/store, casts,
masks, and intrinsic spelling belong to the Ascend emitter. Scheduling,
axis-count limits, tile restrictions, and grid limits belong to Ascend SSA
passes and runtime launch validation. The historical SDPA AST rewrite is not
ported initially; unsupported patterns must fail explicitly until an
SSA-semantic replacement is designed and tested.

The initial supported scope is contiguous FP32 elementwise kernels, including
only one-dimensional singleton input broadcast (``N + 1 -> N``), load/store,
and tail masks. The output defines the launch domain and core limit; every
input must be a contiguous one-dimensional tensor of length ``N`` or ``1``.
Other broadcast forms remain rejected. Ascend build uses only the
``fp32-elementwise-256`` SSA schedule and records no runtime tuning candidates;
``num_warps``, ``num_stages``, and multiple build candidates for one runtime
configuration are rejected. Reduction, lower-precision types, non-contiguous
views, dot, autotuning, attention, and jagged layouts are enabled only after
their individual runtime and numerical tests exist. Ascend autotuning must use
validated dynamic grid/core checks and NPU event (or official synchronization)
benchmarking before it is enabled.

Compatibility And Errors
------------------------

``backend=None`` remains Triton even when an NPU is available. The optional
``NINETOOTHED_BACKEND=ascend`` environment selection uses the same strict
target normalization as an explicit backend argument; ``npu`` and ``cann``
are not aliases. An Ascend request never falls back to CUDA, generic Triton,
or CPU.

Ascend-only dependencies remain lazy: importing NineToothed and using another
backend does not import torch_npu or CANN APIs. Loading an Ascend source module
reports a missing Triton Ascend/CANN dependency with its package name; launch
reports a missing torch_npu runtime or unavailable NPU stream with backend
context. Tensor inputs must be on one NPU device, contiguous, FP32, rank and
shape compatible with the ABI, and within their storage span. The only
broadcast contract is one-dimensional ``N + 1 -> N``; output broadcasting,
multi-dimensional broadcast, scalar ABI extensions, and views are not enabled.
Empty tensors return their output without a zero-grid launch. Jagged tensors,
non-contiguous views, additional dtypes, reductions, and matmul remain
explicitly unsupported in this first tier.
