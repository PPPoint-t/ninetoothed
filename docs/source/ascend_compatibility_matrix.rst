Ascend 旧实现兼容矩阵
=======================

本文档记录旧分支 ``/root/ninetoothed-dev-ascend`` 中 Ascend 专用行为在当前
SSA 架构中的归属和迁移状态。它是实现清单，不是兼容承诺：只有状态为“已替换”的
行为已具备对应实现或明确错误路径；“延期”行为在后续阶段实现前不得通过隐式回退启用。

决策规则
--------

旧版 ``Ascendifier`` 在生成后的 Triton Python AST 上按名称、语句顺序和装饰器形状改写。
新版只在下列层次实现确定职责：

* SSA pass：合法 schedule、静态 dtype/layout/pattern 限制和候选元数据；
* Ascend emitter：源码 import、Triton/CANN intrinsic 拼写、load/store/mask/cast 等 operation
  级渲染；
* materializer/runtime：工具链导入、artifact 缓存、动态 grid/core、stream 和 ABI 验证；
* LaunchPlan/tuning：运行时 shape、candidate 过滤、计时和调优缓存。

不得恢复旧版的 AST 复制、CUDA/NPU 双源码或按设备探测选择 symbol。每个 Artifact 只服务一个
显式 ``backend="ascend"``。

状态说明
--------

* **已迁移**：旧行为可由同层、同语义的新版实现直接承接。本阶段没有此类项目。
* **已替换**：新版已有等价的结构化 metadata 或故意的显式拒绝，不保留旧版隐式行为。
* **延期**：已确定目标归属，但尚未实现，调用时必须拒绝。
* **不需要**：旧机制与新版单后端 Artifact 合约冲突，永久删除。

兼容矩阵
----------

.. list-table::
   :header-rows: 1
   :widths: 22 15 20 18 25

   * - 旧行为和证据
     - 状态
     - 新归属
     - 当前处理
     - 后续验收条件
   * - ``triton.language.extra.libdevice`` 改为 ``triton.language.extra.cann``
       （旧 ``ascendifier.py`` 第 498--506 行）
     - 已迁移（限 FP16/BF16/FP32 elementwise）
     - Ascend emitter 的 module rendering
     - ``AscendTarget`` 固定生成 ``from triton.language.extra.cann import libdevice``，不复用 CUDA
       import。
     - 阶段 5 已有 stable golden source；需在阶段 6 的 materializer/NPU 集成测试中完成真实编译和
       intrinsic 数值验证。
   * - ``tl.float64`` 静默改成 ``tl.float32`` （第 490--496 行）
     - 已替换
     - Ascend SSA validation 与 emitter dtype policy
     - 当前接受 FP16、BF16 和 FP32 elementwise；FP64 和整数会在 source emission 前明确失败。
     - FP16/BF16 已以数值容差和 910B3 硬件测试单独启用；不得恢复 FP64 到 FP32 的静默降级。
   * - ``tl.load(..., other=None)`` 改为 ``other=0.0`` （第 464--475 行）
     - 已替换（限 FP16/BF16/FP32 elementwise）
     - ``EmitterTarget.load``
     - ``AscendTarget`` 复用 operation 级 ``EmitterTarget.load``，对本阶段允许的低精度与 FP32 tail
       mask 稳定发射 ``other=0.0``；没有 AST 改写。
     - Ascend 910B3 / CANN 9.0.0 已以 257 元素 tail add 验证 ``other=0.0`` 的 mask 路径，以及
       NaN、``+inf``、``-inf`` 和 ``+inf + -inf -> NaN`` 的 FP32 数值结果。FP16/BF16 tail 的 add 和
       0-D output 已在 910B3 验收；其它 operation 的 NaN 语义仍需独立验收。
   * - 三参数 ``tl.clamp`` 改写为 ``minimum(maximum(...))`` （第 477--488 行）
     - 延期
     - Ascend emitter scalar-call rendering，必要时辅以 opcode validation
     - 当前未生成 clamp source；不能基于原始函数调用形状做字符串或 AST 替换。
     - 阶段 5 针对 SSA clamp/minimum/maximum operation 给出直接语义或明确拒绝，并覆盖 dtype、NaN
       和上下界广播。
   * - 只匹配两项 ``BLOCK_SIZE``、值恰为 ``32/64/128`` 的 autotune decorator，再把方阵改为
       三个非方阵 config（第 350--405 行）
     - 延期
     - ``AscendOptimizeSchedule`` 与 LaunchPlan/tuning
     - 当前只提供一个 ``fp16-bf16-fp32-elementwise-256`` candidate；不改写 decorator，也不生成
       ``num_warps``/``num_stages``。
     - 在阶段 7 依据真实 SSA layout、dtype、tile 和 core/grid 限制产生候选；需要 NPU benchmark
       与全部 candidate 合法性验证。
   * - autotune key 仅保留带 ``size`` 的名称，按参数名优先级排序，再按
       ``valid_axis_names`` 数量截断（第 421--460 行）
     - 延期
     - LaunchPlan/tuning
     - 阶段 0 已确认当前 runtime 有六个合法 axis 名；阶段 3 不读取参数名称或进行 key 截断。
     - 阶段 7 从 LaunchABI/LaunchPlan 的真实动态 shape 构造 tuning key，并把 SoC、CANN、layout、
       dtype 和 schedule 纳入缓存键。
   * - 通过变量名 ``qk``、前序 ``tl.dot``、``tl.where(..., -inf)``、循环语句顺序和随后的
       ``exp2`` 识别 SDPA tail-key boundary，并插入 bias/mask AST（第 79--348 行）
     - 已替换
     - Ascend SSA validation；未来可选的独立 SSA pass
     - 阶段 3 将 exp-reduction-dot region、blocked linalg 和 layout transfer 明确拒绝，未声称
       支持 SDPA 或 attention。
     - 只有能以明确 SSA ``linalg.dot``、select/mask、exp、reduction 结构定义模式时才实现独立 pass；
       需要正反例、尾部长度、``-inf``、NaN 和数值稳定性 NPU 测试。
   * - ``Ascendifier`` 在普通 CUDA/Triton AST 的深拷贝上运行，生成两个函数
       （旧 ``generation.py`` 第 132--153 行）
     - 不需要
     - 删除
     - 新版 frontend 一次生成 target-neutral SSA；Ascend 由单独 backend 从同一 SSA program 降低。
     - 永不恢复双 AST 或复制应用逻辑；每个 Artifact 的 ``backend`` 必须唯一。
   * - 在生成的 Python 模块中探测 ``torch.npu.is_available()``，运行时选择 CUDA 或 ``_npu``
       函数（旧 ``generation.py`` 第 146--153 行）
     - 不需要
     - 删除
     - target 只能由显式 ``backend="ascend"`` 选择；阶段 2 registry 不接受 ``npu``/``cann``
       作为 backend alias。
     - 测试 Ascend 请求缺少 emitter/materializer 时明确报错，且 CUDA/Triton 请求不受 NPU 环境影响。
   * - 按 ``coreDim=`` / ``UINT16_MAX`` 错误文本触发候选回退，固定上限 65535
       （旧 ``AscendAOTBackend.py``）
     - 已替换
     - schedule metadata，后续 runtime validator
     - 阶段 3 以已校验的 ``max_core_dim`` 和 ``core_dim_limit`` metadata 表达 65535 上限；不解析
       编译器错误文本或改变计算域。
     - 阶段 6 使用实际 LaunchPlan/grid 和动态 shape 做运行时拒绝；阶段 7 仅 benchmark 已验证的候选。
   * - 私有 JSON manifest、UUID Python module 名和 ``sys.modules`` 管理
       （旧 ``AscendAOTBackend.py``）
     - 不需要
     - ``BuiltArtifact`` 与 ``AscendMaterializer``
     - 阶段 6 采用 source-only Ascend Triton Python artifact：共享 content-addressed cache、原子
       source 发布和通用 ``BuiltArtifact`` manifest；loader 用隔离 module namespace 执行 source，
       不修改 ``sys.modules``。
     - 以持久进程/NPU 集成测试验证 AOT/JIT reload 的首个真实 launch；不得恢复私有 JSON、UUID
       module 名或全局 module 注册。

阶段 4 结论
-----------

旧 ``Ascendifier`` 没有可直接复制的函数，因此本矩阵没有“已迁移”项目。它所解决的问题应被拆成 emitter 方言、SSA
validation、schedule/LaunchPlan 和 materializer/runtime 四类职责。当前已替换的内容只包括：
FP64 的显式拒绝、复杂 SDPA/linalg/reduction/layout pattern 的显式拒绝、65535 的静态 schedule
metadata，以及删除双源码运行时 guard。其余行为均未实现，不得以“兼容旧分支”为由提前启用。

阶段 5 已完成 CANN import 和 FP16/BF16/FP32 masked load/store 的源码发射，并以白名单限制 operation；没有
恢复 AST 重写。``tl.clamp``、扩展 scalar call、mixed dtype，以及 add 以外 operation 的 NaN 语义仍延期。阶段 6 和阶段 7
分别处理 runtime/materialization 与 autotuning。阶段 6 已实现 source-only materializer；真实 NPU
编译/launch 已在 Ascend 910B3 / CANN 9.0.0 的 capability-gated 测试中完成首版验收。

后续增量：Ascend emitter 对一维 singleton 广播采用 SSA 坐标级发射，输入长度为 ``1`` 时固定读取
offset ``0``，而不是沿输出 index 访问。materializer 以 ABI 的 output 为 launch domain，只接受连续
FP16/BF16/FP32 的 ``N + 1 -> N``；二维和更一般的广播、scalar ABI、view/aliasing 仍延期。
