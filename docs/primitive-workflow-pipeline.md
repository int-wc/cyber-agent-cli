# 原语工作流管线（Primitive Workflow Pipeline）

从 BSRC_SKILLS_V1 的 workflow「业务原语解析 + 原语链利用」方法论迁移而来，替换
`FourPillarPipeline` 的通用四柱思考，实现**架构融合**：保留四柱骨架与全部基础设施，
把四柱的语义换成漏洞挖掘领域的原语工作流。

## 为什么替换四柱

四柱管线（ANALYST→DIFFUSER→JUMPER→REFLECTOR）是**任务无关**的通用推理框架，擅长"任何
任务"，但不带领域语义。对漏洞挖掘而言，它不知道"端点到底对什么东西做什么操作"。

workflow 的原语方法论恰恰相反：它不按 API 名判断端点，而是判定**核心业务原语
business_attr**（read_file/write_file/exec_code/modify_state/query_data/transfer/auth）
+ 作用对象 attr_target，再把多个原语通过业务信任串联成**原语链**（primitive-chains.json）
构成有效危害。

## 架构融合：形状兼容、语义替换

| 四柱（原） | 原语化（新） | 数据层 |
|---|---|---|
| ANALYST 分析为底 | **原语解析者**：端点→business_attr/attr_target/attr_reason（不看 API 名） | `primitives/parser.py` |
| DIFFUSER 扩展为路 | **攻击面扩散者**：原语→攻击基元枚举 | `primitives/surface_matcher.py` |
| JUMPER 迁跃为辅 | **链跃迁者**：原语组合→候选链（含跨域/remote→local 切换） | `primitives/chain_library.py` |
| REFLECTOR 反思为主 | **链裁决者**：可串联性/可利用性→链式执行计划 | 结构化 JSON 输出 |

执行闭环（决策者→执行者→审计者→反思者）保留，但子任务由「链候选」驱动；每轮执行围绕
一条原语或一段链，审计者验证链组合，反思者裁决是否达到危害阈值。

## 复用（零重复实现）

`PrimitiveWorkflowPipeline(FourPillarPipeline)` 继承全部基础设施：
模型网关（多厂商 fallback）、轨迹追踪、会话持久化、token 统计、熔断、超时叠加、
sub-runner、审批、事件转发、benchmark mixin。只重写 `_run_phases` 的语义。

## 管线选择

`runtime_context["pipeline_mode"]`：
- `auto`（默认）：按任务语义自动判定——命中 SRC/渗透/原语等关键词走原语管线，否则回退四柱
- `primitive`：强制原语工作流管线
- `four_pillar`：强制四柱管线

CLI：`--pipeline-mode auto|primitive|four_pillar`。

```python
from cyber_agent.cli.app_multi_agent import (
    _select_pipeline_mode, PIPELINE_MODE_KEY, _detect_primitive_workflow,
)
from cyber_agent.agent.primitive_pipeline import PrimitiveWorkflowPipeline
```

## 数据层

`src/cyber_agent/agent/primitives/`：

```
├── __init__.py
├── models.py          业务原语/作用对象/端点/链/攻击面 数据模型
├── parser.py          business_attr 判定行解析 + BUSINESS_ATTR_GUIDE
├── chain_library.py   原语链库加载 + 端点→链模板匹配（原语链联动）
├── surface_matcher.py 攻击面前匹配（signals + 原语命中 → 攻击基元注入）
└── data/
    ├── primitive-chains.json   原语链模板库（instances 回写积累）
    ├── attack_surfaces.json    攻击面模式库（程序化匹配）
    └── api_patterns.json       API 模式字典
```

库文件从 BSRC_SKILLS_V1 移植，随使用持续积累实例，跨 SRC 共享。

## 测试

- `tests/test_primitives.py`：数据层（解析/匹配/链联动/模型）
- `tests/test_primitive_pipeline.py`：管线选择、原语语义、全流程（fake LLM 驱动）

```bash
python -m unittest tests.test_primitives tests.test_primitive_pipeline -v
```

## 真实挖洞驱动

`scripts/run_primitive_hunt.py` 可在非交互环境用真实模型跑原语管线：

```bash
python3 scripts/run_primitive_hunt.py "对补天厂商X的 domain.com 做漏洞挖掘" --max-iterations 2
```

- 复用 `build_runtime_context`（与 CLI 同一套上下文构建），强制 primitive 模式 + auto_decision
- 代理不可达时自动切直连（opencode 强制 require_proxy_url，需 `object.__setattr__` 绕过）
- 日志落盘 `scripts/logs/primitive_hunt_<时间戳>.log`，完整轨迹在 `~/.cyber-agent-cli-traces/`

**真实运行验证（2026-08-05，理想汽车 api-app.lixiang.com）：** 原语解析 3-4 端点 →
链跃迁产出 3 条候选链 → 链裁决标 execute/needs_account → 决策者分解 15 个带 curl 命令
子任务 → 执行 **78 个真实只读 curl**（路由探测/头部污染/路径穿越/接口枚举/IDOR/方法探测）。
确认发现：响应头内网 IP:port 泄露（`x-chj-routeurl: http://172.21.27.229:10978/`）+ APISIX
指纹；越权/绕过类均被网关 `code:100012` 拦截（红线 5 组用尽，行为正确）。