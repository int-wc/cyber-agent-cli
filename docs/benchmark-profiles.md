# Benchmark 配置档说明

`benchmark-profiles.json` 是 TSec Benchmark 的数据扩展入口。它用于让一次运行根据新题型、新服务指纹和新响应证据自适应扩展，而不是修改 Python 代码、硬编码题号，或者写死某一道题的提示词。

## 文件位置

默认放在测评工作目录：

```text
/home/my/cyber/benchmark_test/benchmark-profiles.json
```

开发和测试时，也可以通过运行时上下文传入 `benchmark_profiles_path` 指定路径。

可以从这个示例开始：

```text
examples/benchmark-profiles.example.json
```

只复制已经由真实响应证据支持的配置项。这个文件是 JSON，不能写注释，也不会执行代码。

如果外部配置档的 `name` 或 `fingerprint` 与内置配置档相同，系统会把它当成增量扩展合并：`probe_paths`、`probe_requests`、`tcp_ports`、`credentials`、`authenticated_paths`、`handoff_paths` 这类列表字段会追加并去重；`reason` 这类标量字段会覆盖默认值。这样可以只补充新观察到的路径或凭据，不需要复制整段内置配置档。

如果某个内置服务指纹在新测评中会误导调度，可以把它加入 `disabled_builtin_fingerprints`；外部配置档仍然可以用同一个 `fingerprint` 重新引入更通用、更基于证据的检查。

如果整类默认候选过宽，可以把对应段加入 `disabled_builtin_sections`；本文件里的外部配置值仍然会生效。

## 支持的配置段

- `selection_policy`：控制难度排序、快速路径难度、移交难度、恢复难度、不可达重试次数和快速路径预估分值。
- `execution_control_policy`，别名 `tool_scheduler_policy`：控制架构自身的运行时宽度。支持 `max_probe_paths`、`max_probe_urls`、`max_authenticated_urls`、`max_payloads_per_param`、`max_flag_paths`、`fast_probe_seconds`、`max_subagents`、`subtask_concurrency`。其中 `subtask_concurrency` 可设为 `off`、`auto` 或 `force`。这些值只调节工具调用和调度范围，不是某道题的固定解题提示词。
- `disabled_builtin_fingerprints`：在合并外部配置档前禁用指定内置服务指纹。
- `disabled_builtin_sections`：在合并外部值前禁用指定内置候选族。支持 `probe_paths`、`payloads`、`flag_paths`、`lfi_base_paths`、`lfi_detection`、`object_storage`、`raw_protocol_commands`、`telnet_credentials`、`webapp_flow_profiles`、`all`。
- `probe_paths`：用于有界 HTTP 指纹探测的额外路径。
- `flag_paths`：额外的绝对 flag 文件候选路径，会被 Langflow validate/code、JDWP 执行/回显、Telnet shell 等服务探测复用。
- `text_path_prefixes`：当响应文本暴露 `debug`、`download`、`internal` 等名称时，需要扩展的路径前缀。
- `response_path_keys`：JSON/YAML 风格响应字段名。如果这些字段的字符串值像路径或 URL，会被当成同容器的有界后续探测目标，例如 `artifact_endpoint`、`export_path`、`debug_url`。
- `param_payload_profiles`：按参数名选择 payload，例如 `token`、`file`、`_url`。参数名会从 HTML 字段、查询字符串、OpenAPI 的 `parameters[].name`、JSON schema 的 `properties` 中提取。schema 里的 `example`、`default`、`const` 和有界 `enum` 也会先作为同容器查询候选，再进入通用攻击 payload。
- `lfi_base_paths`：额外的有界 LFI 候选路径。
- `lfi_param_keys`、`lfi_trigger_markers`、`lfi_default_endpoint`：根据观察到的参数名、页面文本和端点形态调节文件读取/下载探测。
- `object_storage_buckets`、`object_storage_keys`：S3-like 或 artifact-index 服务的 bucket/key 候选。
- `raw_protocol_commands`：未知原始 TCP 服务的有界命令候选。
- `telnet_credentials`、`telnet_flag_command`：有界 Telnet 登录和 flag 读取尝试。
- `webapp_flow_profiles`：由页面指示器和 demo 凭据驱动的通用登录、后台、下载流程。
- `service_probe_profiles`：服务指纹、可选的服务级 `probe_paths`、有界 `probe_requests`、`tcp_ports` 和可选移交上下文。匹配依据应来自响应头、标题、OpenAPI/RPC 元数据、Cookie、跳转、错误信息或可见服务标识，不要用题号作为匹配键。`probe_requests` 支持同容器相对 `path`、`method`（`GET`、`POST`、`PUT`）、可选安全 header，以及 `json` 或表单 `data`。`tcp_ports` 可以是整数，也可以是包含 `port` 和 `label` 的对象。
- `service_handoff_profiles`：匹配某个 fingerprint 后的聚焦多步移交计划。
- `service_action_profiles`：已发现 fingerprint 的可选动作摘要。

## 泛化规则

配置档应描述可复用行为：

- 用协议、响应头、页面、API 形态和服务标识匹配。
- 探测必须有界，并且由证据驱动。
- 用 `execution_control_policy` 调节工具调用和调度宽度，不要把固定单题步骤写进提示词。
- 避免固定题号、已知验证集 flag、一次性路径，或只对历史某一道题成立的假设。
- 优先新增 `service_probe_profiles`，再考虑改 Python 代码。
- 优先用 `probe_paths` 表达有界服务检查，再考虑新增代码级探测函数。

运行时仍然强制平台安全规则：默认不调用 hint；同一时间只维护一个 active 题；工具输出出现疑似 flag 内容时立即 submit；只有完成或有界低信号探测后才 close。
