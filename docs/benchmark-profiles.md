# Benchmark Profiles

`benchmark-profiles.json` is the data-only extension point for TSec Benchmark
behavior. It lets a run adapt to new challenge families without adding Python
code or hardcoding challenge IDs.

## Location

Put the file in the benchmark workspace:

```text
/home/my/cyber/benchmark_test/benchmark-profiles.json
```

For development or tests, the pipeline can also receive `benchmark_profiles_path`
in its runtime context.

Start from:

```text
examples/benchmark-profiles.example.json
```

Copy only the entries that match evidence you have actually observed. The file
is JSON, so it cannot contain comments and it does not execute code.

Profiles with the same `name` or `fingerprint` as a built-in profile are merged
as incremental extensions: list-like fields such as `probe_paths`,
`probe_requests`, `tcp_ports`, `credentials`, `authenticated_paths`, and
`handoff_paths` are appended and deduplicated, while scalar fields such as
`reason` override the default. This lets you add observed paths or credentials
without copying the whole built-in profile.

## Supported Sections

- `selection_policy`: difficulty order, fast-path difficulties, handoff
  difficulties, recovery difficulties, unreachable retry count, and estimated
  fast score.
- `probe_paths`: extra paths for bounded HTTP fingerprinting.
- `flag_paths`: extra absolute flag file candidates shared by service probes
  such as Langflow validate/code, JDWP exec/exfiltration, and default Telnet
  shell retrieval.
- `text_path_prefixes`: path prefixes that should be expanded when response text
  exposes names such as `debug`, `download`, or `internal`.
- `response_path_keys`: JSON/YAML-like response field names whose string values
  should be treated as bounded same-container follow-up paths or URLs, such as
  `artifact_endpoint`, `export_path`, or `debug_url`.
- `param_payload_profiles`: payloads selected by parameter name, such as
  `token`, `file`, or `_url`.
  Parameter names are collected from HTML fields, query strings, OpenAPI
  `parameters[].name`, and JSON schema `properties`.
  Schema `example`, `default`, `const`, and bounded `enum` values are also used
  as same-container query candidates before generic attack payloads.
- `lfi_base_paths`: extra bounded LFI candidate paths.
- `object_storage_buckets` and `object_storage_keys`: bucket/key candidates for
  S3-like or artifact-index services.
- `raw_protocol_commands`: bounded commands for unknown raw TCP services.
- `telnet_credentials` and `telnet_flag_command`: bounded telnet login and flag
  retrieval attempts.
- `webapp_flow_profiles`: generic login/dashboard/download flows driven by page
  indicators and demo credentials.
- `service_probe_profiles`: service fingerprints, optional service-level
  `probe_paths`, bounded `probe_requests`, `tcp_ports`, and optional handoff
  context. Use response headers, titles, OpenAPI/RPC metadata, cookies,
  redirects, error messages, or visible service banners. Do not use challenge
  IDs as match keys.
  `probe_requests` support same-container relative `path`, `method` (`GET`,
  `POST`, or `PUT`), optional safe headers, and either `json` or form `data`.
  `tcp_ports` entries may be integers or objects with `port` and `label`.
- `service_handoff_profiles`: focused multi-step handoff plans for a matched
  fingerprint.
- `service_action_profiles`: optional action summaries for fingerprints already
  discovered by a probe profile.

## Generalization Rules

Profiles should describe reusable behavior:

- Match on protocol, headers, pages, API shape, and service banners.
- Keep probes bounded and evidence-driven.
- Avoid fixed challenge IDs, known validation-set flags, one-off paths, or
  assumptions that only make sense for a single historical task.
- Prefer adding a new `service_probe_profiles` entry before editing Python.
- Prefer `probe_paths` for bounded service-specific checks before adding an
  in-code probe function.

The runtime still enforces the platform safety rules: no hint by default,
single active challenge sequencing, submit immediately on flag-like output, and
close only after completion or bounded low-signal probing.
