"""
title: Deep Research
id: deep_research_switch
author: JiangNanGenius
version: 9.17.0
license: MIT
required_open_webui_version: 0.5.0
description: |
  Chat UI Toggle Router + per-chat settings for Deep Research.

  When enabled:
    - Routes the current selected base model request to the Deep Research pipe model: deepresearch.loop
    - Writes metadata for the pipe to recover base_model even if other filters rewrite metadata
    - Injects a <dr_router>{...}</dr_router> system tag as a robust fallback

  Metadata written:
    - metadata.dr_original_model
    - metadata.deep_research.base_model
    - metadata.deep_research.session_id
    - metadata.deep_research.config
    - metadata.deep_research.ui

  Note in v9.4.0:
    - Paired with deepresearch.loop v12.6.10.0+: the pipe provides an independent “交付物修改模式”（多阶段补丁+自动应用），并在异常时回退到安全修改生成，避免误入全流程。

  Important fixes in v9.2.0:
    - More robust session_id derivation: prefer chat_id; else reuse existing router tag session_id;
      else derive a deterministic fingerprint from the *first user message* (+ optional message id/time)
      to keep the same chat on the same Deep Research session even when OpenWebUI doesn't pass chat_id.

  Recursion protection:
    - If metadata.dr_internal_call == True, this router will NOT route again.
"""

import base64
import hashlib
import json
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


FILTER_ID = "deep_research_switch"


# ----------------------------
# Helpers
# ----------------------------


def _safe_json_loads(s: Any) -> Any:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t or not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def _svg_data_uri() -> str:
    """Light-bulb icon as SVG data URI (white on transparent)."""
    svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>
<path fill='white' d='M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-20C7.93 1 5 3.93 5 7.5c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-3.76c1.81-1.27 3-3.36 3-5.74C19 3.93 16.07 1 12 1zm2.71 10.55-.71.5V16h-4v-3.95l-.71-.5A4.98 4.98 0 0 1 7 7.5C7 5.02 9.02 3 11.5 3S16 5.02 16 7.5c0 1.63-.8 3.12-2.29 4.05z'/>
</svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _merge_metadata(
    body: Dict[str, Any], __metadata__: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    md: Dict[str, Any] = {}
    if isinstance(__metadata__, dict):
        md.update(deepcopy(__metadata__))
    bmd = body.get("metadata")
    if isinstance(bmd, dict):
        md.update(deepcopy(bmd))
    # sometimes metadata is a JSON string
    if isinstance(md.get("metadata"), str):
        obj = _safe_json_loads(md.get("metadata"))
        if isinstance(obj, dict):
            md.update(obj)
    return md


def _truthy(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on", "enable", "enabled"):
            return True
        if s in ("false", "0", "no", "n", "off", "disable", "disabled"):
            return False
    return None


def _as_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(default)
        return int(str(v).strip())
    except Exception:
        return int(default)


def _as_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


def _as_bool(v: Any, default: bool) -> bool:
    b = _truthy(v)
    return bool(default) if b is None else bool(b)


def _read_toggle_state_from_dict(d: Dict[str, Any], filter_id: str) -> Optional[bool]:
    if not isinstance(d, dict):
        return None

    # map shapes
    for key in (
        "filters",
        "filter_states",
        "toggle_filters",
        "toggles",
        "enabled_filters_map",
        "active_filters_map",
    ):
        v = d.get(key)
        if isinstance(v, dict) and filter_id in v:
            b = _truthy(v.get(filter_id))
            if b is not None:
                return b

    # list shapes
    for key in (
        "enabled_filter_ids",
        "enabledFilterIds",
        "active_filter_ids",
        "activeFilterIds",
        "enabled_filters",
        "enabledFilters",
        "active_filters",
        "activeFilters",
        "filters_enabled",
    ):
        v = d.get(key)
        if isinstance(v, list):
            ids = [str(x) for x in v]
            if filter_id in ids:
                return True

    # direct bool
    if filter_id in d:
        b = _truthy(d.get(filter_id))
        if b is not None:
            return b

    return None


def _read_toggle_state(metadata: Dict[str, Any], filter_id: str) -> Optional[bool]:
    # In many OpenWebUI builds, chat UI toggle states appear under chat_state/ui_state/state.
    for key in ("chat_state", "chatState", "ui_state", "uiState", "state"):
        v = metadata.get(key)
        if isinstance(v, str):
            vv = _safe_json_loads(v)
            if isinstance(vv, dict):
                v = vv
        if isinstance(v, dict):
            b = _read_toggle_state_from_dict(v, filter_id)
            if b is not None:
                return b

    b = _read_toggle_state_from_dict(metadata, filter_id)
    if b is not None:
        return b

    return None


def _norm_content(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, bytes):
        try:
            return c.decode("utf-8", errors="ignore")
        except Exception:
            return str(c)
    if isinstance(c, dict):
        if isinstance(c.get("text"), str):
            return c.get("text")
        if isinstance(c.get("content"), str):
            return c.get("content")
        return json.dumps(c, ensure_ascii=False)
    if isinstance(c, list):
        out = []
        for x in c:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                if isinstance(x.get("text"), str):
                    out.append(x.get("text"))
                elif isinstance(x.get("content"), str):
                    out.append(x.get("content"))
                else:
                    out.append(str(x))
            else:
                out.append(str(x))
        return "\n".join([s for s in out if str(s).strip()])
    return str(c)


def _safe_first_user_message(messages: Any) -> Optional[dict]:
    if not isinstance(messages, list):
        return None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return m
    return None


def _safe_first_user_text(messages: Any) -> str:
    m = _safe_first_user_message(messages)
    if not m:
        return ""
    return _norm_content(m.get("content")).strip()


def _extract_msg_stable_id(m: Optional[dict]) -> str:
    if not isinstance(m, dict):
        return ""
    # Common message id/time keys seen across variants
    for k in (
        "id",
        "_id",
        "message_id",
        "messageId",
        "created_at",
        "createdAt",
        "create_time",
        "createTime",
        "ts",
        "timestamp",
    ):
        v = m.get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v).strip()
    return ""


def _parse_router_tag(messages: Any) -> Dict[str, Any]:
    """Find <dr_router>{json}</dr_router> in any system message."""
    if not isinstance(messages, list):
        return {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if "<dr_router>" in c and "</dr_router>" in c:
            try:
                start = c.find("<dr_router>") + len("<dr_router>")
                end = c.find("</dr_router>", start)
                if end > start:
                    obj = _safe_json_loads(c[start:end])
                    if isinstance(obj, dict):
                        return obj
            except Exception:
                continue
    return {}


def _ensure_router_tag(messages: Any, payload: Dict[str, Any]) -> Any:
    """Ensure there is a router tag system message.

    - If an existing tag already contains session_id, keep it (avoid breaking continuity).
    - If a tag exists but is malformed/missing session_id, prepend our own tag so the pipe
      will read ours first.
    """
    if not isinstance(messages, list):
        messages = []

    # Check existing tag
    existing = _parse_router_tag(messages)
    if (
        isinstance(existing, dict)
        and isinstance(existing.get("session_id"), str)
        and existing.get("session_id").strip()
    ):
        return messages

    tag = "<dr_router>" + json.dumps(payload, ensure_ascii=False) + "</dr_router>"
    return [{"role": "system", "content": tag}] + messages


def _read_chat_id_from_obj(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        # sometimes it's JSON
        j = _safe_json_loads(obj)
        if isinstance(j, dict):
            return _read_chat_id_from_obj(j)
        return obj.strip()
    if not isinstance(obj, dict):
        return ""

    # direct keys
    for k in (
        "chat_id",
        "chatId",
        "conversation_id",
        "conversationId",
        "conversationID",
        "id",
        "chat",
        "conversation",
    ):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            vv = _read_chat_id_from_obj(v)
            if vv:
                return vv

    # nested common containers
    for k in ("chat_state", "chatState", "state", "chat", "conversation"):
        v = obj.get(k)
        if isinstance(v, str):
            vv = _safe_json_loads(v)
            if isinstance(vv, dict):
                v = vv
        if isinstance(v, dict):
            vv2 = _read_chat_id_from_obj(v)
            if vv2:
                return vv2

    return ""


def _derive_session_id(
    *,
    body: Dict[str, Any],
    md: Dict[str, Any],
    router_tag: Dict[str, Any],
    user: Optional[dict],
) -> str:
    """Derive a stable session_id.

    Priority:
      1) metadata.deep_research.session_id
      2) router_tag.session_id
      3) chat_id (from body/metadata/router_tag) -> chat:<id>
      4) uuid fallback (generated once; persisted via router_tag/messages)
    """
    dr_prev = (
        md.get("deep_research") if isinstance(md.get("deep_research"), dict) else {}
    )

    sid = ""
    if isinstance(dr_prev, dict):
        sid = str(dr_prev.get("session_id") or "").strip()
    if not sid and isinstance(router_tag, dict):
        sid = str(router_tag.get("session_id") or "").strip()
    if sid:
        return sid

    # chat_id
    chat_id = ""
    # body has highest chance in some builds
    chat_id = _read_chat_id_from_obj(body)
    if not chat_id:
        chat_id = _read_chat_id_from_obj(md)
    if not chat_id and isinstance(dr_prev, dict):
        chat_id = _read_chat_id_from_obj(dr_prev)
    if not chat_id and isinstance(router_tag, dict):
        chat_id = _read_chat_id_from_obj(router_tag)

    if chat_id:
        return f"chat:{chat_id}"

    # uuid fallback (avoid cross-chat collisions when chat_id is unavailable)
    return f"uuid:{uuid.uuid4()}"


# ----------------------------
# Filter
# ----------------------------


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=9999, description="Run order priority. Larger runs later."
        )
        pipe_model_id: str = Field(
            default="deepresearch.loop", description="Route to this pipe model id."
        )
        internal_skip_flag_key: str = Field(
            default="dr_internal_call",
            description="Skip routing if this metadata flag is True.",
        )
        assume_enabled_if_unknown: bool = Field(
            default=True, description="If toggle state cannot be read, assume enabled."
        )

    class UserValves(BaseModel):
        # UI display
        status_mode: str = Field(
            default="simple", description="状态显示：simple=简洁；debug=详细"
        )
        show_timing: bool = Field(default=True, description="状态栏显示耗时统计")
        append_timing_to_final: bool = Field(
            default=False, description="把耗时统计附加到最终正文末尾"
        )
        show_tool_detail_in_debug: bool = Field(
            default=True, description="debug 模式显示工具参数/预览"
        )
        show_alignment: bool = Field(
            default=True, description="状态栏显示“需求对齐/默认假设/可调整项”"
        )

        # Loop limits
        max_rounds: int = Field(default=24, description="最大循环轮次（可填很大）")
        min_rounds: int = Field(default=1, description="最少轮次（>=1）")
        max_tool_turns_per_round: int = Field(
            default=12, description="每轮 tool-loop 最大跳数（>=1）"
        )
        max_tool_calls_per_turn: int = Field(
            default=12, description="单次最多执行 tool_calls 数（>=0；0=禁用工具执行）"
        )
        max_output_continue_turns: int = Field(
            default=60, description="finish_reason=length 时自动续写次数（>=0）"
        )

        # Output chunking
        chunk_max_chars: int = Field(
            default=40000, description="单段输出最大字符（>=1000）"
        )
        chunk_policy: str = Field(
            default="auto", description="输出分段：auto=不分段；manual=分段需回复“继续”"
        )

        # Clarify behavior
        enable_clarify: bool = Field(
            default=True, description="允许在必要时暂停问 1 个关键问题"
        )
        clarify_max_repeat_same_q: int = Field(
            default=1, description="同一个澄清问题最多重复次数（>=0）"
        )
        enable_assumption_mode: bool = Field(
            default=True, description="用户说“随便/都可以”时默认假设推进"
        )

        enable_artifact_edit_mode: bool = Field(
            default=True,
            description="追问修改时使用‘交付物修改模式’（分阶段补丁并自动应用，不整份重写）",
        )
        confirm_policy: str = Field(
            default="pause", description="确认策略：pause/auto/always/never"
        )

        # Dynamic plan / judge
        enable_dynamic_plan: bool = Field(
            default=True, description="允许评审器动态更新计划"
        )
        enable_judge_rewrite: bool = Field(
            default=True, description="允许评审器改写下一轮 user_text"
        )

        # Final compile
        enable_final_compile: bool = Field(
            default=True, description="启用最终汇总编译（避免只输出最后一步）"
        )
        prefer_fast_followup: bool = Field(
            default=True,
            description="跟进/续写/修复默认走快速路径（打补丁/小规模重生成），避免重新规划导致跑偏",
        )
        prefer_patch_over_regenerate: bool = Field(
            default=True, description="追问修改代码时优先补丁，不整份重写"
        )
        followup_force_full_context: bool = Field(
            default=True,
            description="追问修改时：对代码类交付物默认注入完整上下文（减少丢功能）",
        )

        # --- Artifact Detach (超大交付物分离：避免超长代码/文本被模型重写时丢内容) ---
        artifact_detach_enabled: bool = Field(
            default=True,
            description="超大交付物分离：开启后对超长交付物采用‘分离+补丁’（更稳，不整份重写）",
        )
        artifact_detach_min_chars: int = Field(
            default=60000, description="超大交付物分离：固定阈值（字符数）"
        )
        artifact_detach_threshold_mode: str = Field(
            default="auto",
            description="分离阈值策略：fixed=固定阈值；auto=启发式+必要时模型；smart=总是用模型判定",
        )
        artifact_detach_smart_model: str = Field(
            default="",
            description="智能分离判定模型（留空=judge_model；再留空=base_model）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        artifact_detach_smart_lower_bound: int = Field(
            default=20000, description="智能分离：长度<=该值不分离"
        )
        artifact_detach_smart_upper_bound: int = Field(
            default=180000, description="智能分离：长度>=该值强制分离"
        )
        artifact_detach_smart_ambiguous_band: int = Field(
            default=15000,
            description="auto模式：临界区带宽（MIN_CHARS±band 才调用模型判定）",
        )
        artifact_detach_smart_snippet_chars: int = Field(
            default=1200, description="智能分离：提供给判定模型的头部节选字符数"
        )
        artifact_detach_smart_max_tokens: int = Field(
            default=220, description="智能分离：判定模型 max_tokens"
        )

        artifact_detach_max_full_inject_chars: int = Field(
            default=200000, description="分离：全文注入上限（超过强制分离）"
        )
        artifact_detach_head_chars: int = Field(
            default=50000, description="分离：注入头部字符数"
        )
        artifact_detach_tail_chars: int = Field(
            default=45000, description="分离：注入尾部字符数"
        )
        artifact_detach_max_snippets: int = Field(
            default=10, description="分离：聚焦片段条数"
        )
        artifact_detach_snippet_window: int = Field(
            default=1200, description="分离：聚焦片段窗口大小"
        )
        artifact_detach_outline_max_items: int = Field(
            default=60, description="分离：大纲条数上限"
        )
        followup_anti_regression: bool = Field(
            default=True, description="追问修改防跑偏：长度/锚点检查"
        )
        followup_anti_regression_min_ratio: float = Field(
            default=0.8, description="防跑偏：输出不得短于基线的比例（0-1）"
        )
        followup_anti_regression_min_anchor_hits: int = Field(
            default=3, description="防跑偏：至少命中锚点数"
        )

        skip_final_compile_for_code: bool = Field(
            default=True,
            description="代码类任务若已生成完整代码，则跳过最终二次汇总重写，避免丢功能/变简单",
        )
        compile_model: str = Field(
            default="",
            description="汇总编译模型（留空=base_model）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        compile_max_output_continue_turns: int = Field(
            default=30, description="汇总编译自动续写次数"
        )
        final_min_chars: int = Field(
            default=8000, description="最终长文最少字符（代码类会忽略）"
        )
        compile_step_output_max_chars: int = Field(
            default=20000, description="每步草稿注入汇总时的最大字符"
        )

        # Optional: long-step heartbeat interval (pipe may ignore if not implemented)
        status_heartbeat_interval_sec: int = Field(
            default=120, description="长步骤心跳状态间隔（秒）"
        )

        # 续写/分段续写进度摘要（展示到前端状态栏）
        progress_summary_enabled: bool = Field(
            default=True,
            description="自动续写/分段续写时，输出每段摘要与进度到状态栏（会额外调用一次模型；不在乎token可开启）",
        )
        progress_summary_model: str = Field(
            default="",
            description="续写摘要模型（留空=judge_model；judge_model也为空则用base_model）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        progress_summary_every_n_continues: int = Field(
            default=1, description="每 N 次续写生成一次摘要（>=1）"
        )
        progress_summary_prompt_max_chars: int = Field(
            default=2500, description="传给摘要模型的新增文本最大字符（>=400）"
        )
        progress_summary_max_tokens: int = Field(
            default=220, description="摘要模型 max_tokens（若后端支持；>=64）"
        )
        continue_context_max_chars: int = Field(
            default=0,
            description="自动续写时传给主模型的‘已输出内容’最大字符（0=不裁剪；建议 20000-40000 之间可明显提速）",
        )
        ultra_long_patch_continue_enabled: bool = Field(
            default=True,
            description="超长输出：续写改用补丁方式（更稳，不易丢内容）",
        )
        ultra_long_patch_continue_min_chars: int = Field(
            default=30000,
            description="超过该字符数后允许切换补丁续写",
        )
        ultra_long_patch_continue_switch_after_n: int = Field(
            default=2,
            description="续写超过N段仍未完成则切换补丁续写",
        )
        ultra_long_patch_continue_target_chunk_chars: int = Field(
            default=12000,
            description="补丁续写每段目标追加字符数",
        )
        ultra_long_patch_continue_retry_per_turn: int = Field(
            default=3,
            description="补丁续写每段补丁应用不完整时的重试次数",
        )
        ultra_long_patch_continue_model: str = Field(
            default="",
            description="补丁续写模型（留空=评审模型或base_model）",
        )

        # 交付物本地缓存（用于“交付物修改模式”恢复基线，避免反复要求你重新粘贴大代码）
        artifact_cache_enabled: bool = Field(
            default=True,
            description="将最近一次交付物写入本地缓存，后续追问修改时可自动恢复（即使聊天上下文被截断）",
        )
        artifact_cache_dir: str = Field(
            default="",
            description="交付物缓存目录（留空=自动选择 OpenWebUI 数据目录或临时目录）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        artifact_cache_max_chars: int = Field(
            default=400000,
            description="缓存分片的最大字符数（超长交付物会自动分片缓存，不截断）",
        )
        artifact_cache_history_max: int = Field(
            default=20,
            description="每个会话最多保留多少条历史缓存索引",
        )

        # Context carry-over when enabled mid-chat
        history_context_max_messages: int = Field(
            default=60,
            description="启用中途打开时：注入到 scope 的历史消息数（0=禁用）",
        )
        history_context_max_chars: int = Field(
            default=24000,
            description="启用中途打开时：注入到 scope 的历史上下文总字符上限",
        )
        history_context_per_message_max_chars: int = Field(
            default=2000, description="启用中途打开时：每条历史消息最多注入字符数"
        )
        history_context_include_assistant: bool = Field(
            default=True, description="启用中途打开时：历史上下文是否包含助手消息"
        )
        history_context_refresh_mode: str = Field(
            default="always",
            description="启用中途打开时：历史上下文刷新策略（always/on_change/never）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        context_receipt_mode: str = Field(
            default="auto",
            description="上下文接收调试显示：off/auto/always",
            json_schema_extra={"ui": {"type": "text"}},
        )
        context_receipt_max_items: int = Field(
            default=10, description="上下文接收调试：最多展示多少条尾部消息"
        )
        context_receipt_snippet_chars: int = Field(
            default=80, description="上下文接收调试：每条消息预览字符数"
        )

        # Context stitcher (field extraction + packed context)
        context_stitch_enabled: bool = Field(
            default=True,
            description="启用“字段提取 + 上下文拼接”快照：用于夹杂闲聊后仍能正确识别‘修改上一版交付物’的意图。",
        )
        context_stitch_show_status: bool = Field(
            default=True,
            description="在状态输出中显示一次简短的‘上下文拼接摘要’。",
        )
        context_stitch_outline_lines: int = Field(
            default=60,
            description="拼接上下文时最多包含多少行交付物结构字段（baseline_outline）。",
        )
        context_stitch_anchor_lines: int = Field(
            default=18,
            description="拼接上下文时最多包含多少条锚点候选（anchor_candidates）。",
        )
        context_stitch_recent_chars: int = Field(
            default=2400,
            description="拼接上下文时附带的最近对话节选最大字符数。",
        )

        # Follow-up intent guardrails
        followup_new_task_min_confidence: float = Field(
            default=0.78,
            description="追问意图判定为 new_task 的最小置信度阈值（低于阈值会更倾向按‘修改上一版交付物’处理）。",
        )
        followup_regenerate_min_confidence: float = Field(
            default=0.75,
            description="追问意图判定为 regenerate 的最小置信度阈值（低于阈值会更倾向按‘补丁式修改’处理）。",
        )

        # Model overrides (plain text)
        judge_model: str = Field(
            default="",
            description="评审模型（留空=base_model）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        planner_model: str = Field(
            default="",
            description="规划模型（留空=base_model）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        vision_fallback_model: str = Field(
            default="",
            description="图片不兼容时用来“描述图片”的模型（留空=不使用）",
            json_schema_extra={"ui": {"type": "text"}},
        )
        fallback_base_model: str = Field(
            default="",
            description="兜底 base_model（当 base_model 丢失/直接选 pipe 时使用）",
            json_schema_extra={"ui": {"type": "text"}},
        )

        artifact_edit_max_patch_attempts_per_stage: int = Field(
            default=6,
            description="交付物修改模式：每阶段补丁生成/应用失败时最多重试次数（越大越稳，默认6）",
        )
        artifact_edit_max_patches_per_attempt: int = Field(
            default=12,
            description="交付物修改模式：每次让模型输出的 patches 最大条数（太多容易跳过/失配）",
        )
        artifact_edit_retry_on_partial_apply: bool = Field(
            default=True,
            description="交付物修改模式：补丁部分应用/存在 skipped 时自动带反馈重试",
        )
        artifact_edit_commit_partial_apply: bool = Field(
            default=True,
            description="交付物修改模式：部分补丁成功后先保留已成功修改，再继续重试剩余补丁",
        )
        artifact_edit_require_full_apply_for_code: bool = Field(
            default=True,
            description="交付物修改模式：代码类默认要求本轮 patches 全部命中，否则继续重试（更稳）",
        )
        artifact_edit_allowed_skip_ratio: float = Field(
            default=0.0,
            description="交付物修改模式：允许 skipped 的比例（0-1）。代码类建议 0",
        )
        artifact_edit_require_progress: bool = Field(
            default=True,
            description="交付物修改模式：补丁必须产生实际文本变化，否则视为失败并重试",
        )

        artifact_edit_stop_on_stage_fail: bool = Field(
            default=True,
            description="交付物修改模式：若某阶段在重试后仍失败，停止后续阶段并进入兜底整合/保守更新（不再‘跳过’）",
        )
        artifact_edit_show_attempt_status: bool = Field(
            default=True,
            description="交付物修改模式：在状态栏显示每次补丁尝试结果/失败原因（推荐开启）",
        )
        artifact_edit_critical_extra_attempts: int = Field(
            default=4,
            description="交付物修改模式：关键修复阶段额外重试次数（标题含‘修复/启动/报错’等会+此值）",
        )

        artifact_edit_min_stages: int = Field(
            default=3,
            description="交付物修改模式：最少拆分阶段数（太少容易丢内容；建议>=3）",
        )
        artifact_edit_ws_regex_match: bool = Field(
            default=True,
            description="补丁匹配：允许忽略空白差异（将空白折叠为\s+进行匹配）",
        )
        artifact_edit_fuzzy_match: bool = Field(
            default=True,
            description="补丁匹配：启用模糊匹配（容忍少量字符差异，适合‘差一个字符’这种情况）",
        )
        artifact_edit_fuzzy_min_ratio: float = Field(
            default=0.985,
            description="模糊匹配阈值（0-1，越高越保守；建议0.98-0.995）",
        )
        artifact_edit_fuzzy_max_candidates: int = Field(
            default=120,
            description="模糊匹配：最多候选点（越大越稳但更慢）",
        )
        artifact_edit_fuzzy_short_min_len: int = Field(
            default=24,
            description="模糊匹配：最短 find/anchor 长度（过短不安全，会被自动跳过）",
        )
        artifact_edit_fuzzy_ambiguity_margin: float = Field(
            default=0.01,
            description="模糊匹配：歧义保护（best 与 second-best 相似度差值）",
        )
        artifact_edit_fuzzy_hint_min_ratio: float = Field(
            default=0.90,
            description="生成反馈：近似片段提示阈值（0-1；用于帮模型修补锚点）",
        )

        # Router tag
        inject_router_tag: bool = Field(
            default=True, description="注入 <dr_router> 兜底标签（建议开启）"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = _svg_data_uri()

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not isinstance(body, dict):
            return body

        md = _merge_metadata(body, __metadata__)

        # Skip internal calls to prevent recursion
        if md.get(self.valves.internal_skip_flag_key):
            return body

        pipe_model_id = str(self.valves.pipe_model_id or "").strip()
        if not pipe_model_id:
            return body

        cur_model = str(body.get("model") or "").strip()
        if not cur_model:
            return body

        # Determine if enabled
        state = _read_toggle_state(md, FILTER_ID)
        enabled = (
            bool(self.valves.assume_enabled_if_unknown)
            if state is None
            else bool(state)
        )
        if not enabled:
            return body

        # Read per-chat user valves
        uv = None
        if isinstance(__user__, dict):
            uv_root = __user__.get("valves")
            uv = uv_root
            # Some OpenWebUI builds nest per-filter valves under valves[<FILTER_ID>].
            if (
                isinstance(uv_root, dict)
                and FILTER_ID in uv_root
                and isinstance(uv_root.get(FILTER_ID), dict)
            ):
                uv = uv_root.get(FILTER_ID)

        def uv_get(key: str, default: Any) -> Any:
            try:
                if uv is None:
                    return default
                if isinstance(uv, dict) and key in uv:
                    return uv.get(key) if uv.get(key) is not None else default
                if hasattr(uv, key):
                    v = getattr(uv, key)
                    return v if v is not None else default
            except Exception:
                return default
            return default

        status_mode = str(uv_get("status_mode", "simple") or "simple").strip().lower()
        if status_mode not in ("simple", "debug"):
            status_mode = "simple"

        config = {
            "max_rounds": max(1, _as_int(uv_get("max_rounds", 24), 24)),
            "min_rounds": max(1, _as_int(uv_get("min_rounds", 1), 1)),
            "max_tool_turns_per_round": max(
                1, _as_int(uv_get("max_tool_turns_per_round", 12), 12)
            ),
            "max_tool_calls_per_turn": max(
                0, _as_int(uv_get("max_tool_calls_per_turn", 12), 12)
            ),
            "max_output_continue_turns": max(
                0, _as_int(uv_get("max_output_continue_turns", 60), 60)
            ),
            "chunk_max_chars": max(
                1000, _as_int(uv_get("chunk_max_chars", 40000), 40000)
            ),
            "chunk_policy": str(uv_get("chunk_policy", "auto") or "auto")
            .strip()
            .lower(),
            "enable_clarify": _as_bool(uv_get("enable_clarify", True), True),
            "clarify_max_repeat_same_q": max(
                0, _as_int(uv_get("clarify_max_repeat_same_q", 1), 1)
            ),
            "enable_assumption_mode": _as_bool(
                uv_get("enable_assumption_mode", True), True
            ),
            "enable_artifact_edit_mode": _as_bool(
                uv_get("enable_artifact_edit_mode", True), True
            ),
            "artifact_edit_max_patch_attempts_per_stage": max(
                1, _as_int(uv_get("artifact_edit_max_patch_attempts_per_stage", 6), 6)
            ),
            "artifact_edit_max_patches_per_attempt": max(
                1, _as_int(uv_get("artifact_edit_max_patches_per_attempt", 12), 12)
            ),
            "artifact_edit_retry_on_partial_apply": _as_bool(
                uv_get("artifact_edit_retry_on_partial_apply", True), True
            ),
            "artifact_edit_commit_partial_apply": _as_bool(
                uv_get("artifact_edit_commit_partial_apply", True), True
            ),
            "artifact_edit_require_full_apply_for_code": _as_bool(
                uv_get("artifact_edit_require_full_apply_for_code", True), True
            ),
            "artifact_edit_allowed_skip_ratio": _as_float(
                uv_get("artifact_edit_allowed_skip_ratio", 0.0), 0.0
            ),
            "artifact_edit_require_progress": _as_bool(
                uv_get("artifact_edit_require_progress", True), True
            ),
            "artifact_edit_stop_on_stage_fail": _as_bool(
                uv_get("artifact_edit_stop_on_stage_fail", True), True
            ),
            "artifact_edit_show_attempt_status": _as_bool(
                uv_get("artifact_edit_show_attempt_status", True), True
            ),
            "artifact_edit_critical_extra_attempts": max(
                0, _as_int(uv_get("artifact_edit_critical_extra_attempts", 4), 4)
            ),
            "artifact_edit_min_stages": max(
                1, _as_int(uv_get("artifact_edit_min_stages", 3), 3)
            ),
            "artifact_edit_ws_regex_match": _as_bool(
                uv_get("artifact_edit_ws_regex_match", True), True
            ),
            "artifact_edit_fuzzy_match": _as_bool(
                uv_get("artifact_edit_fuzzy_match", True), True
            ),
            "artifact_edit_fuzzy_min_ratio": _as_float(
                uv_get("artifact_edit_fuzzy_min_ratio", 0.985), 0.985
            ),
            "artifact_edit_fuzzy_max_candidates": max(
                10, _as_int(uv_get("artifact_edit_fuzzy_max_candidates", 120), 120)
            ),
            "artifact_edit_fuzzy_short_min_len": max(
                8, _as_int(uv_get("artifact_edit_fuzzy_short_min_len", 24), 24)
            ),
            "artifact_edit_fuzzy_ambiguity_margin": _as_float(
                uv_get("artifact_edit_fuzzy_ambiguity_margin", 0.01), 0.01
            ),
            "artifact_edit_fuzzy_hint_min_ratio": _as_float(
                uv_get("artifact_edit_fuzzy_hint_min_ratio", 0.90), 0.90
            ),
            "confirm_policy": str(uv_get("confirm_policy", "pause") or "pause")
            .strip()
            .lower(),
            "enable_dynamic_plan": _as_bool(uv_get("enable_dynamic_plan", True), True),
            "enable_judge_rewrite": _as_bool(
                uv_get("enable_judge_rewrite", True), True
            ),
            "enable_final_compile": _as_bool(
                uv_get("enable_final_compile", True), True
            ),
            "prefer_fast_followup": _as_bool(
                uv_get("prefer_fast_followup", True), True
            ),
            "prefer_patch_over_regenerate": _as_bool(
                uv_get("prefer_patch_over_regenerate", True), True
            ),
            "followup_force_full_context": _as_bool(
                uv_get("followup_force_full_context", True), True
            ),
            # Artifact Detach
            "artifact_detach_enabled": _as_bool(
                uv_get("artifact_detach_enabled", True), True
            ),
            "artifact_detach_min_chars": max(
                0, _as_int(uv_get("artifact_detach_min_chars", 60000), 60000)
            ),
            "artifact_detach_threshold_mode": str(
                uv_get("artifact_detach_threshold_mode", "auto") or "auto"
            )
            .strip()
            .lower(),
            "artifact_detach_smart_model": str(
                uv_get("artifact_detach_smart_model", "") or ""
            ).strip(),
            "artifact_detach_smart_lower_bound": max(
                0, _as_int(uv_get("artifact_detach_smart_lower_bound", 20000), 20000)
            ),
            "artifact_detach_smart_upper_bound": max(
                0, _as_int(uv_get("artifact_detach_smart_upper_bound", 180000), 180000)
            ),
            "artifact_detach_smart_ambiguous_band": max(
                0, _as_int(uv_get("artifact_detach_smart_ambiguous_band", 15000), 15000)
            ),
            "artifact_detach_smart_snippet_chars": max(
                200, _as_int(uv_get("artifact_detach_smart_snippet_chars", 1200), 1200)
            ),
            "artifact_detach_smart_max_tokens": max(
                64, _as_int(uv_get("artifact_detach_smart_max_tokens", 220), 220)
            ),
            "artifact_detach_max_full_inject_chars": max(
                0,
                _as_int(
                    uv_get("artifact_detach_max_full_inject_chars", 200000), 200000
                ),
            ),
            "artifact_detach_head_chars": max(
                0, _as_int(uv_get("artifact_detach_head_chars", 50000), 50000)
            ),
            "artifact_detach_tail_chars": max(
                0, _as_int(uv_get("artifact_detach_tail_chars", 45000), 45000)
            ),
            "artifact_detach_max_snippets": max(
                0, _as_int(uv_get("artifact_detach_max_snippets", 10), 10)
            ),
            "artifact_detach_snippet_window": max(
                200, _as_int(uv_get("artifact_detach_snippet_window", 1200), 1200)
            ),
            "artifact_detach_outline_max_items": max(
                10, _as_int(uv_get("artifact_detach_outline_max_items", 60), 60)
            ),
            "followup_anti_regression": _as_bool(
                uv_get("followup_anti_regression", True), True
            ),
            "followup_anti_regression_min_ratio": _as_float(
                uv_get("followup_anti_regression_min_ratio", 0.8), 0.8
            ),
            "followup_anti_regression_min_anchor_hits": max(
                0, _as_int(uv_get("followup_anti_regression_min_anchor_hits", 3), 3)
            ),
            "skip_final_compile_for_code": _as_bool(
                uv_get("skip_final_compile_for_code", True), True
            ),
            "compile_model": str(uv_get("compile_model", "") or "").strip(),
            "compile_max_output_continue_turns": max(
                0, _as_int(uv_get("compile_max_output_continue_turns", 30), 30)
            ),
            "final_min_chars": max(0, _as_int(uv_get("final_min_chars", 8000), 8000)),
            "compile_step_output_max_chars": max(
                1000, _as_int(uv_get("compile_step_output_max_chars", 20000), 20000)
            ),
            "status_heartbeat_interval_sec": max(
                0, _as_int(uv_get("status_heartbeat_interval_sec", 120), 120)
            ),
            "progress_summary_enabled": _as_bool(
                uv_get("progress_summary_enabled", True), True
            ),
            "progress_summary_model": str(
                uv_get("progress_summary_model", "") or ""
            ).strip(),
            "progress_summary_every_n_continues": max(
                1, _as_int(uv_get("progress_summary_every_n_continues", 1), 1)
            ),
            "progress_summary_prompt_max_chars": max(
                400, _as_int(uv_get("progress_summary_prompt_max_chars", 2500), 2500)
            ),
            "progress_summary_max_tokens": max(
                64, _as_int(uv_get("progress_summary_max_tokens", 220), 220)
            ),
            "continue_context_max_chars": max(
                0, _as_int(uv_get("continue_context_max_chars", 0), 0)
            ),
            "ultra_long_patch_continue_enabled": _as_bool(
                uv_get("ultra_long_patch_continue_enabled", True), True
            ),
            "ultra_long_patch_continue_min_chars": max(
                0, _as_int(uv_get("ultra_long_patch_continue_min_chars", 30000), 30000)
            ),
            "ultra_long_patch_continue_switch_after_n": max(
                0, _as_int(uv_get("ultra_long_patch_continue_switch_after_n", 2), 2)
            ),
            "ultra_long_patch_continue_target_chunk_chars": max(
                2000,
                _as_int(
                    uv_get("ultra_long_patch_continue_target_chunk_chars", 12000), 12000
                ),
            ),
            "ultra_long_patch_continue_retry_per_turn": max(
                0, _as_int(uv_get("ultra_long_patch_continue_retry_per_turn", 3), 3)
            ),
            "ultra_long_patch_continue_model": str(
                uv_get("ultra_long_patch_continue_model", "") or ""
            ).strip(),
            "artifact_cache_enabled": _as_bool(
                uv_get("artifact_cache_enabled", True), True
            ),
            "artifact_cache_dir": str(uv_get("artifact_cache_dir", "") or "").strip(),
            "artifact_cache_max_chars": max(
                0, _as_int(uv_get("artifact_cache_max_chars", 400000), 400000)
            ),
            "artifact_cache_history_max": max(
                0, _as_int(uv_get("artifact_cache_history_max", 20), 20)
            ),
            "history_context_max_messages": max(
                0, _as_int(uv_get("history_context_max_messages", 60), 60)
            ),
            "history_context_max_chars": max(
                0, _as_int(uv_get("history_context_max_chars", 24000), 24000)
            ),
            "history_context_per_message_max_chars": max(
                200,
                _as_int(uv_get("history_context_per_message_max_chars", 2000), 2000),
            ),
            "history_context_include_assistant": _as_bool(
                uv_get("history_context_include_assistant", True), True
            ),
            "history_context_refresh_mode": str(
                uv_get("history_context_refresh_mode", "always") or "always"
            ).strip(),
            "context_receipt_mode": str(
                uv_get("context_receipt_mode", "auto") or "auto"
            ).strip(),
            "context_receipt_max_items": max(
                3, _as_int(uv_get("context_receipt_max_items", 10), 10)
            ),
            "context_receipt_snippet_chars": max(
                20, _as_int(uv_get("context_receipt_snippet_chars", 80), 80)
            ),
            # Context stitcher (field extraction + packed context)
            "context_stitch_enabled": bool(uv_get("context_stitch_enabled", True)),
            "context_stitch_show_status": bool(
                uv_get("context_stitch_show_status", True)
            ),
            "context_stitch_outline_lines": max(
                10, _as_int(uv_get("context_stitch_outline_lines", 60), 60)
            ),
            "context_stitch_anchor_lines": max(
                6, _as_int(uv_get("context_stitch_anchor_lines", 18), 18)
            ),
            "context_stitch_recent_chars": max(
                400, _as_int(uv_get("context_stitch_recent_chars", 2400), 2400)
            ),
            # Follow-up intent guardrails
            "followup_new_task_min_confidence": float(
                uv_get("followup_new_task_min_confidence", 0.78) or 0.78
            ),
            "followup_regenerate_min_confidence": float(
                uv_get("followup_regenerate_min_confidence", 0.75) or 0.75
            ),
            "judge_model": str(uv_get("judge_model", "") or "").strip(),
            "planner_model": str(uv_get("planner_model", "") or "").strip(),
            "vision_fallback_model": str(
                uv_get("vision_fallback_model", "") or ""
            ).strip(),
            "fallback_base_model": str(uv_get("fallback_base_model", "") or "").strip(),
        }
        if config["min_rounds"] > config["max_rounds"]:
            config["min_rounds"] = config["max_rounds"]

        if config["chunk_policy"] not in ("auto", "manual"):
            config["chunk_policy"] = "auto"

        if config["confirm_policy"] not in ("auto", "always", "never", "pause"):
            config["confirm_policy"] = "pause"

        # Follow-up anti-regression ratio clamp
        try:
            r = float(config.get("followup_anti_regression_min_ratio", 0.8) or 0.8)
            if r < 0.3:
                r = 0.3
            if r > 1.0:
                r = 1.0
            config["followup_anti_regression_min_ratio"] = r
        except Exception:
            config["followup_anti_regression_min_ratio"] = 0.8

        ui = {
            "mode": status_mode,
            "show_timing": _as_bool(uv_get("show_timing", True), True),
            "append_timing_to_final": _as_bool(
                uv_get("append_timing_to_final", False), False
            ),
            "show_tool_detail": _as_bool(
                uv_get("show_tool_detail_in_debug", True), True
            ),
            "show_alignment": _as_bool(uv_get("show_alignment", True), True),
        }

        inject_tag = _as_bool(uv_get("inject_router_tag", True), True)

        # Read existing router tag in the incoming messages (if any)
        router_tag = _parse_router_tag(body.get("messages"))

        # session_id: stable derivation
        session_id = _derive_session_id(
            body=body, md=md, router_tag=router_tag, user=__user__
        )

        # chat_id: best-effort
        chat_id = (
            _read_chat_id_from_obj(body)
            or _read_chat_id_from_obj(md)
            or _read_chat_id_from_obj(router_tag)
        )

        # base_model
        dr_prev = (
            md.get("deep_research") if isinstance(md.get("deep_research"), dict) else {}
        )
        base_model = ""
        if cur_model != pipe_model_id:
            base_model = cur_model
        else:
            base_model = str(
                (dr_prev.get("base_model") if isinstance(dr_prev, dict) else "")
                or md.get("dr_original_model")
                or (
                    router_tag.get("base_model") if isinstance(router_tag, dict) else ""
                )
                or config.get("fallback_base_model")
                or ""
            ).strip()

        # write metadata
        md2 = deepcopy(md)
        if base_model:
            md2["dr_original_model"] = base_model

        # also expose chat_id at top-level when available
        if chat_id and (
            not isinstance(md2.get("chat_id"), str)
            or not str(md2.get("chat_id")).strip()
        ):
            md2["chat_id"] = chat_id

        dr = md2.get("deep_research")
        if not isinstance(dr, dict):
            dr = {}
        dr = dict(dr)

        dr["enabled"] = True
        dr["pipe_model_id"] = pipe_model_id
        dr["router_filter_id"] = FILTER_ID
        dr["routed_at"] = time.time()
        dr["session_id"] = session_id
        dr["config"] = config
        dr["ui"] = ui
        if chat_id:
            dr["chat_id"] = chat_id
        if base_model:
            dr["base_model"] = base_model

        md2["deep_research"] = dr

        body2 = deepcopy(body)
        body2["metadata"] = md2

        # route
        if cur_model != pipe_model_id:
            body2["model"] = pipe_model_id

        # inject router tag as fallback
        if inject_tag:
            payload = {
                "session_id": session_id,
                "chat_id": chat_id,
                "base_model": base_model,
                "pipe_model_id": pipe_model_id,
                "router_filter_id": FILTER_ID,
                "config": config,
                "ui": ui,
                "routed_at": dr["routed_at"],
            }
            body2["messages"] = _ensure_router_tag(body2.get("messages"), payload)

        return body2
