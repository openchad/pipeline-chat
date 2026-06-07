from openchadpy.tool_base import ToolRegistry
from openchadpy.pipeline_base import PipelineBase
import asyncio
from parser import Parser
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import json
import time
import copy
import re
logger = logging.getLogger(__name__)

def _format_chunk(chunk: Any, max_length: int = 200) -> str:
    """Gracefully format any chunk type for logging."""
    try:
        if chunk is None:
            return "None"
        elif isinstance(chunk, (bytes, bytearray)):
            text = chunk.decode("utf-8", errors="replace")
            preview = text[:max_length]
            suffix = f"... [{len(text)} chars]" if len(text) > max_length else ""
            return f"bytes({preview!r}{suffix})"
        elif isinstance(chunk, str):
            preview = chunk[:max_length]
            suffix = f"... [{len(chunk)} chars]" if len(chunk) > max_length else ""
            return f"str({preview!r}{suffix})"
        elif isinstance(chunk, dict):
            preview = json.dumps(chunk, default=str)
            if len(preview) > max_length:
                preview = preview[:max_length] + f"... [{len(chunk)} keys]"
            return f"dict({preview})"
        elif isinstance(chunk, (list, tuple)):
            type_name = type(chunk).__name__
            sample = chunk[:5]
            preview = json.dumps(list(sample), default=str)
            suffix = f" ... [{len(chunk)} items total]" if len(chunk) > 5 else ""
            return f"{type_name}({preview}{suffix})"
        elif isinstance(chunk, (int, float, bool)):
            return f"{type(chunk).__name__}({chunk})"
        else:
            r = repr(chunk)
            if len(r) > max_length:
                return r[:max_length] + f"... [type={type(chunk).__name__}]"
            return r
    except Exception as e:
        return f"<unformattable chunk type={type(chunk).__name__} error={e}>"

def _sha256_short(input_str: str) -> str:
    """SHA-256 hash truncated to 32 hex chars."""
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()[:32]

def _make_empty_model_output(model: Optional[str] = "") -> Dict[str, Any]:
    """Create an empty ModelOutput dict."""
    return {
        "isStreaming": True,
        "content": "<div></div>",
        "token_per_second": None,
        "costs": [],
        "model": model or "",
        "date": int(time.time()),
    }

def _extract_content_from_response(response: Any) -> str:
    """Extract the text content from a ModelOutput."""
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    return ""

def _escape_xml_attr(s: str) -> str:
    """Escape a string for use inside a double-quoted XML attribute."""
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
_RE_THINK_BLOCK = re.compile(
    r"<Think>.*?</Think>",
    re.DOTALL,
)
_RE_TOOL_CALL = re.compile(
    r"<ToolCall\b[^>]*/>",
    re.DOTALL,
)
_RE_CODE_BLOCK = re.compile(
    r"<CodeBlock\b[^>]*>.*?</CodeBlock>",
    re.DOTALL,
)

def _clean_assistant_content(text: str) -> str:
    """Strip MDX-rendered tags (think, tool call, code block wrappers) from
    assistant responses before they are inserted into the history context.
    The raw markdown content inside CodeBlock is preserved.
    """
    # Remove think blocks entirely
    text = _RE_THINK_BLOCK.sub("", text)
    # Remove ToolCall self-closing tags
    text = _RE_TOOL_CALL.sub("", text)
    # Unwrap CodeBlock  keep the inner ``` fence
    text = _RE_CODE_BLOCK.sub(lambda m: re.sub(r"<CodeBlock\b[^>]*>", "", m.group(0)).replace("</CodeBlock>", ""), text)
    return text.strip()

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return default
    return data

# Content segment types
# Each segment is a dict with a "type" key:
#   {"type": "text",       "content": str}
#   {"type": "tool_call",  "id": str, "name": str, "parameters": str}
#   {"type": "code_block", "id": str, "lang": str, "code": str}

class Chat(PipelineBase):
    logs: Dict[str, Any]
    r: Dict[str, Any]
    message_template: List[Dict[str, Any]]
    content: str
    tool_logs: List[List[Dict[str, Any]]]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = re.sub(
            r"^(\s*)\.",
            "",
            """
        .You are a helpful assistant.
        """,
            flags=re.MULTILINE,
        ).strip()
        self.message_template = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]
        self.tools = self.tool_manager.get_openai_schemas() if self.tool_manager else []
        if self.mcp_manager:
            self.tools.extend(self.mcp_manager.get_openai_schemas())
        self.logs = {}
        self.parser = Parser(detect_code_blocks=False)
        self._stream_start_time = 0.0
        self._stream_end_time = 0.0
        self._completion_tokens = 0
        self._prompt_tokens = 0
        self._costs = []
        self.r: Dict[str, Any] = {}
        self.content = ""
        self.tool_logs = []
        # Think content is tracked separately and always rendered at the top.
        self._think_content: str = ""       # all completed think block text
        self._think_in_progress: bool = False
        # Ordered content segments (text / tool_call / code_block).
        self._content_segments: List[Dict] = []
        # Text that has been parsed by the parser but not yet flushed into a
        # segment.  Gets flushed whenever a tool_call or code_block finalises.
        self._pending_text: str = ""
        # How many chars of parser.parsed_buffer we have already consumed.
        self._consumed_parsed_len: int = 0
        # Counters for unique element IDs (reset in start()).
        self._tool_call_counter: int = 0
        self._code_block_counter: int = 0
        # Native (provider-level) tool calls that have already been written
        # to _content_segments so we don't duplicate them.
        self._serialized_native_tc_ids: set = set()
        # Native TCs whose args are complete but whose insertion is deferred
        # until content_delta stops flowing, so preceding text is fully flushed.
        self._queued_native_tcs: List[Dict] = []
        
    
    # Content helpers
    
    def _next_tool_id(self) -> str:
        tc_id = f"tc_{self._tool_call_counter}"
        self._tool_call_counter += 1
        return tc_id
    def _next_code_id(self) -> str:
        cb_id = f"cb_{self._code_block_counter}"
        self._code_block_counter += 1
        return cb_id
    def _flush_pending_text(self) -> None:
        """Move accumulated pending text into a text segment (if non-empty)."""
        if self._pending_text:
            self._content_segments.append(
                {"type": "text", "content": self._pending_text}
            )
            self._pending_text = ""
    def _render_tool_call_tag(self, tc_id: str, name: str) -> str:
        """Self-closing tool_call XML tag  always valid, never left open."""
        return (
            f'<ToolCall id="{tc_id}" name="{_escape_xml_attr(name)}"/>'
        )
    def _split_lang_code(self, buf: str):
        """Extract (lang, code) from a code_buffer whose first line is the language tag.
        The parser now stores the language as the first line of code_buffer."""
        first_nl = buf.find("\n")
        if first_nl != -1:
            first_line = buf[:first_nl].strip()
            if re.match(r"^[a-zA-Z0-9_+\-]+$", first_line):
                return first_line, buf[first_nl + 1:]
        return "", buf

    def _render_code_block_tag(self, cb_id: str, lang: str, code: str) -> str:
        """code_block wrapper  always closed even for in-progress content."""
        lang_tag = lang or ""
        return (
            f'<CodeBlock id="{cb_id}">\n'
            f"```{lang_tag}\n{code}\n```\n"
            f"</CodeBlock>"
        )
    def _build_content(
        self,
        *,
        is_parsing_think: bool = False,
        current_think_buf: str = "",
        is_parsing_code: bool = False,
        current_code_buf: str = "",
        code_lang: str = "",
        is_parsing_tool: bool = False,
    ) -> str:
        """
        Assemble the full rendered content string from current state.
        Structure:
          <Think>…</Think>          ← if any think content exists
          content 0
          <ToolCall id=… …/>
          content 1
          <CodeBlock id=…>…</CodeBlock>
          …
          [in-progress code block]  ← always closed
          [in-progress tool call]   ← placeholder, always closed
        """
        parts: List[str] = []
        think_text = self._think_content
        if is_parsing_think and current_think_buf:
            think_text = think_text + current_think_buf
        if think_text.strip():
            parts.append(f"<Think>\n{think_text}\n</Think>")
        for seg in self._content_segments:
            t = seg["type"]
            if t == "text":
                parts.append(seg["content"])
            elif t == "tool_call":
                parts.append(
                    self._render_tool_call_tag(
                        seg["id"], seg["name"]
                    )
                )
            elif t == "code_block":
                parts.append(
                    self._render_code_block_tag(
                        seg["id"], seg.get("lang", ""), seg["code"]
                    )
                )
        if self._pending_text:
            parts.append(self._pending_text)
        if is_parsing_code and current_code_buf:
            cb_id = f"cb_{self._code_block_counter}"  # peek, don't increment
            _lang, _code = self._split_lang_code(current_code_buf)
            parts.append(self._render_code_block_tag(cb_id, _lang, _code))

        if is_parsing_tool:
            parts.append('<ToolCall id="pending" name="" parameters=""/>')
        # Show placeholders so the UI knows tool calls are coming, even
        # though they haven't been committed to _content_segments yet.
        for qtc in self._queued_native_tcs:
            parts.append(
                self._render_tool_call_tag(qtc["id"], qtc["name"])
            )
        return "\n".join(p for p in parts if p)
    
    # Delta extraction helpers
    
    def _consume_new_parsed_text(self) -> str:
        """
        Return only the *new* text that the parser has confirmed since the last
        call.  Uses parser.parsed_buffer (monotonically growing confirmed text)
        as the source of truth, so we never re-process the same chars.
        Think tags are stripped from the delta here and routed to
        _think_content / _think_in_progress instead.
        """
        raw_delta = self.parser.parsed_buffer[self._consumed_parsed_len:]
        self._consumed_parsed_len = len(self.parser.parsed_buffer)
        if not raw_delta:
            return ""
        completed_thinks = re.findall(
            r"<Think>(.*?)</Think>", raw_delta, flags=re.DOTALL
        )
        for think_text in completed_thinks:
            self._think_content += think_text
        clean_delta = re.sub(r"<Think>.*?</Think>", "", raw_delta, flags=re.DOTALL)
        # (The parser is still inside a think block  content tracked via
        # current_think_buf from the parser return value.)
        clean_delta = re.sub(r"<Think>[^<]*$", "", clean_delta)
        return clean_delta
    
    # Tool-call accumulation (unchanged logic, same as original)
    
    def _update_tool_calls(self, incoming: List[Dict[str, Any]]) -> None:
        """Accumulates tool call deltas to maintain full state during streaming."""
        if not incoming:
            return
        if self.tool_calls is None:
            self.tool_calls = []
        for tc in incoming:
            idx = tc.get("index")
            if idx is not None:
                existing = next(
                    (x for x in self.tool_calls if x.get("index") == idx), None
                )
                if not existing:
                    self.tool_calls.append(copy.deepcopy(tc))
                else:
                    if "function" in tc:
                        f_inc = tc["function"]
                        f_ext = existing.setdefault("function", {})
                        if f_inc.get("name"):
                            f_ext["name"] = f_inc["name"]
                        if "arguments" in f_inc:
                            curr_args = f_ext.get("arguments", "")
                            if isinstance(curr_args, str):
                                f_ext["arguments"] = curr_args + (
                                    f_inc["arguments"] or ""
                                )
                            else:
                                f_ext["arguments"] = f_inc["arguments"]
            else:
                name = safe_get(tc, "function", "name")
                if name:
                    is_new = not any(
                        safe_get(e, "function", "name") == name
                        and safe_get(e, "function", "arguments")
                        == safe_get(tc, "function", "arguments")
                        for e in self.tool_calls
                    )
                    if is_new:
                        self.tool_calls.append(copy.deepcopy(tc))
    
    # Cost calculation
    
    def _calculate_costs(
        self,
        pricing: Optional[Dict[str, Any]],
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ):
        if not pricing:
            return []
        prompt_price = pricing.get("prompt", 0.0)
        if prompt_tokens and prompt_price:
            self._costs.append(
                {
                    "type": "input",
                    "description": f"Input tokens ({prompt_tokens})",
                    "cost": round(prompt_tokens * prompt_price, 10),
                }
            )
        completion_price = pricing.get("completion", 0.0)
        if completion_tokens and completion_price:
            self._costs.append(
                {
                    "type": "output",
                    "description": f"Output tokens ({completion_tokens})",
                    "cost": round(completion_tokens * completion_price, 10),
                }
            )
        cache_read_price = pricing.get("input_cache_read", 0.0)
        if cache_read_tokens and cache_read_price:
            self._costs.append(
                {
                    "type": "input",
                    "description": f"Cached input read ({cache_read_tokens})",
                    "cost": round(cache_read_tokens * cache_read_price, 10),
                }
            )
        cache_write_price = pricing.get("input_cache_write", 0.0)
        if cache_write_tokens and cache_write_price:
            self._costs.append(
                {
                    "type": "input",
                    "description": f"Cached input write ({cache_write_tokens})",
                    "cost": round(cache_write_tokens * cache_write_price, 10),
                }
            )
        return self._costs
    
    # PipelineBase lifecycle
    
    async def get_history(self) -> List[Dict[str, Any]]:
        """Walk the recursive message chain from the DB, mirroring frontend MessageContainer logic.
        Returns a list of {role, content} dicts in chronological order,
        up to (but not including) the current request being processed.
        Only the last MAX_HISTORY_TURNS exchanges are collected.
        NOTE: The loop must always start from index=0 / branch=sha256("0") because
        each branch hash depends on the previous one  there is no shortcut to jump
        directly to turn N-8.  However, we derive the target index from self.tb
        (format: msg_{branch}_{index}) and only append entries once we are within
        MAX_HISTORY_TURNS steps of the end, so old turns are never stored in memory.
        """
        MAX_HISTORY_TURNS = 2
        result: List[Dict[str, str]] = []
        # Starting values – same as the frontend:
        #   index  = 0
        #   branch = sha256("0").slice(0, 32)
        if not self.tb or not isinstance(self.tb, str):
            logger.error("[%s] get_history called without valid tb | self.tb=%s", self.__class__.__name__, self.tb)
            return []
        # Derive the index of the current (target) message from self.tb so we
        # know at which iteration to start collecting.
        try:
            target_index = int(self.tb.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            target_index = None  # fallback: collect everything, slice later
        collect_from = (target_index - MAX_HISTORY_TURNS) if target_index is not None else 0
        current_index = 0
        current_branch = _sha256_short("0")
        logger.info("[%s] get_history starting | self.tb=%s target_index=%s collect_from=%s",
                    self.__class__.__name__, self.tb, target_index, collect_from)
        while True:
            table_name = "msg_" + current_branch + "_" + str(current_index)
            # CRITICAL: Stop once we reach the message currently being generated.
            if table_name == self.tb:
                logger.debug("[%s] get_history reached current table %s | stopping", self.__class__.__name__, table_name)
                break
            if self.tab_db:
                msg = await self.tab_db.get(table_name)
            else:
                logger.error("[%s] get_history called without valid tab_db | self.tab_db=%s", self.__class__.__name__, self.tab_db)
                break
            if not msg or not isinstance(msg, dict):
                logger.debug("[%s] get_history reached end at %s (no data)", self.__class__.__name__, table_name)
                break
            branch_num = msg.get("branch", 0)
            if isinstance(branch_num, str):
                try:
                    branch_num = int(branch_num)
                except ValueError:
                    branch_num = 0
            content = msg.get("content", {})
            # SQLite stores JSON fields as strings  parse them just like setup() does.
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {}
            if not isinstance(content, dict):
                logger.warning("[%s] get_history found invalid content at %s (type=%s)", self.__class__.__name__, table_name, type(content).__name__)
                break
            # Compute the active branch hash – mirrors frontend:
            #   b = sha256(branch > 0 ? branch.toString() + currentBranch : currentBranch).slice(0, 32)
            if branch_num > 0:
                b = _sha256_short(str(branch_num) + current_branch)
            else:
                b = _sha256_short(current_branch)
            branch_data = content.get(b)
            if not branch_data or not isinstance(branch_data, dict):
                logger.debug("[%s] get_history: active branch %s not found in %s", self.__class__.__name__, b, table_name)
                break
            query = branch_data.get("query")
            if not query:
                logger.debug("[%s] get_history: query missing in branch %s of %s", self.__class__.__name__, b, table_name)
                break
            responses = branch_data.get("responses", [])
            response_branch = branch_data.get("response_branch", 0)
            if isinstance(response_branch, str):
                try:
                    response_branch = int(response_branch)
                except ValueError:
                    response_branch = 0
            response_text = None
            candidate = None
            if isinstance(responses, list) and 0 <= response_branch < len(responses):
                candidate = responses[response_branch]
            elif isinstance(responses, dict):
                candidate = responses.get(str(response_branch))
            if candidate is not None:
                # Support both ModelOutput (dict) and legacy string
                text = _extract_content_from_response(candidate)
                if isinstance(candidate, str) and not text:
                    text = candidate  # legacy plain-string response
                if text and text.strip() and text.strip() != "<div></div>":
                    response_text = text
            if response_text is None:
                logger.warning(
                    "[%s] get_history: skipping index=%d branch=%s – no usable response "
                    "(candidate type=%s, extracted=%r)",
                    self.__class__.__name__, current_index, b,
                    type(candidate).__name__, _extract_content_from_response(candidate) if candidate else None,
                )
                # Do NOT break  keep walking so older valid turns are not lost.
                # Advance branch so the chain stays consistent.
                current_index += 1
                current_branch = b
                continue
            # Only collect entries within the last MAX_HISTORY_TURNS window.
            # We still walk every index so the branch hash stays correct.
            if current_index >= collect_from:
                result.append({"role": "user", "content": query})
                result.append({"role": "assistant", "content": response_text})
            # Advance to the next message in the chain
            current_index += 1
            current_branch = b
        logger.info("[%s] get_history complete | collected %d messages (last %d turns)", self.__class__.__name__, len(result) // 2, MAX_HISTORY_TURNS)
        return result
    async def setup(self) -> None:
        if not self.tab_db:
            raise RuntimeError("[setup] database (tab_db) is not configured or is None")
        try:
            if self.tb:
                existing = await self.tab_db.get(self.tb)
                if existing and isinstance(existing, dict):
                    self.r = existing
        except Exception as e:
            logger.error("[%s] Failed to load existing record for %s: %s", self.__class__.__name__, self.tb, e)
        try:
            if not self.r or not isinstance(self.r, dict):
                self.r = {"branch": 0, "content": {}}
            if "content" in self.r and isinstance(self.r["content"], str):
                try:
                    self.r["content"] = json.loads(self.r["content"])
                except Exception:
                    self.r["content"] = {}
            if "content" not in self.r or not isinstance(self.r["content"], dict):
                self.r["content"] = {}
        except Exception as e:
            raise RuntimeError(
                f"[setup] STEP 2 - r structure check failed | r type={type(self.r)} r value={self.r!r}: {e}"
            ) from e
        try:
            if self.branch_id not in self.r["content"]:
                self.r["content"][self.branch_id] = {
                    "query": self.query,
                    "responses": [],
                    "response_branch": 0,
                }
            else:
                self.r["content"][self.branch_id]["query"] = self.query
        except Exception as e:
            raise RuntimeError(
                f"[setup] STEP 3 - branch_id init failed | content type={type(self.r['content'])} content value={self.r['content']!r}: {e}"
            ) from e
        try:
            self.history = await self.get_history()
            logger.info("[%s] !!!!HISTORY!!!!! | %s", self.__class__.__name__, str(self.history))
        except Exception as e:
            raise RuntimeError(f"[setup] STEP 4 - get_history failed: {e}") from e
        try:
            self.r["branch"] = int(self.index or 0)
            self.r["content"][self.branch_id]["response_branch"] = int(
                self.response_branch or 0
            )
        except Exception as e:
            raise RuntimeError(
                f"[setup] STEP 5 - branch assignment failed | branch_id={self.branch_id!r} content={self.r['content']!r}: {e}"
            ) from e
        try:
            responses = self.r["content"][self.branch_id].get("responses", [])
            target_res_branch = int(self.response_branch or 0)
            while len(responses) <= target_res_branch:
                responses.append(_make_empty_model_output(self.model_name))
            self.r["content"][self.branch_id]["responses"] = responses
        except Exception as e:
            raise RuntimeError(
                f"[setup] STEP 6 - responses padding failed | branch_id={self.branch_id!r}: {e}"
            ) from e
        idx: Optional[int] = None
        responses_len: int = 0
        try:
            idx = int(self.response_branch or 0)
            responses_len = len(responses) if 'responses' in locals() else 0
            self.r["content"][self.branch_id]["responses"][idx] = (
                _make_empty_model_output(self.model_name)
            )
            if target_res_branch < len(responses):
                existing_content = _extract_content_from_response(
                    responses[target_res_branch]
                )
                if existing_content and existing_content != "<div></div>":
                    self.content = existing_content
        except Exception as e:
            raise RuntimeError(
                f"[setup] STEP 7 - content extraction failed | idx={idx} responses len={responses_len}: {e}"
            ) from e
        try:
            if self.tab_db and self.tb:
                await self.tab_db.sync(self.tb, self.r)
        except Exception as e:
            raise RuntimeError(f"[setup] STEP 8 - database.sync failed: {e}") from e
        logger.info(
            "[%s] setup complete | branch_id=%s history_len=%d",
            self.__class__.__name__,
            self.branch_id,
            len(self.history),
        )
    
    # process_chunk [core streaming handler]
    async def process_chunk(self, chunk, **kwargs) -> Any:
        if not isinstance(chunk, dict):
            return chunk
        idx = int(self.response_branch or 0)
        current_responses = self.r["content"][self.branch_id]["responses"]
        model_output = current_responses[idx]
        choices = chunk.get("choices", [{}])
        delta = choices[0].get("delta", {}) if choices else {}
        content_delta = delta.get("content", "")
        usage_this_chunk = chunk.get("usage")
        has_explicit_usage = isinstance(usage_this_chunk, dict)
        if has_explicit_usage:
            self._prompt_tokens = usage_this_chunk.get(
                "prompt_tokens", self._prompt_tokens
            )
            ct = usage_this_chunk.get("completion_tokens", 0)
            if ct > 0:
                self._completion_tokens = ct
        # Some providers (e.g. DeepSeek) stream reasoning separately.
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if reasoning_delta:
            self._think_content += reasoning_delta
            self._think_in_progress = True  # will be closed at stream end
        is_parsing_tool = False
        is_parsing_code = False
        current_code_buf = ""
        current_think_buf = ""
        if content_delta:
            try:
                if self._stream_start_time == 0.0:
                    self._stream_start_time = time.time()
                # Parser returns 9 values:
                #   parsed_snapshot      – full text snapshot (confirmed + pending)
                #   tool_calls           – tool calls completed THIS chunk
                #   code_blocks          – code blocks completed THIS chunk
                #   thinks               – think blocks completed THIS chunk
                #   is_parsing_tool      – bool: currently inside a tool call
                #   is_parsing_code      – bool: currently inside a code block
                #   is_parsing_think     – bool: currently inside a think block
                #   current_code_buf     – partial code content being buffered
                #   current_think_buf    – partial think content being buffered
                (
                    _parsed_snapshot,
                    tool_calls,
                    code_blocks_this_chunk,
                    thinks_this_chunk,
                    is_parsing_tool,
                    is_parsing_code,
                    is_parsing_think,
                    current_code_buf,
                    current_think_buf,
                ) = self.parser.process_chunk(content_delta)
                self._think_in_progress = is_parsing_think
                for think_text in thinks_this_chunk:
                    self._think_content += think_text
                new_text = self._consume_new_parsed_text()
                if new_text:
                    self._pending_text += new_text
                for cb in code_blocks_this_chunk:
                    # Detect language: first line of code_buffer is the lang tag
                    lang = ""
                    code = cb
                    first_nl = cb.find("\n")
                    if first_nl != -1:
                        first_line = cb[:first_nl].strip()
                        if re.match(r"^[a-zA-Z0-9_+\-]+$", first_line):
                            lang = first_line
                            code = cb[first_nl + 1:]
                    self._flush_pending_text()
                    cb_id = self._next_code_id()
                    self._content_segments.append(
                        {"type": "code_block", "id": cb_id, "lang": lang, "code": code}
                    )
                    self.logs.setdefault("code_blocks", []).append(cb)
                if tool_calls:
                    self._update_tool_calls(tool_calls)
                    for tc in tool_calls:
                        name = safe_get(tc, "function", "name", default="")
                        args = safe_get(tc, "function", "arguments", default={})
                        if isinstance(args, dict):
                            params_str = json.dumps(args)
                        elif isinstance(args, str):
                            params_str = args
                        else:
                            params_str = "{}"
                        self._flush_pending_text()
                        tc_id = self._next_tool_id()
                        self._content_segments.append(
                            {
                                "type": "tool_call",
                                "id": tc_id,
                                "name": name,
                                "parameters": params_str,
                            }
                        )
                self._stream_end_time = time.time()
                if not has_explicit_usage:
                    self._completion_tokens += 1
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(
                    "[%s] accumulation skipped: %s", self.__class__.__name__, e
                )
        native_tcs = delta.get("tool_calls")
        if native_tcs:
            self._update_tool_calls(native_tcs)
            for tc in self.tool_calls or []:
                tc_provider_id = tc.get("id") or safe_get(tc, "function", "name")
                if tc_provider_id in self._serialized_native_tc_ids:
                    continue
                name = safe_get(tc, "function", "name", default="")
                args = safe_get(tc, "function", "arguments", default="")
                # Only serialise once the arguments are valid JSON (complete)
                if isinstance(args, str):
                    if not args:
                        continue  # args not yet streamed, wait for more chunks
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        continue  # still streaming argument fragments
                elif not isinstance(args, dict):
                    continue  # unexpected type, skip
                params_str = (
                    json.dumps(args) if isinstance(args, dict) else (args or "{}")
                )
                self._serialized_native_tc_ids.add(tc_provider_id)
                seg_id = tc.get("id") or self._next_tool_id()
                self._queued_native_tcs.append(
                    {"id": seg_id, "name": name, "parameters": params_str}
                )

        rendered = self._build_content(
            is_parsing_think=self._think_in_progress,
            current_think_buf=current_think_buf,
            is_parsing_code=is_parsing_code,
            current_code_buf=current_code_buf,
            is_parsing_tool=is_parsing_tool,
        )

        # Fallback: never write an empty string to the DB
        model_output["content"] = rendered or "<div></div>"
        self.logs["content_segments_count"] = len(self._content_segments)
        self.logs["think_content_len"] = len(self._think_content)
        if self.tab_db and self.tb:
            await self.tab_db.sync(self.tb, self.r)
            if self.tab_id:
                await self.tab_db.sync(self.tab_id + "_log", self.logs)
        return chunk
    
    async def finalize(self, **kwargs) -> Any:
        idx = int(self.response_branch or 0)
        total_len = 0
        try:
            model_output = self.r["content"][self.branch_id]["responses"][idx]
            if isinstance(model_output, dict):
                total_len = len(model_output.get("content", ""))
                elapsed = (
                    (self._stream_end_time - self._stream_start_time)
                    if self._stream_start_time
                    else 0
                )
                if elapsed > 0 and self._completion_tokens > 0:
                    model_output["token_per_second"] = int(
                        self._completion_tokens / elapsed
                    )
                model_output["costs"] = self._calculate_costs(
                    self.pricing,
                    self._prompt_tokens,
                    self._completion_tokens,
                )
                model_output["date"] = int(time.time())
            elif isinstance(model_output, str):
                total_len = len(model_output)
        except (KeyError, IndexError, TypeError):
            pass
        logger.info(
            "[%s] finalize | branch_id=%s response_branch=%s | total_length=%d",
            self.__class__.__name__,
            self.branch_id,
            self.response_branch,
            total_len,
        )
        # Flush any remaining parser output
        remaining_raw = self.parser.finalize()
        # Consume any leftover confirmed text
        leftover_text = self._consume_new_parsed_text()
        if leftover_text:
            self._pending_text += leftover_text
        # Close the think block if we were still inside one
        if self._think_in_progress and self.parser.think_buffer:
            self._think_content += self.parser.think_buffer
            self._think_in_progress = False
        # Close any in-progress code block
        if self.parser.in_code_block and self.parser.code_buffer:
            cb_id = self._next_code_id()
            self._flush_pending_text()
            _lang, _code = self._split_lang_code(self.parser.code_buffer)
            self._content_segments.append(
                {
                    "type": "code_block",
                    "id": cb_id,
                    "lang": _lang,
                    "code": _code,
                }
            )
        # Flush remaining pending text
        self._flush_pending_text()
        # Commit any native TCs that were queued but never drained
        # (edge case: stream ended while content_delta was still active).
        for qtc in self._queued_native_tcs:
            self._flush_pending_text()
            self._content_segments.append(
                {
                    "type": "tool_call",
                    "id": qtc["id"],
                    "name": qtc["name"],
                    "parameters": qtc["parameters"],
                }
            )
        self._queued_native_tcs = []
        # Final render (no more in-progress states)
        final_rendered = self._build_content()
        idx = int(self.response_branch or 0)
        try:
            model_output = self.r["content"][self.branch_id]["responses"][idx]
            if isinstance(model_output, dict) and final_rendered:
                model_output["content"] = final_rendered
        except (KeyError, IndexError, TypeError):
            pass
        if self.tab_db and self.tb:
            await self.tab_db.sync(self.tb, self.r)
        if remaining_raw or final_rendered:
            self.logs["final_content"] = final_rendered
            if self.tab_db and self.tab_id:
                await self.tab_db.sync(self.tab_id + "_log", self.logs)
        return remaining_raw or None
    
    # response [non-streaming path]
    async def response(self, res, **kwargs) -> Any:
        """Accumulate content and sync for non-streaming completions."""
        if isinstance(res, dict):
            choices = res.get("choices", [])
            if choices:
                msg_content = choices[0].get("message", {}).get("content", "")
                if msg_content:
                    idx = int(self.response_branch or 0)
                    try:
                        usage = res.get("usage", {})
                        if isinstance(usage, dict):
                            self._prompt_tokens = usage.get("prompt_tokens", 0)
                            self._completion_tokens = usage.get("completion_tokens", 0)
                        else:
                            self._prompt_tokens = (
                                getattr(usage, "prompt_tokens", 0) if usage else 0
                            )
                            self._completion_tokens = (
                                getattr(usage, "completion_tokens", 0) if usage else 0
                            )
                        model_output = _make_empty_model_output(self.model_name)
                        model_output["content"] = self.content + msg_content
                        self.last_response = msg_content
                        model_output["costs"] = self._calculate_costs(
                            self.pricing,
                            self._prompt_tokens,
                            self._completion_tokens,
                        )
                        model_output["date"] = int(time.time())
                        self.r["content"][self.branch_id]["responses"][idx] = (
                            model_output
                        )
                        logger.info(
                            "[%s] response | received full content (%d chars)",
                            self.__class__.__name__,
                            len(msg_content),
                        )
                    except (KeyError, IndexError, TypeError):
                        pass
        if self.tab_db and self.tb:
            await self.tab_db.sync(self.tb, self.r)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[%s] response | branch_id=%s response_branch=%s | response=%s",
                self.__class__.__name__,
                self.branch_id,
                self.response_branch,
                _format_chunk(res),
            )
        return res
    
    # start [reset state for a new attempt]
    async def start(self, **kwargs) -> None:
        if not self.messages:
            # Build history context: clean think/tool tags from assistant turns
            cleaned_history: List[Dict[str, str]] = []
            context = ""
            if self.tab_db and self.tb:
                tab_info = await self.tab_db.get("message_state")
                if isinstance(tab_info["context"], str):
                    context = tab_info["context"]

            for turn in (self.history or []):
                if turn.get("role") == "assistant":
                    cleaned_history.append({
                        "role": "assistant",
                        "content": _clean_assistant_content(turn.get("content", "")),
                    })
                else:
                    cleaned_history.append(turn)
            self.messages = (
                self.message_template
                + cleaned_history
                + [
                    {
                        "role": "user",
                        "content": f"{context}{self.query}",
                    }
                ]
            )
        elif self.tool_calls and isinstance(self.tool_calls, list):
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": self.tool_calls,
                    "content": "",
                }
            )
            self.tool_logs.append(copy.deepcopy(self.tool_calls))
            cancel_sentinel: Optional[asyncio.Task] = None
            if self.cancel_event:
                cancel_sentinel = asyncio.get_event_loop().create_task(
                    self.cancel_event.wait(),
                    name=f"cancel_sentinel_{id(self)}",
                )
            try:
                for tool_call in self.tool_calls:
                    # Fast-path: already cancelled before we even start the next tool.
                    if self.cancel_event and self.cancel_event.is_set():
                        logger.info(
                            "[%s] start | cancelled before tool '%s'  aborting loop",
                            self.__class__.__name__,
                            safe_get(tool_call, "function", "name", default="?"),
                        )
                        break
                    name = safe_get(tool_call, "function", "name", default=None)
                    args_str = safe_get(tool_call, "function", "arguments", default="")
                    tool_result = None
                    if args_str and name:
                        try:
                            p_args = json.loads(args_str)
                        except json.JSONDecodeError:
                            p_args = None
                        if isinstance(p_args, dict):
                            # Wrap the real tool call in a cancellable task.
                            if self.tool_manager:
                                tool_task: asyncio.Task = asyncio.get_event_loop().create_task(
                                    self.tool_manager.execute_tool(name, **p_args),
                                    name=f"tool_{name}_{id(self)}",
                                )
                                try:
                                    if cancel_sentinel is not None:
                                        # Race: whichever finishes first wins.
                                        done, _ = await asyncio.wait(
                                            {tool_task, cancel_sentinel},
                                            return_when=asyncio.FIRST_COMPLETED,
                                        )
                                        if cancel_sentinel in done:
                                            # Stop was requested while the tool was in flight.
                                            logger.info(
                                                "[%s] start | cancel_event fired during tool '%s'  cancelling task",
                                                self.__class__.__name__,
                                                name,
                                            )
                                            tool_task.cancel()
                                            try:
                                                await tool_task
                                            except (asyncio.CancelledError, Exception):
                                                pass
                                            break  # exit the tool loop
                                        # Tool finished normally  retrieve result.
                                        tool_result = tool_task.result()
                                    else:
                                        tool_result = await tool_task
                                except asyncio.CancelledError:
                                    # The outer coroutine itself was cancelled.
                                    tool_task.cancel()
                                    raise
                                except Exception as e:
                                    tool_result = str(e)
                            else:
                                tool_result = "Tool manager is not available"
                    if tool_result:
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(tool_result),
                            }
                        )
            finally:
                # Always clean up the sentinel so it never leaks.
                if cancel_sentinel is not None and not cancel_sentinel.done():
                    cancel_sentinel.cancel()
                    try:
                        await cancel_sentinel
                    except (asyncio.CancelledError, Exception):
                        pass
            self.logs["messages"] = self.messages
        self.args["tools"] = self.tools
        self.tool_calls = None
        self.last_response = ""
        # Flush any buffered text from the previous attempt into a completed
        # segment BEFORE resetting, so prior loop content is preserved.
        self._flush_pending_text()
        self._think_in_progress = False
        self._pending_text = ""
        self._consumed_parsed_len = 0          # reset: new parser = new offset
        self._serialized_native_tc_ids = set() # reset: new native tool calls incoming
        self._queued_native_tcs = []           # reset: no deferred tool calls
        self.parser = Parser()                 # fresh parser for this attempt
    
    # end [end of loop]
    async def end(self, **kwargs) -> None:
        self.content += self.last_response
        is_continue = False
        self.logs["tools_called_" + str(self.attempt)] = json.dumps(self.tool_calls)
        if self.tool_calls:
            is_continue = True
        if self.set_continue:
            self.set_continue(is_continue)
        self.logs["output_" + str(self.attempt)] = self.last_response
        if self.tab_db and self.tab_id:
            await self.tab_db.sync(self.tab_id + "_log", self.logs)
        logger.warning(
            "[%s] !!!!LAST_RESPONSE!!!!! | %s",
            self.__class__.__name__,
            self.last_response,
        )
        logger.warning(
            "[%s] !!!!ATTEMPT!!!!! | %s", self.__class__.__name__, self.attempt
        )
        logger.info("[%s] end")
    
    # stop [end of response]
    async def stop(self, **kwargs) -> None:
        idx = int(self.response_branch or 0)
        current_responses = self.r["content"][self.branch_id]["responses"]
        model_output = current_responses[idx]
        model_output["isStreaming"] = False
        if self.tab_db and self.tb:
            await self.tab_db.set("message_state", "isStreaming", False)
            await self.tab_db.set("message_state", "activeId", "")
            await self.tab_db.sync(self.tb, self.r)
            tab_info = await self.tab_db.get("message_state")
            if not isinstance(tab_info['title'], str):
                async def tool_func(**kwargs):
                    title = kwargs.get("title", None)
                    if title:
                        await self.tab_db.set("message_state", "title", title)
                    return {"result":"OK"}      
                update_title = ToolRegistry(call=tool_func, schema={
                    "type": "function",
                    "function": {
                        "name": "update_title",
                        "description": "Generate a title based on its content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "1-5 word title"
                                }
                            },
                            "required": ["title"]
                        }
                    }
                })
                query = (
                    "Generate a concise, 1-5 word title."
                    "### Guidelines:"
                    "- The title should clearly represent the main theme or subject of the conversation."
                    "- Write the title in the chat's primary language; default to English if multilingual."
                    "- Prioritize accuracy over excessive creativity; keep it clear and simple."
                    "- Ensure no conversational text, affirmations, or explanations precede"
                    "### Chat History:"
                    f"{json.dumps(self.r)}"
                            )
                try:
                    res = await self.llm_tool(query, tool_registry={"update_title": update_title})
                    logger.info(f"[{self.__class__.__name__}] Title: {json.dumps(res)}")
                except Exception as e:
                    logger.error(f"[{self.__class__.__name__}] Error calling LLM: {str(e)}")
        logger.info("[%s] !!!!END_STOP!!!!", self.__class__.__name__)
        logger.info("[%s] !!!!TARGET TABLE!!!!", self.tb)
        