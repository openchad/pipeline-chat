import json
import re
import io
from typing import List, Dict, Any, Optional, Tuple, Union
class Parser:
    """
    Parses text-based tool calls from a stream with high robustness and performance.
    Supports various formats including XML tags, JSON blocks, and indexed calls.
    """

    def __init__(self, yield_deltas: bool = False, detect_tool_calls: bool = True, detect_code_blocks: bool = True):
        self.buffer = ""
        self.in_call = False
        self.call_buffer = ""
        self.current_pattern = None
        self.brace_count = 0
        self.bracket_count = 0
        self.in_think_block = False
        self.in_string = False
        self.string_char = None
        self.is_escaped = False
        self.dynamic_tag_name = None
        self.yield_deltas = yield_deltas
        self.detect_tool_calls = detect_tool_calls
        self.detect_code_blocks = detect_code_blocks
        self.last_yielded_call = None
        self.call_history = [] # List of (name, args_hash)
        self.max_history = 5
        self.parsed_buffer = ""  # Accumulated text content (excludes confirmed tool calls/code blocks)
        self._call_start_tag = ""  # Stores start tag text for non-counting patterns
        # Think block tracking
        self.in_think_block = False
        self.think_buffer = ""
        self.think_blocks = [] # All accumulated think blocks
        # Code block tracking
        self.in_code_block = False
        self.code_buffer = ""
        self.code_blocks = []  # Accumulated code blocks
        self.code_block_depth = 0  # Nesting depth for nested ```python inside code blocks
        # Registry for tool call patterns
        self.registry: Dict[str, Dict[str, Any]] = {
            "gemma": {"start": "call:", "end": "<end_function_call>", "parser": self._parse_gemma},
            "gemma_start": {"start": "<start_function_call>", "end": "<end_function_call>", "parser": self._parse_gemma},
            "mistral": {"start": "[TOOL_CALLS]", "end": "counting", "parser": self._parse_mistral},
            "xml": {"start": "<tool_call>", "end": "</tool_call>", "parser": self._parse_json_block},
            "markdown": {"start": "```json", "end": "```", "parser": self._parse_json_block},
            "tools_tag": {"start": "<tools>", "end": "</tools>", "parser": self._parse_json_block},
            "function_calls": {"start": "<function_calls>", "end": "</function_calls>", "parser": self._parse_function_calls_xml},
            "section_based": {"start": "<|tool_calls_section_begin|>", "end": "<|tool_calls_section_end|>", "parser": self._parse_section_based},
            "json": {"start": '{"tool_calls":', "end": "counting", "parser": self._parse_json_block},
            "json_array": {"start": '[', "end": "counting", "parser": self._parse_json_block},
            "json_object": {"start": '{', "end": "counting", "parser": self._parse_json_block},
        }
        # Regex for dynamic patterns
        self.dynamic_patterns: Dict[str, Dict[str, Any]] = {
            "llama_pythonic": {
                "regex": re.compile(r"\[(\w+)\s*\("),
                "parser": self._parse_llama_pythonic,
                "end": "counting"
            },
            "xml_arg_key_value": {
                "regex": re.compile(r"<tool_call>(\w+)<arg_key>"),
                "parser": self._parse_arg_key_value,
                "end": "</tool_call>"
            },
            "indexed_json": {
                "regex": re.compile(r"(?:^|[\n\r])\s*(\w+)(?::\d+)?:\s*\{"),
                "parser": self._parse_indexed_json,
                "end": "counting"
            },
            "xml_params": {
                "regex": re.compile(r"<(\w+)>\s*<parameter"),
                "parser": self._parse_xml_parameters,
                "end": "dynamic_xml"
            },
            "generic_xml": {
                "regex": re.compile(r"<(\w+)>\s*\{"),
                "parser": self._parse_json_block,
                "end": "dynamic_xml_counting"
            },
            "function_tag": {
                "regex": re.compile(r"<function=([\w\.]+)>"),
                "parser": self._parse_function_tag,
                "end": "</function>"
            },
        }
        # Consolidate static patterns into a single regex for performance
        escaped_starts = [re.escape(v["start"]) for v in self.registry.values()]
        self.static_regex = re.compile("|".join(escaped_starts))
        # Map start tag back to pattern name
        self.start_to_name = {v["start"]: k for k, v in self.registry.items()}

    def _validate_pattern_match(self, pattern_name: str, idx: int, start_tag: str) -> Optional[bool]:
        """Validates whether a pattern match is a true tool call or a false positive. Returns None to defer."""
        if pattern_name == "gemma":
            if idx > 0 and self.buffer[idx-1] not in ('\n', ' ', '\r'):
                return False
            after = self.buffer[idx + len(start_tag) : idx + len(start_tag) + 30].strip()
            if not after and len(self.buffer) - idx < 30:
                return None
            if after and not re.match(r'^[\w\.]+', after):
                return False
        elif pattern_name == "mistral":
            # Should be followed by whitespace and then '[' for the JSON array
            after = self.buffer[idx + len(start_tag) : idx + len(start_tag) + 10].strip()
            if not after and len(self.buffer) - idx < 10:
                return None
            if after and not after.startswith('['):
                return False
        elif pattern_name == "xml":
            # Need sufficient lookahead to determine format
            after = self.buffer[idx + len(start_tag) : idx + len(start_tag) + 50]
            # If we see the arg_key_value format, reject (let dynamic pattern handle it)
            if re.match(r'^\w+<arg_key>', after):
                return False
            # If we don't have enough data yet (< 15 chars), defer matching
            # This allows more data to accumulate so we can make a proper determination
            if len(after) < 15:
                # Check if we have any of: JSON start, function name with {, or complete whitespace
                # These would indicate it's definitely the JSON format
                if after.strip().startswith(('{', '[', '"')):
                    return True
                # If ambiguous, wait for more data
                return None
        elif pattern_name == "json_array":
            after = self.buffer[idx + 1 : idx + 20].strip()
            if not after and len(self.buffer) - idx < 20:
                return None
            # Reject if this looks like Llama pythonic format: [function_name(
            if re.match(r'^\w+\s*\(', after):
                return False
            if after and not after.startswith("{"):
                return False
        elif pattern_name == "json_object":
            # For naked JSON objects, apply strict heuristics
            after = self.buffer[idx + 1 : idx + 150].strip()
            # Must have tool-related keywords near the start
            if not re.search(r'(["\']?(?:name|function|tool|tool_calls|parameters|arguments)["\']?\s*:)', after[:100], re.IGNORECASE):
                if len(after) < 40 and not after.endswith("}"):
                    return None
                return False
            # Should not appear in the middle of prose
            before = self.buffer[max(0, idx-20):idx].strip()
            # If preceded by prose-like text (letters/words), likely false positive
            if before and re.search(r'\w{3,}$', before):
                return False
        return True

    def process_chunk(self, chunk: str) -> Tuple[str, List[Dict[str, Any]], List[str], List[str], bool, bool, bool, str, str]:
        """
        Processes a chunk of text and returns a tuple of (text_to_yield, tool_calls, code_blocks, thinks_this_chunk, is_parsing_tool, is_parsing_code, is_parsing_think, current_code_block, current_think).
        Args:
            chunk: Text chunk to process
        Returns:
            Tuple of:
            - text_to_yield: Regular text content
            - tool_calls: List of parsed tool calls
            - code_blocks: List of code blocks from ```python ... ``` markers
            - thinks_this_chunk: List of finished think blocks in this chunk
            - is_parsing_tool: True if currently parsing a tool call
            - is_parsing_code: True if currently parsing a code block
            - is_parsing_think: True if currently parsing a think block
            - current_code_block: The content of the code block currently being parsed
            - current_think: The content of the think block currently being parsed
        """
        self.buffer += chunk
        text_to_yield = io.StringIO()
        tool_calls = []
        code_blocks_this_chunk = []
        thinks_this_chunk = []
        while True:
            # Handle code blocks with nesting support
            if self.in_code_block:
                handled = False
                while True:
                    idx = self.buffer.find("```")
                    if idx == -1:
                        # No fence found  buffer safely, keep last 2 chars for partial match
                        safe_len = max(0, len(self.buffer) - 2)
                        if safe_len > 0:
                            self.code_buffer += self.buffer[:safe_len]
                            self.buffer = self.buffer[safe_len:]
                        break
                    after_backticks = self.buffer[idx + 3:]
                    # Check if this opens a nested fence (```python, ```js, etc.)
                    nested_match = re.match(r'\w+', after_backticks)
                    if nested_match:
                        # Nested fence open  increment depth and keep buffering
                        self.code_block_depth += 1
                        end_pos = idx + 3 + len(nested_match.group(0))
                        self.code_buffer += self.buffer[:end_pos]
                        self.buffer = self.buffer[end_pos:]
                        continue
                    # It's a closing ``` fence
                    if self.code_block_depth > 0:
                        # Close a nested fence  decrement depth and keep buffering
                        self.code_block_depth -= 1
                        self.code_buffer += self.buffer[:idx + 3]
                        self.buffer = self.buffer[idx + 3:]
                        continue
                    # Depth == 0: this is the real closing fence
                    self.code_buffer += self.buffer[:idx]
                    self.buffer = self.buffer[idx + 3:]
                    if self.code_buffer.strip():
                        self.code_blocks.append(self.code_buffer)
                        code_blocks_this_chunk.append(self.code_buffer)
                    self.code_buffer = ""
                    self.in_code_block = False
                    self.code_block_depth = 0
                    handled = True
                    break
                if handled:
                    continue
                else:
                    break
            if self.in_think_block:
                end_tag = "</think>"
                idx = self.buffer.find(end_tag)
                if idx != -1:
                    content = self.buffer[:idx]
                    self.think_buffer += content
                    self.think_blocks.append(self.think_buffer)
                    thinks_this_chunk.append(self.think_buffer)
                    text_to_yield.write(self.buffer[: idx + len(end_tag)])
                    self.buffer = self.buffer[idx + len(end_tag) :]
                    self.in_think_block = False
                    self.think_buffer = ""
                    continue
                else:
                    safe_len = max(0, len(self.buffer) - len(end_tag) + 1)
                    if safe_len > 0:
                        content = self.buffer[:safe_len]
                        self.think_buffer += content
                        text_to_yield.write(content)
                        self.buffer = self.buffer[safe_len:]
                    break
            if not self.in_call:
                think_start = "<think>"
                think_idx = self.buffer.find(think_start)
                # Check for code block start
                code_idx = self.buffer.find("```") if self.detect_code_blocks else -1
            
                defer_code = False
                if code_idx != -1:
                    newline_idx = self.buffer.find("\n", code_idx)
                    space_idx = self.buffer.find(" ", code_idx)
                    match_end = -1
                    if newline_idx != -1 and space_idx != -1:
                        match_end = min(newline_idx, space_idx)
                    elif newline_idx != -1: match_end = newline_idx
                    elif space_idx != -1: match_end = space_idx
                    if match_end == -1 and len(self.buffer) - code_idx < 15:
                    
                        defer_code = True
                earliest_idx = -1
            
                defer_idx = -1
                found_pattern = None
                pattern_type = None
                match_obj = None
                # Check dynamic patterns FIRST (more specific)
                if self.detect_tool_calls:
                    for p_name, p_data in self.dynamic_patterns.items():
                        match = p_data["regex"].search(self.buffer)
                        if match:
                            idx = match.start()
                            # For indexed_json, validate it's not in the middle of prose
                            if p_name == "indexed_json":
                                # Check if there's non-whitespace immediately before (except at start or newline)
                                if idx > 0:
                                    before_match = self.buffer[:idx]
                                    # If the match doesn't start at beginning or after newline, skip it
                                    if not before_match or before_match[-1] not in ('\n', '\r'):
                                        # Allow leading whitespace
                                        if before_match.strip():
                                            continue
                            if earliest_idx == -1 or idx < earliest_idx:
                                earliest_idx = idx
                                found_pattern = p_name
                                pattern_type = 'dynamic'
                                match_obj = match
                    # Check static patterns (more generic, lower priority)
                    for static_match in self.static_regex.finditer(self.buffer):
                        idx = static_match.start()
                        start_tag = static_match.group(0)
                        p_name = self.start_to_name[start_tag]
                        is_valid = self._validate_pattern_match(p_name, idx, start_tag)
                        if is_valid is False:
                            continue
                        if is_valid is None:
                            if defer_idx == -1 or idx < defer_idx:
                            
                                defer_idx = idx
                            continue
                        if earliest_idx == -1 or idx < earliest_idx:
                            earliest_idx = idx
                            found_pattern = p_name
                            pattern_type = 'static'
                            match_obj = static_match
                            break
                if think_idx != -1 and (earliest_idx == -1 or think_idx < earliest_idx):
                    text_to_yield.write(self.buffer[:think_idx + len(think_start)])
                    self.buffer = self.buffer[think_idx + len(think_start) :]
                    self.in_think_block = True
                    self.think_buffer = ""
                    continue
                # Handle code block start - check before patterns to give it priority
                if code_idx != -1 and not defer_code and (earliest_idx == -1 or code_idx < earliest_idx) and (think_idx == -1 or code_idx < think_idx):
                    # Yield text before code block
                    text_to_yield.write(self.buffer[:code_idx])
                    # Read the language tag
                    newline_idx = self.buffer.find("\n", code_idx)
                    space_idx = self.buffer.find(" ", code_idx)
                    match_end = -1
                    if newline_idx != -1 and space_idx != -1:
                        match_end = min(newline_idx, space_idx)
                    elif newline_idx != -1: match_end = newline_idx
                    elif space_idx != -1: match_end = space_idx
                    else: match_end = len(self.buffer)
                    # Skip the ```marker and optional language tag
                    start_pos = match_end
                    if start_pos < len(self.buffer) and self.buffer[start_pos] == '\n':
                        start_pos += 1
                    self.buffer = self.buffer[start_pos:]
                    self.in_code_block = True
                    self.code_block_depth = 0
                    self.code_buffer = ""
                    continue
                if found_pattern:
                    assert match_obj is not None
                    text_to_yield.write(self.buffer[:earliest_idx])
                    self.in_call = True
                    self.current_pattern = found_pattern
                    self.call_buffer = ""
                    if pattern_type == 'static':
                        p_data = self.registry[found_pattern]
                        start_tag = match_obj.group(0)
                        self.buffer = self.buffer[earliest_idx + len(start_tag) :]
                        if p_data["end"] == "counting":
                            self.brace_count = 1 if start_tag.endswith("{") else 0
                            self.bracket_count = 1 if start_tag.endswith("[") else 0
                            self.call_buffer = start_tag
                            self._call_start_tag = ""
                        else:
                            self.call_buffer = ""
                            self._call_start_tag = start_tag
                    else:
                        p_data = self.dynamic_patterns[found_pattern]
                        self.call_buffer = match_obj.group(0)
                        self.buffer = self.buffer[match_obj.end() :]
                        self.dynamic_tag_name = match_obj.group(1) if match_obj.groups() else None
                        self._call_start_tag = ""
                        if "counting" in p_data["end"]:
                            # For llama_pythonic format, check if call_buffer starts with [
                            if found_pattern == "llama_pythonic":
                                self.bracket_count = 1  # Already have opening [
                                self.brace_count = 0
                            else:
                                self.brace_count = 1 if self.call_buffer.endswith("{") else 0
                                self.bracket_count = 1 if self.call_buffer.endswith("[") else 0
                else:
                    preserve_from = -1
                    if defer_idx != -1:
                        preserve_from = defer_idx
                    if defer_code and code_idx != -1 and (preserve_from == -1 or code_idx < preserve_from):
                        preserve_from = code_idx
                    if preserve_from != -1:
                        if preserve_from > 0:
                            text_to_yield.write(self.buffer[:preserve_from])
                            self.buffer = self.buffer[preserve_from:]
                    else:
                        # We didn't find a pattern to process right now.
                        # Safe streaming with reduced lookahead:
                        max_lookahead = 20
                        if len(self.buffer) > max_lookahead:
                            text_to_yield.write(self.buffer[:-max_lookahead])
                            self.buffer = self.buffer[-max_lookahead:]
                    break
            else:
                assert isinstance(self.current_pattern, str)
                p_data = self.registry.get(self.current_pattern) or self.dynamic_patterns.get(self.current_pattern)
                assert p_data is not None
                end_type = p_data["end"]
                assert isinstance(end_type, str)
                if end_type == "dynamic_xml":
                    end_tag = f"</{self.dynamic_tag_name}>"
                elif end_type == "dynamic_xml_counting":
                    end_tag = f"</{self.dynamic_tag_name}>"
                else:
                    end_tag = end_type
                if end_tag not in ("counting", "dynamic_xml_counting"):
                    idx = self.buffer.find(end_tag)
                    if idx != -1:
                        self.call_buffer += self.buffer[:idx]
                        self.buffer = self.buffer[idx + len(end_tag) :]
                        parsed = p_data["parser"](self.call_buffer)
                        if parsed:
                            calls = parsed if isinstance(parsed, list) else [parsed]
                            for call in calls:
                                if not self._is_duplicate(call):
                                    tool_calls.append(call)
                                    self._record_call(call)
                        self._reset_call_state()
                    else:
                        safe_len = max(0, len(self.buffer) - len(end_tag) + 1)
                        if safe_len > 0:
                            self.call_buffer += self.buffer[:safe_len]
                            self.buffer = self.buffer[safe_len:]
                        if self.yield_deltas:
                            delta = self._get_delta(p_data)
                            if delta: tool_calls.append(delta)
                        break
                else:
                    stop_tag = f"</{self.dynamic_tag_name}>" if end_type == "dynamic_xml_counting" else None
                    found_end = False
                    for i, char in enumerate(self.buffer):
                        if self.is_escaped:
                            self.is_escaped = False
                        elif char == "\\":
                            self.is_escaped = True
                        elif char in ('"', "'"):
                            if not self.in_string:
                                self.in_string = True
                                self.string_char = char
                            elif char == self.string_char:
                                self.in_string = False
                                self.string_char = None
                        elif not self.in_string:
                            if char == "{": self.brace_count += 1
                            elif char == "}": self.brace_count -= 1
                            elif char == "[": self.bracket_count += 1
                            elif char == "]": self.bracket_count -= 1
                        is_end = False
                        if not self.in_string:
                            if self.brace_count == 0 and self.bracket_count == 0:
                                is_end = True
                            elif stop_tag and self.buffer[i:].startswith(stop_tag):
                                is_end = True
                                i -= 1
                        if is_end:
                            self.call_buffer += self.buffer[: i + 1]
                            self.buffer = self.buffer[i + 1 :]
                            if stop_tag and self.buffer.startswith(stop_tag):
                                self.buffer = self.buffer[len(stop_tag):]
                            parsed = p_data["parser"](self.call_buffer)
                            if parsed:
                                calls = parsed if isinstance(parsed, list) else [parsed]
                                for call in calls:
                                    if not self._is_duplicate(call):
                                        tool_calls.append(call)
                                        self._record_call(call)
                            else:
                                text_to_yield.write(self.call_buffer)
                            self._reset_call_state()
                            found_end = True
                            break
                    if not found_end:
                        self.call_buffer += self.buffer
                        self.buffer = ""
                        if self.yield_deltas:
                            delta = self._get_delta(p_data)
                            if delta: tool_calls.append(delta)
                        break
        yielded_text = text_to_yield.getvalue()
        self.parsed_buffer += yielded_text
        # Snapshot: accumulated text + currently held-back text
        if self.in_call:
            parsed_snapshot = self.parsed_buffer + self._call_start_tag + self.call_buffer + self.buffer
        else:
            parsed_snapshot = self.parsed_buffer + self.buffer
        return (
            parsed_snapshot, 
            tool_calls, 
            code_blocks_this_chunk, 
            thinks_this_chunk,
            self.in_call, 
            self.in_code_block, 
            self.in_think_block, 
            self.code_buffer, 
            self.think_buffer, 
        )

    def _get_delta(self, p_data: Dict) -> Optional[Dict]:
        """Extracts partial tool call as a delta, avoiding redundant yields."""
        try:
            temp_buffer = self.call_buffer
            if self.brace_count > 0: temp_buffer += "}" * self.brace_count
            if self.bracket_count > 0: temp_buffer += "]" * self.bracket_count
            parsed = p_data["parser"](temp_buffer)
            if parsed:
                call = parsed[0] if isinstance(parsed, list) else parsed
                # Deduplicate deltas: only yield if arguments changed
                if self.last_yielded_call:
                    if call["function"]["name"] == self.last_yielded_call["function"]["name"] and \
                       call["function"]["arguments"] == self.last_yielded_call["function"]["arguments"]:
                        return None
                self.last_yielded_call = call
                call["is_partial"] = True
                return call
        except:
            pass
        return None

    def _is_duplicate(self, call: Dict) -> bool:
        """Checks if a tool call is an unintentional repetition."""
        name = call["function"]["name"]
        args_hash = hash(json.dumps(call["function"]["arguments"], sort_keys=True))
        # Check if it matches the last call exactly
        if self.call_history and self.call_history[-1] == (name, args_hash):
            return True
        return False

    def _record_call(self, call: Dict):
        """Records a completed tool call in history."""
        name = call["function"]["name"]
        args_hash = hash(json.dumps(call["function"]["arguments"], sort_keys=True))
        self.call_history.append((name, args_hash))
        if len(self.call_history) > self.max_history:
            self.call_history.pop(0)

    def _reset_call_state(self):
        self.in_call = False
        self.call_buffer = ""
        self.current_pattern = None
        self.brace_count = 0
        self.bracket_count = 0
        self.in_string = False
        self.string_char = None
        self.is_escaped = False
        self.dynamic_tag_name = None
        self.last_yielded_call = None
        self._call_start_tag = ""

    def _parse_mistral(self, call_str: str) -> Optional[Union[Dict, List]]:
        """
        Parses Mistral's [TOOL_CALLS] format:
        [TOOL_CALLS] [{"name": "func1", "arguments": {...}}, {"name": "func2", "arguments": {...}}]
        """
        try:
            call_str = call_str.strip()
            # The call_str will be just the JSON array part (everything after [TOOL_CALLS])
            cleaned = self._clean_json(call_str)
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [self._normalize_json_call(item) for item in data if item]
            elif isinstance(data, dict):
                return self._normalize_json_call(data)
            return None
        except:
            return None

    def _parse_llama_pythonic(self, call_str: str) -> Optional[List[Dict]]:
        """
        Parses Llama 3.2/3.3/4 pythonic format:
        [get_weather(city='San Francisco', metric='celsius'), get_user_info(user_id=7890)]
        The call_str includes the matched opening bracket and function name.
        """
        try:
            call_str = call_str.strip()
            # Extract all function calls using regex
            # Pattern matches: function_name(arg1=value1, arg2=value2, ...)
            pattern = r'(\w+)\s*\(([^)]*)\)'
            tool_calls = []
            for match in re.finditer(pattern, call_str):
                name = match.group(1)
                args_str = match.group(2).strip()
                # Parse arguments
                arguments = {}
                if args_str:
                    # Split by comma, but respect quotes
                    arg_parts = []
                    current = []
                    in_quote = False
                    quote_char = None
                    for char in args_str + ',':
                        if char in ('"', "'") and (not current or current[-1] != '\\'):
                            if not in_quote:
                                in_quote = True
                                quote_char = char
                            elif char == quote_char:
                                in_quote = False
                        elif char == ',' and not in_quote:
                            if current:
                                arg_parts.append(''.join(current).strip())
                                current = []
                            continue
                        current.append(char)
                    # Parse each argument
                    for part in arg_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes from value
                            if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                                value = value[1:-1]
                            # Try to convert to appropriate type
                            elif value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif value.lower() == 'none' or value.lower() == 'null':
                                value = None
                            elif value.isdigit():
                                value = int(value)
                            elif re.match(r'^-?\d+\.\d+$', value):
                                value = float(value)
                            arguments[key] = value
                tool_calls.append({
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                })
            return tool_calls if tool_calls else None
        except:
            return None

    def _parse_arg_key_value(self, call_str: str) -> Optional[Dict]:
        """
        Parses tool calls in the format:
        <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
        The function name is already extracted by the regex and stored in self.dynamic_tag_name
        """
        try:
            call_str = call_str.strip()
            # Function name is captured by the regex match
            name = self.dynamic_tag_name
            if not name:
                return None
            # Extract all arg_key and arg_value pairs
            arguments = {}
            keys = re.findall(r'<arg_key>([^<]*)</arg_key>', call_str)
            values = re.findall(r'<arg_value>([^<]*)</arg_value>', call_str)
            # Pair up keys and values
            if len(keys) != len(values):
                return None
            for key, value in zip(keys, values):
                arguments[key] = value
            return {"function": {"name": name, "arguments": arguments}}
        except:
            return None

    def _parse_json_block(self, call_str: str) -> Optional[Union[Dict, List]]:
        try:
            call_str = call_str.strip()
            call_str = re.sub(r"^<(?:tool_call|tools|json)>", "", call_str, flags=re.I)
            call_str = re.sub(r"</(?:tool_call|tools|json)>$", "", call_str, flags=re.I)
            call_str = re.sub(r"^```json", "", call_str, flags=re.I)
            call_str = re.sub(r"```$", "", call_str, flags=re.I)
            call_str = call_str.strip()
            if not call_str: return None
            if "<function=" in call_str:
                return self._parse_nested_function_tags(call_str)
            cleaned = self._clean_json(call_str)
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [self._normalize_json_call(item) for item in data if item]
            if isinstance(data, dict):
                if "tool_calls" in data:
                    return [self._normalize_json_call(item) for item in data["tool_calls"] if item]
                return self._normalize_json_call(data)
            return None
        except:
            return None

    def _parse_gemma(self, call_str: str) -> Optional[Dict]:
        try:
            call_str = re.sub(r"<start_function_call>", "", call_str, flags=re.I).strip()
            brace_idx = call_str.find("{")
            if brace_idx == -1: return None
            name_part = call_str[:brace_idx].strip()
            name = name_part[5:].strip() if name_part.lower().startswith("call:") else name_part
            args_str = self._clean_json(call_str[brace_idx:].strip())
            return {"function": {"name": name, "arguments": json.loads(args_str)}}
        except: return None

    def _parse_indexed_json(self, call_str: str) -> Optional[Dict]:
        try:
            # The call_str includes the matched pattern, extract the name and JSON
            match = re.match(r"(?:^|[\n\r])\s*(\w+)(?::\d+)?:\s*\{", call_str)
            if not match: return None
            name = match.group(1)
            # Find where the JSON actually starts
            json_start = call_str.find("{")
            args_str = self._clean_json(call_str[json_start:].strip())
            return {"function": {"name": name, "arguments": json.loads(args_str)}}
        except: return None

    def _parse_xml_parameters(self, call_str: str) -> Optional[Dict]:
        try:
            name = self.dynamic_tag_name
            params = {}
            for m in re.finditer(r'<parameter\s+name="([^"]+)">([^<]*)</parameter>', call_str, flags=re.I):
                params[m.group(1)] = m.group(2)
            if not params and not name: return None
            return {"function": {"name": name, "arguments": params}}
        except: return None

    def _parse_function_tag(self, call_str: str) -> Optional[Dict]:
        try:
            name = self.dynamic_tag_name
            json_str = re.sub(r"^<function=[\w\.]+>", "", call_str, flags=re.I).strip()
            args = json.loads(self._clean_json(json_str))
            return {"function": {"name": name, "arguments": args}}
        except: return None

    def _parse_section_based(self, call_str: str) -> List[Dict]:
        tool_calls = []
        pattern = r"<\|tool_call_begin\|>\s*([\w\.]+)(?::\d+)?\s*<\|tool_call_argument_begin\|>\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*<\|tool_call_end\|>"
        for match in re.finditer(pattern, call_str, flags=re.I):
            name = match.group(1)
            if name.startswith("functions."): name = name[10:]
            try:
                args = json.loads(self._clean_json(match.group(2)))
                tool_calls.append({"function": {"name": name, "arguments": args}})
            except: continue
        return tool_calls

    def _parse_function_calls_xml(self, call_str: str) -> List[Dict]:
        tool_calls = []
        for inv_match in re.finditer(r'<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>', call_str, flags=re.I):
            name, inner = inv_match.group(1), inv_match.group(2)
            params = {m.group(1): m.group(2) for m in re.finditer(r'<parameter\s+name="([^"]+)">([^<]*)</parameter>', inner, flags=re.I)}
            tool_calls.append({"function": {"name": name, "arguments": params}})
        return tool_calls

    def _parse_nested_function_tags(self, call_str: str) -> List[Dict]:
        tool_calls = []
        for match in re.finditer(r"<function=([\w\.]+)>([\s\S]*?)</function>", call_str, flags=re.I):
            name, json_str = match.group(1), match.group(2).strip()
            try:
                args = json.loads(self._clean_json(json_str))
                tool_calls.append({"function": {"name": name, "arguments": args}})
            except: continue
        return tool_calls

    def _normalize_json_call(self, data: dict) -> Optional[Dict]:
        if not isinstance(data, dict): return None
        if "function" in data:
            func = data["function"]
            name = func.get("name")
            args = func.get("arguments") or func.get("parameters") or {}
        else:
            name = data.get("name") or data.get("tool") or data.get("function_name")
            args = data.get("arguments") or data.get("parameters") or data.get("function_args") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except: pass
        if name:
            if name.startswith("functions."): name = name[10:]
            return {"function": {"name": name, "arguments": args}}
        return None

    def _clean_json(self, json_str: str) -> str:
        """Heals common LLM JSON errors including single quotes and unquoted keys."""
        # Remove trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        # Quote unquoted keys (including single quotes)
        json_str = re.sub(r"([{,]\s*)(['\"]?)([a-zA-Z_][a-zA-Z0-9_]*)\2\s*:", r'\1"\3":', json_str)
        # Convert single quoted values to double quotes (simple case)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        # Handle unescaped newlines in strings (replace with \n)
    
        def fix_newlines(m):
            return m.group(0).replace('\n', '\\n').replace('\r', '\\r')
        json_str = re.sub(r'"[^"]*"', fix_newlines, json_str)
        json_str = json_str.replace("<escape>", '"')
        return json_str

    def finalize(self) -> str:
        """Flushes remaining text into parsed_buffer and returns the complete accumulated text."""
        if self.in_call:
            assert isinstance(self.current_pattern, str)
            p_data = self.registry.get(self.current_pattern) or self.dynamic_patterns.get(self.current_pattern)
            assert p_data is not None
            end_type = p_data["end"]
            assert isinstance(end_type, str)
            temp_buffer = self.call_buffer
            if not self.in_string:
                if end_type == "counting" or end_type == "dynamic_xml_counting":
                    temp_buffer += "}" * self.brace_count
                    temp_buffer += "]" * self.bracket_count
                elif end_type == "dynamic_xml":
                    temp_buffer += f"</{self.dynamic_tag_name}>"
                elif isinstance(end_type, str) and end_type.startswith("</"):
                    temp_buffer += end_type
            # Try to parse - if not a valid tool call, add text back to parsed_buffer
            parsed_call = p_data["parser"](temp_buffer)
            if not parsed_call:
                self.parsed_buffer += self._call_start_tag + temp_buffer
            self.parsed_buffer += self.buffer
        else:
            self.parsed_buffer += self.buffer
        self.buffer = ""
        self._reset_call_state()
        return self.parsed_buffer

    def get_code_blocks(self) -> List[str]:
        """Returns all accumulated code blocks."""
        return self.code_blocks.copy()

    def get_think_blocks(self) -> List[str]:
        """Returns all accumulated think blocks."""
        return self.think_blocks.copy()

    def clear_think_blocks(self):
        """Clears the accumulated think blocks."""
        self.think_blocks = []

    def clear_code_blocks(self):
        """Clears the accumulated code blocks."""
        self.code_blocks = []