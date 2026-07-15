# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class HYV3ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_sep_token: str = "<tool_sep>"

        self.arg_key_start_token: str = "<arg_key>"
        self.arg_key_end_token: str = "</arg_key>"

        self.arg_value_start_token: str = "<arg_value>"
        self.arg_value_end_token: str = "</arg_value>"

        self.tool_call_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}"
            rf"(.*?){self.tool_call_end_token}",
            re.DOTALL,
        )

        self.tool_call_v3_regex = re.compile(
            rf"{re.escape(self.tool_call_start_token)}(\w+)\s*```json\s*(.*?)\s*```\s*{re.escape(self.tool_call_end_token)}",
            re.DOTALL,
        )

        self.tool_call_portion_regex = re.compile(
            rf"(.*?){self.tool_sep_token}(.*?){self.tool_call_end_token}"
        )

        self.func_args_regex = re.compile(
            rf"{self.arg_key_start_token}(.*?){self.arg_key_end_token}\s*"
            rf"{self.arg_value_start_token}(.*?){self.arg_value_end_token}",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "HYV3 Tool parser could not locate tool call "
                "start/end tokens in the tokenizer!"
            )

    def _extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> list[ToolCall]:
        def _is_string_type(
            function_name: str,
            arg_key: str,
            tools: list[ChatCompletionToolsParam] | None,
        ) -> bool:
            if tools is None:
                return False
            for tool in tools:
                if tool.function.name == function_name:
                    if tool.function.parameters is None:
                        return False
                    arg_type = (
                        tool.function.parameters.get("properties", {})
                        .get(arg_key, {})
                        .get("type", None)
                    )
                    return arg_type == "string"
            logger.warning("No tool named '%s'.", function_name)
            return False

        def _deserialize(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass

            return value

        try:
            function_call_tuples = []
            # start_token{name}sep_token{args}end_token...
            function_calls = self.tool_call_regex.findall(model_output)
            if function_calls:
                function_call_tuples.extend(function_calls)
            else:
                # Try V3 format: <tool_call>name\n```json\n{...}\n```</tool_call>
                function_calls_v3 = self.tool_call_v3_regex.findall(model_output)
                function_call_tuples.extend(function_calls_v3)

            tool_calls = []
            for match in function_call_tuples:
                function_name, function_args = match
                function_name = function_name.strip()
                function_args = function_args.strip()

                # Try to parse as JSON first (V3 format)
                try:
                    arg_dict = json.loads(function_args)
                except json.JSONDecodeError:
                    # Fall back to V2.1 format with arg_key/arg_value tags
                    arg_pairs = self.func_args_regex.findall(function_args)
                    arg_dict = {}
                    for key, value in arg_pairs:
                        if not _is_string_type(function_name, key, request.tools):
                            parsed_value = _deserialize(value)
                        else:
                            parsed_value = value
                        logger.debug("arguments: key = %s, value = %s", key, value)
                        arg_dict[key] = parsed_value
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(arg_dict, ensure_ascii=False),
                        ),
                    )
                )
            return tool_calls
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            try:
                tool_calls = self._extract_tool_calls(model_output, request)

                s_index = model_output.find(self.tool_calls_start_token)
                content = model_output[:s_index] if s_index != -1 else model_output
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        # check to see if we should be streaming a tool call - is there a
        if self.tool_calls_start_token_id not in current_token_ids:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)

        if self.tool_calls_start_token in delta_text:
            text_parts = delta_text.split(self.tool_calls_start_token)
            self._buffer += text_parts[-1]
            if text_parts[0]:
                return DeltaMessage(content=text_parts[0])
            return None

        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            return None
        logger.debug("cur_text = %s", cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            extracted_tool_calls = self._extract_tool_calls(cur_text, request)

            if len(extracted_tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None

            tool_calls = []
            for tool_call in extracted_tool_calls:
                self.current_tool_id += 1
                self.prev_tool_call_arr.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                )
                self.streamed_args_for_tool.append(tool_call.function.arguments)
                tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                )

            delta = DeltaMessage(tool_calls=tool_calls)
            self._buffer = cur_text.split(self.tool_call_end_token)[-1]
            return delta

        self._buffer = cur_text[start_idx:]
        return None
