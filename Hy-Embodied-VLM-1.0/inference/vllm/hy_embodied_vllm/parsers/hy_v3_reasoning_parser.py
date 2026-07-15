# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class HYV3ReasoningParser(BaseThinkingReasoningParser):
    """
    HYV3 parser that delegates to either the standard <think>...</think>
    thinking parser or IdentityReasoningParser (no reasoning) based on the
    request-level `enable_thinking` kwarg.

    The HYV3 model uses <think>...</think> tokens to denote reasoning
    text when thinking is enabled. When the caller sets
    `chat_template_kwargs["enable_thinking"] = False`, the model is prompted
    with a closed `<think></think>` pair and produces the answer directly —
    in that case we swap in `IdentityReasoningParser` which does NOT try to
    extract reasoning content from the raw output.

    We use `enable_thinking` (Qwen3 convention) rather than
    `reasoning_effort` because vLLM prior to v0.22 has a top-level
    `request.reasoning_effort` field that silently clobbers
    `chat_template_kwargs["reasoning_effort"]` (fixed by
    vllm-project/vllm#43401).
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Read enable_thinking from chat_template_kwargs. Default: True
        # (chain-of-thought mode, same as the model's evaluation-time posture).
        chat_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        enable_thinking = chat_kwargs.pop("enable_thinking", None)
        if enable_thinking is None:
            enable_thinking = True
        logger.debug("enable_thinking for choosing parser: %s", enable_thinking)

        self._identity_parser: IdentityReasoningParser | None
        if not enable_thinking:
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)

        return super().is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning(model_output, request)

        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            ret is not None
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                # end token in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # end token in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no end token in previous or delta, reasoning content continues
                return DeltaMessage(reasoning=delta_text)

        return ret
