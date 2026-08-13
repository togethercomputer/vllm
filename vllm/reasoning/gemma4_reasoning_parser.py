# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Role label that Gemma4 emits at the start of the thinking channel.
# The model generates: <|channel>thought\n...reasoning...<channel|>
# This prefix must be stripped to expose only the actual reasoning content.
_THOUGHT_PREFIX = "thought\n"


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning/thinking
    content within its output. Thinking mode is activated by passing
    ``enable_thinking=True`` in the chat template kwargs.

    Output pattern when thinking is enabled::

        <|channel>thought
        ...chain of thought reasoning...<channel|>
        Final answer text here.

    Streaming uses a marker-driven state machine instead of the base
    class's previous/delta token-id branch logic. The base logic breaks on
    real Gemma4 streams in two ways (both leak marker text into
    ``delta.content`` because ``adjust_request`` forces
    ``skip_special_tokens=False`` so marker text is present in
    ``delta_text``):

    1. A stray ``<channel|>`` with no opener hits the base "no start token"
       branch and is emitted verbatim as content (the single-special-token
       skip misses it whenever the detokenizer batches it with adjacent
       tokens).
    2. After the first thought block closes, the base
       "start-in-previous, end-in-previous" branch passes every subsequent
       delta to content verbatim - including the *second* thought block
       Gemma4 emits before tool calls.

    The state machine consumes marker text, routes text inside a channel to
    ``reasoning`` and outside to ``content``, strips the ``thought\\n`` role
    label at the start of every block (buffering across delta boundaries),
    and silently drops stray end markers. Multiple blocks per response and
    multiple markers per delta are handled.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Streaming state.
        self._in_thought: bool = False
        # Buffer for label stripping at the start of the current block;
        # None means the label for this block was already resolved.
        self._label_buf: str | None = None
        # Gemma4 sometimes emits a bare "thought\n" label with no channel
        # markers at a block boundary (google's response_schema models this
        # as an optional leading "(thought)?"). Buffer content at
        # boundaries (stream start / right after <channel|>) so the bare
        # label can be stripped; None = past the boundary.
        self._content_label_buf: str | None = ""
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]
        self._marker_re = re.compile(
            "({}|{})".format(re.escape(self.start_token), re.escape(self.end_token))
        )

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        """Disable special-token stripping to preserve boundary tokens."""
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<|channel>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        # Search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                # We're generating a tool call, so reasoning must be ended.
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                # We found a new turn or tool response token so don't consider
                # reasoning ended yet, since the model starts new reasoning
                # after these tokens.
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning, stripping the ``thought\\n`` role label.

        Handles multiple thought blocks (Gemma4 may emit a second block
        before a tool call): all in-channel text is concatenated into
        ``reasoning``, all out-of-channel text into ``content``.
        """
        if self.start_token not in model_output and self.end_token not in model_output:
            # Default to content history if no tags are present
            # (or if they were stripped)
            return None, model_output

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        in_thought = False
        at_boundary = True
        for segment in self._marker_re.split(model_output):
            if segment == self.start_token:
                in_thought = True
            elif segment == self.end_token:
                in_thought = False
                at_boundary = True
            elif segment:
                if in_thought:
                    reasoning_parts.append(_strip_thought_label(segment))
                else:
                    # Strip a bare "thought\n" label at block boundaries
                    # (output start / right after <channel|>).
                    if at_boundary:
                        segment = _strip_thought_label(segment)
                        at_boundary = False
                    if segment:
                        content_parts.append(segment)
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Marker-driven streaming split of ``delta_text``.

        ``skip_special_tokens=False`` (forced by ``adjust_request``)
        guarantees the channel markers appear literally in ``delta_text``,
        and each marker is a single token so its text never splits across
        deltas. The ``thought\\n`` role label is multi-token, so it is
        buffered until it can be confirmed or ruled out.
        """
        if not delta_text:
            return None

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for segment in self._marker_re.split(delta_text):
            if segment == self.start_token:
                self._in_thought = True
                self._label_buf = ""
            elif segment == self.end_token:
                # Flush an unresolved label buffer: the block ended before
                # the label diverged (e.g. reasoning text was exactly
                # "thought"), so it was real reasoning text after all.
                if self._label_buf:
                    reasoning_parts.append(self._label_buf)
                self._in_thought = False
                self._label_buf = None
                # New block boundary: re-arm bare-label stripping.
                self._content_label_buf = ""
            elif segment:
                if not self._in_thought:
                    if self._content_label_buf is None:
                        content_parts.append(segment)
                    else:
                        # At a block boundary: strip a bare "thought\n"
                        # label (buffered across deltas).
                        self._content_label_buf += segment
                        if self._content_label_buf.startswith(_THOUGHT_PREFIX):
                            remainder = self._content_label_buf[
                                len(_THOUGHT_PREFIX) :
                            ]
                            if remainder:
                                content_parts.append(remainder)
                            self._content_label_buf = None
                        elif not _THOUGHT_PREFIX.startswith(
                            self._content_label_buf
                        ):
                            content_parts.append(self._content_label_buf)
                            self._content_label_buf = None
                elif self._label_buf is None:
                    # Label for this block already resolved.
                    reasoning_parts.append(segment)
                else:
                    self._label_buf += segment
                    if self._label_buf.startswith(_THOUGHT_PREFIX):
                        # Label confirmed: emit whatever follows it.
                        remainder = self._label_buf[len(_THOUGHT_PREFIX) :]
                        if remainder:
                            reasoning_parts.append(remainder)
                        self._label_buf = None
                    elif not _THOUGHT_PREFIX.startswith(self._label_buf):
                        # Diverged: not a label, emit everything buffered.
                        reasoning_parts.append(self._label_buf)
                        self._label_buf = None
                    # else: still a strict prefix of the label - keep
                    # buffering across deltas.

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning=reasoning or None, content=content or None
        )


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\\n`` role label from the beginning of text.

    Mirrors ``vllm.reasoning.gemma4_utils._strip_thought_label`` from the
    offline parser.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
