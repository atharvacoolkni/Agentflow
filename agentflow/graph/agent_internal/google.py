"""Google GenAI request helpers for Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from .constants import GOOGLE_THINKING_BUDGET_BY_EFFORT


logger = logging.getLogger("agentflow.agent")


class AgentGoogleMixin:
    """Google GenAI message conversion and request helpers."""

    def _convert_to_google_format(self, messages: list[dict[str, Any]]) -> tuple[str | None, list]:
        """Convert chat-completion style messages into Google GenAI content objects."""
        from google.genai import types  # noqa: PLC0415

        system_instruction = None
        google_contents: list[types.Content] = []
        call_id_to_name: dict[str, str] = {}

        for message in messages:
            for tool_call in message.get("tool_calls", []) or []:
                tool_call_id = tool_call.get("id", "")
                function_name = tool_call.get("function", {}).get("name", "")
                if tool_call_id and function_name:
                    call_id_to_name[tool_call_id] = function_name

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                if system_instruction is None:
                    system_instruction = str(content)
                else:
                    system_instruction += "\n" + str(content)
                continue

            if role == "assistant" and message.get("tool_calls"):
                parts: list[types.Part] = []

                # Handle multimodal content
                content_parts = self._convert_content_to_parts(content)
                parts.extend(content_parts)

                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    function_name = function.get("name", "")
                    try:
                        function_args = json.loads(function.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        function_args = {}
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=function_name,
                                args=function_args,
                            )
                        )
                    )

                google_contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                function_name = call_id_to_name.get(
                    tool_call_id,
                    message.get("name", "") or tool_call_id or "unknown_function",
                )
                google_contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=function_name,
                                response={"result": str(content) if content else ""},
                            )
                        ],
                    )
                )
                continue

            google_role = "model" if role == "assistant" else "user"
            parts = self._convert_content_to_parts(content)
            google_contents.append(
                types.Content(
                    role=google_role,
                    parts=parts,
                )
            )

        return system_instruction, google_contents

    def _convert_content_to_parts(self, content: Any) -> list:
        """
        Convert message content to Google types.Part objects.

        Handles both string content and list of ContentBlock objects for multimodal support.

        Args:
            content: Message content (str or list[ContentBlock])

        Returns:
            list: List of types.Part objects for Google API
        """
        from google.genai import types  # noqa: PLC0415

        from agentflow.state.message_block import (
            AudioBlock,
            ImageBlock,
            TextBlock,
            ToolCallBlock,
            VideoBlock,
        )

        # Handle string content (backwards compatibility)
        if isinstance(content, str):
            return [types.Part(text=content if content else "")]

        # Handle list of ContentBlock objects
        if not isinstance(content, list):
            return [types.Part(text=str(content) if content else "")]

        parts = []
        for block in content:
            # Text blocks
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append(types.Part(text=block.text))

            # Image blocks
            elif isinstance(block, ImageBlock):
                part = self._convert_image_block_to_part(block)
                if part:
                    parts.append(part)

            # Audio blocks
            elif isinstance(block, AudioBlock):
                part = self._convert_audio_block_to_part(block)
                if part:
                    parts.append(part)

            # Video blocks
            elif isinstance(block, VideoBlock):
                part = self._convert_video_block_to_part(block)
                if part:
                    parts.append(part)

            # ToolCallBlock - skip (handled separately in assistant+tool_calls branch)
            elif isinstance(block, ToolCallBlock):
                continue

            # Unknown block types - convert to text
            else:
                text = str(block)
                if text:
                    parts.append(types.Part(text=text))

        # Return at least one empty text part if no content
        return parts if parts else [types.Part(text="")]

    def _convert_image_block_to_part(self, block: Any) -> Any | None:
        """
        Convert ImageBlock to Google types.Part.

        Args:
            block: ImageBlock object containing media reference

        Returns:
            types.Part or None if conversion fails
        """
        from google.genai import types  # noqa: PLC0415
        import base64

        media = block.media
        mime_type = media.mime_type or "image/jpeg"

        # Handle base64 data
        if media.kind == "data" and media.data_base64:
            try:
                image_bytes = base64.b64decode(media.data_base64)
                return types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_bytes,
                    )
                )
            except Exception as e:
                logger.warning("Failed to decode base64 image: %s", e)
                return None

        # Handle URLs (file URIs or http(s) URLs)
        if media.kind == "url" and media.url:
            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=media.url,
                )
            )

        # Handle file_id (Google File API uploads)
        if media.kind == "file_id" and media.file_id:
            # Google expects file URIs in format: "gs://..." or file API URIs
            file_uri = media.file_id
            if not file_uri.startswith(("gs://", "https://generativelanguage.googleapis.com")):
                file_uri = f"https://generativelanguage.googleapis.com/v1beta/files/{media.file_id}"

            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=file_uri,
                )
            )

        logger.warning("ImageBlock has no valid media reference: %s", media)
        return None

    def _convert_audio_block_to_part(self, block: Any) -> Any | None:
        """
        Convert AudioBlock to Google types.Part.

        Args:
            block: AudioBlock object containing media reference

        Returns:
            types.Part or None if conversion fails
        """
        from google.genai import types  # noqa: PLC0415
        import base64

        media = block.media
        mime_type = media.mime_type or "audio/wav"

        # Handle base64 data
        if media.kind == "data" and media.data_base64:
            try:
                audio_bytes = base64.b64decode(media.data_base64)
                return types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=audio_bytes,
                    )
                )
            except Exception as e:
                logger.warning("Failed to decode base64 audio: %s", e)
                return None

        # Handle URLs (file URIs or http(s) URLs)
        if media.kind == "url" and media.url:
            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=media.url,
                )
            )

        # Handle file_id (Google File API uploads)
        if media.kind == "file_id" and media.file_id:
            file_uri = media.file_id
            if not file_uri.startswith(("gs://", "https://generativelanguage.googleapis.com")):
                file_uri = f"https://generativelanguage.googleapis.com/v1beta/files/{media.file_id}"

            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=file_uri,
                )
            )

        logger.warning("AudioBlock has no valid media reference: %s", media)
        return None

    def _convert_video_block_to_part(self, block: Any) -> Any | None:
        """
        Convert VideoBlock to Google types.Part.

        Args:
            block: VideoBlock object containing media reference

        Returns:
            types.Part or None if conversion fails
        """
        from google.genai import types  # noqa: PLC0415
        import base64

        media = block.media
        mime_type = media.mime_type or "video/mp4"

        # Handle base64 data
        if media.kind == "data" and media.data_base64:
            try:
                video_bytes = base64.b64decode(media.data_base64)
                return types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=video_bytes,
                    )
                )
            except Exception as e:
                logger.warning("Failed to decode base64 video: %s", e)
                return None

        # Handle URLs (file URIs or http(s) URLs)
        if media.kind == "url" and media.url:
            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=media.url,
                )
            )

        # Handle file_id (Google File API uploads)
        if media.kind == "file_id" and media.file_id:
            file_uri = media.file_id
            if not file_uri.startswith(("gs://", "https://generativelanguage.googleapis.com")):
                file_uri = f"https://generativelanguage.googleapis.com/v1beta/files/{media.file_id}"

            return types.Part(
                file_data=types.FileData(
                    mime_type=mime_type,
                    file_uri=file_uri,
                )
            )

        logger.warning("VideoBlock has no valid media reference: %s", media)
        return None

    def _convert_tools_to_google_format(self, tools: list) -> list:
        """Convert OpenAI-style tool definitions into Google FunctionDeclarations."""
        from google.genai import types  # noqa: PLC0415

        google_tools = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                function = tool["function"]
                function_decl_kwargs = {
                    "name": function["name"],
                    "description": function.get("description", ""),
                }
                if "parameters" in function:
                    function_decl_kwargs["parameters_json_schema"] = function["parameters"]
                google_tools.append(types.FunctionDeclaration(**function_decl_kwargs))
        return google_tools

    def _build_google_config(
        self,
        system_instruction: str | None,
        tools: list | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        """Build a Google GenerateContentConfig instance."""
        from google.genai import types  # noqa: PLC0415

        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if "temperature" in call_kwargs:
            config_kwargs["temperature"] = call_kwargs.pop("temperature")
        if "max_tokens" in call_kwargs or "max_output_tokens" in call_kwargs:
            config_kwargs["max_output_tokens"] = call_kwargs.pop(
                "max_tokens",
                call_kwargs.pop("max_output_tokens", None),
            )

        if tools and self.output_type == "text":
            function_declarations = self._convert_tools_to_google_format(tools)
            if function_declarations:
                config_kwargs["tools"] = [types.Tool(function_declarations=function_declarations)]

        if self.reasoning_config and self.output_type == "text":
            thinking_kwargs: dict[str, Any] = {"include_thoughts": True}
            budget = self.reasoning_config.get("thinking_budget")
            effort = self.reasoning_config.get("effort")
            if budget is not None:
                thinking_kwargs["thinking_budget"] = int(budget)
            elif effort and effort in GOOGLE_THINKING_BUDGET_BY_EFFORT:
                thinking_kwargs["thinking_budget"] = GOOGLE_THINKING_BUDGET_BY_EFFORT[effort]
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    async def _call_google(
        self,
        messages: list[dict[str, Any]],
        tools: list | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call Google GenAI text, image, video, or audio endpoints."""
        call_kwargs = {**self.llm_kwargs, **kwargs}

        system_instruction, google_contents = self._convert_to_google_format(messages)
        config = self._build_google_config(system_instruction, tools, call_kwargs)

        if self.output_type == "text":
            if stream:
                logger.debug(
                    "Calling Google aio.models.generate_content_stream with model=%s",
                    self.model,
                )
                return await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=google_contents,
                    config=config,
                )

            logger.debug("Calling Google aio.models.generate_content with model=%s", self.model)
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=google_contents,
                config=config,
            )

        if self.output_type == "image":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_images with model=%s", self.model)
            return await self.client.aio.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "video":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_videos with model=%s", self.model)
            return await self.client.aio.models.generate_videos(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        if self.output_type == "audio":
            prompt = self._extract_prompt(messages)
            logger.debug("Calling Google aio.models.generate_audio with model=%s", self.model)
            return await self.client.aio.models.generate_audio(
                model=self.model,
                prompt=prompt,
                config=config,
            )

        raise ValueError(f"Unsupported output_type '{self.output_type}' for Google provider")
