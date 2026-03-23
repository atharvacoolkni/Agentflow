"""Tests for Google GenAI converter functionality."""

import json
from datetime import datetime
from unittest.mock import Mock

import pytest

from agentflow.adapters.llm.google_genai_converter import GoogleGenAIConverter
from agentflow.state.message import Message
from agentflow.state.message_block import ReasoningBlock, TextBlock, ToolCallBlock


class MockPart:
    """Mock Part for testing."""

    def __init__(
        self,
        text=None,
        thought=None,
        function_call=None,
        inline_data=None,
        file_data=None,
    ):
        self.text = text
        self.thought = thought
        self.function_call = function_call
        self.inline_data = inline_data
        self.file_data = file_data


class MockFunctionCall:
    """Mock FunctionCall for testing."""

    def __init__(self, name, args=None):
        self.name = name
        self.args = args or {}


class MockInlineData:
    """Mock InlineData for testing."""

    def __init__(self, data, mime_type):
        self.data = data
        self.mime_type = mime_type


class MockFileData:
    """Mock FileData for testing."""

    def __init__(self, file_uri, mime_type):
        self.file_uri = file_uri
        self.mime_type = mime_type


class MockContent:
    """Mock Content for testing."""

    def __init__(self, parts=None):
        self.parts = parts or []


class MockCandidate:
    """Mock Candidate for testing."""

    def __init__(self, content=None, finish_reason="STOP"):
        self.content = content
        self.finish_reason = finish_reason


class MockUsageMetadata:
    """Mock UsageMetadata for testing."""

    def __init__(
        self, candidates_token_count=0, prompt_token_count=0, total_token_count=0, thoughts_token_count=0
    ):
        self.candidates_token_count = candidates_token_count
        self.prompt_token_count = prompt_token_count
        self.total_token_count = total_token_count
        self.cached_content_token_count = 0
        self.thoughts_token_count = thoughts_token_count


class MockGenerateContentResponse:
    """Mock GenerateContentResponse for testing."""

    def __init__(
        self,
        candidates=None,
        usage_metadata=None,
        model_version="gemini-2.0-flash",
        response_id=None,
        create_time=None,
    ):
        self.candidates = candidates or []
        self.usage_metadata = usage_metadata
        self.model_version = model_version
        self.response_id = response_id
        self.create_time = create_time


class TestGoogleGenAIConverter:
    """Test class for Google GenAI converter."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance for testing."""
        return GoogleGenAIConverter()

    @pytest.mark.asyncio
    async def test_convert_simple_text_response(self, converter):
        """Test converting a simple text response."""
        # Create mock response
        text_part = MockPart(text="Hello, world!")
        content = MockContent(parts=[text_part])
        candidate = MockCandidate(content=content)
        usage = MockUsageMetadata(
            candidates_token_count=5, prompt_token_count=3, total_token_count=8
        )
        response = MockGenerateContentResponse(
            candidates=[candidate], usage_metadata=usage
        )

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert isinstance(message, Message)
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello, world!"
        assert message.usages.completion_tokens == 5
        assert message.usages.prompt_tokens == 3
        assert message.usages.total_tokens == 8

    @pytest.mark.asyncio
    async def test_convert_response_with_reasoning(self, converter):
        """Test converting a response with reasoning/thoughts."""
        # Create mock response with text and thought
        # Google GenAI uses thought=True as a boolean flag; reasoning text is in part.text
        text_part = MockPart(text="The answer is 42")
        thought_part = MockPart(text="I need to think deeply about this question", thought=True)
        content = MockContent(parts=[text_part, thought_part])
        candidate = MockCandidate(content=content)
        usage = MockUsageMetadata(
            candidates_token_count=5,
            prompt_token_count=3,
            total_token_count=8,
            thoughts_token_count=12
        )
        response = MockGenerateContentResponse(candidates=[candidate], usage_metadata=usage)

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ReasoningBlock)
        assert message.content[1].summary == "I need to think deeply about this question"
        assert message.reasoning == "I need to think deeply about this question"
        # Verify token usage including reasoning tokens
        assert message.usages.completion_tokens == 5
        assert message.usages.prompt_tokens == 3
        assert message.usages.total_tokens == 8
        assert message.usages.reasoning_tokens == 12

    @pytest.mark.asyncio
    async def test_convert_response_with_function_call(self, converter):
        """Test converting a response with function calls."""
        # Create mock function call
        func_call = MockFunctionCall(
            name="get_weather", args={"location": "San Francisco"}
        )
        func_part = MockPart(function_call=func_call)
        content = MockContent(parts=[func_part])
        candidate = MockCandidate(content=content)
        response = MockGenerateContentResponse(candidates=[candidate])

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolCallBlock)
        assert message.content[0].name == "get_weather"
        assert message.content[0].args == {"location": "San Francisco"}
        assert message.tools_calls is not None
        assert len(message.tools_calls) == 1
        assert message.tools_calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_convert_response_with_inline_image(self, converter):
        """Test converting a response with inline image data."""
        # Create mock inline data for image
        inline_data = MockInlineData(data="base64_image_data", mime_type="image/jpeg")
        image_part = MockPart(inline_data=inline_data)
        content = MockContent(parts=[image_part])
        candidate = MockCandidate(content=content)
        response = MockGenerateContentResponse(candidates=[candidate])

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert len(message.content) == 1
        from agentflow.state.message_block import ImageBlock

        assert isinstance(message.content[0], ImageBlock)
        assert message.content[0].media.data_base64 == "base64_image_data"
        assert message.content[0].media.mime_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_convert_response_with_file_uri(self, converter):
        """Test converting a response with file URI."""
        # Create mock file data
        file_data = MockFileData(
            file_uri="gs://bucket/video.mp4", mime_type="video/mp4"
        )
        video_part = MockPart(file_data=file_data)
        content = MockContent(parts=[video_part])
        candidate = MockCandidate(content=content)
        response = MockGenerateContentResponse(candidates=[candidate])

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert len(message.content) == 1
        from agentflow.state.message_block import VideoBlock

        assert isinstance(message.content[0], VideoBlock)
        assert message.content[0].media.url == "gs://bucket/video.mp4"
        assert message.content[0].media.mime_type == "video/mp4"

    @pytest.mark.asyncio
    async def test_convert_empty_response(self, converter):
        """Test converting an empty response (no candidates)."""
        # Create mock response with no candidates
        response = MockGenerateContentResponse(candidates=[])

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert isinstance(message, Message)
        assert message.role == "assistant"
        assert len(message.content) == 0
        assert message.metadata["provider"] == "google_genai"

    @pytest.mark.asyncio
    async def test_convert_response_with_multiple_parts(self, converter):
        """Test converting a response with multiple parts."""
        # Create mock response with text, thought, and function call
        # Google GenAI uses thought=True as a boolean flag; reasoning text is in part.text
        text_part = MockPart(text="Here's what I found:")
        thought_part = MockPart(text="Analyzing the request", thought=True)
        func_call = MockFunctionCall(name="search", args={"query": "python"})
        func_part = MockPart(function_call=func_call)

        content = MockContent(parts=[text_part, thought_part, func_part])
        candidate = MockCandidate(content=content)
        usage = MockUsageMetadata(
            candidates_token_count=20, prompt_token_count=10, total_token_count=30
        )
        response = MockGenerateContentResponse(
            candidates=[candidate], usage_metadata=usage
        )

        # Convert response
        message = await converter.convert_response(response)

        # Assertions
        assert len(message.content) == 3
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ReasoningBlock)
        assert isinstance(message.content[2], ToolCallBlock)
        assert message.usages.total_tokens == 30

    @pytest.mark.asyncio
    async def test_streaming_response_conversion(self, converter):
        """Test converting a streaming response."""

        # Create mock streaming chunks
        class MockStreamingResponse:
            def __init__(self):
                self.chunks = [
                    MockGenerateContentResponse(
                        candidates=[
                            MockCandidate(
                                content=MockContent(parts=[MockPart(text="Hello")])
                            )
                        ]
                    ),
                    MockGenerateContentResponse(
                        candidates=[
                            MockCandidate(
                                content=MockContent(parts=[MockPart(text=" world")])
                            )
                        ]
                    ),
                ]
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        stream = MockStreamingResponse()
        config = {"thread_id": "test-thread"}

        # Convert streaming response
        messages = []
        async for message in converter.convert_streaming_response(
            config=config, node_name="test_node", response=stream
        ):
            messages.append(message)

        # Assertions
        assert len(messages) > 0
        # The last message should be the final (non-delta) message
        final_message = messages[-1]
        assert not final_message.delta
        assert final_message.metadata["thread_id"] == "test-thread"

    @pytest.mark.asyncio
    async def test_convert_response_with_none(self, converter):
        """Test that converter handles None response gracefully."""
        # Try to convert a None response - should raise AttributeError
        with pytest.raises(AttributeError):
            await converter.convert_response(None)

    @pytest.mark.asyncio
    async def test_streaming_with_none(self, converter):
        """Test that streaming converter handles None response gracefully."""
        # Create an async generator that yields nothing
        config = {"thread_id": "test-thread"}
        
        # Try to convert None streaming response
        messages = []
        async for message in converter.convert_streaming_response(
            config=config, node_name="test_node", response=None
        ):
            messages.append(message)

        # Should yield one empty message (error handling behavior)
        assert len(messages) == 1
        assert len(messages[0].content) == 0


class TestGoogleGenAIInputConversion:
    """Test class for Google GenAI input message conversion (Agentflow → Google)."""

    @pytest.fixture
    def mixin(self):
        """Create an AgentGoogleMixin instance for testing."""
        from agentflow.graph.agent_internal.google import AgentGoogleMixin

        return AgentGoogleMixin()

    def test_convert_to_google_format_with_image_url(self, mixin):
        """Test conversion of message with image URL."""
        from agentflow.state.message_block import TextBlock, ImageBlock, MediaRef

        messages = [
            {
                "role": "user",
                "content": [
                    TextBlock(text="What do you see in this image?"),
                    ImageBlock(
                        media=MediaRef(
                            kind="url",
                            url="https://example.com/image.jpg",
                            mime_type="image/jpeg",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert google_contents[0].role == "user"
        assert len(google_contents[0].parts) == 2
        # First part should be text
        assert google_contents[0].parts[0].text == "What do you see in this image?"
        # Second part should be file_data
        assert hasattr(google_contents[0].parts[1], "file_data")
        assert google_contents[0].parts[1].file_data.file_uri == "https://example.com/image.jpg"
        assert google_contents[0].parts[1].file_data.mime_type == "image/jpeg"

    def test_convert_to_google_format_with_image_base64(self, mixin):
        """Test conversion of message with base64 image."""
        from agentflow.state.message_block import TextBlock, ImageBlock, MediaRef
        import base64

        # Create a small test image data
        test_data = b"test_image_data"
        base64_data = base64.b64encode(test_data).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    TextBlock(text="Analyze this image"),
                    ImageBlock(
                        media=MediaRef(
                            kind="data",
                            data_base64=base64_data,
                            mime_type="image/png",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert len(google_contents[0].parts) == 2
        # Second part should be inline_data
        assert hasattr(google_contents[0].parts[1], "inline_data")
        assert google_contents[0].parts[1].inline_data.data == test_data
        assert google_contents[0].parts[1].inline_data.mime_type == "image/png"

    def test_convert_to_google_format_with_image_file_id(self, mixin):
        """Test conversion of message with file_id image."""
        from agentflow.state.message_block import ImageBlock, MediaRef

        messages = [
            {
                "role": "user",
                "content": [
                    ImageBlock(
                        media=MediaRef(
                            kind="file_id",
                            file_id="abc123",
                            mime_type="image/jpeg",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert len(google_contents[0].parts) == 1
        # Should be file_data with formatted URI
        assert hasattr(google_contents[0].parts[0], "file_data")
        assert "abc123" in google_contents[0].parts[0].file_data.file_uri

    def test_convert_to_google_format_mixed_content(self, mixin):
        """Test conversion of message with text + multiple images."""
        from agentflow.state.message_block import TextBlock, ImageBlock, MediaRef

        messages = [
            {
                "role": "user",
                "content": [
                    TextBlock(text="Compare these two images:"),
                    ImageBlock(
                        media=MediaRef(
                            kind="url",
                            url="https://example.com/image1.jpg",
                            mime_type="image/jpeg",
                        )
                    ),
                    ImageBlock(
                        media=MediaRef(
                            kind="url",
                            url="https://example.com/image2.jpg",
                            mime_type="image/jpeg",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert len(google_contents[0].parts) == 3
        assert google_contents[0].parts[0].text == "Compare these two images:"
        assert hasattr(google_contents[0].parts[1], "file_data")
        assert hasattr(google_contents[0].parts[2], "file_data")

    def test_convert_to_google_format_with_audio(self, mixin):
        """Test conversion of message with audio."""
        from agentflow.state.message_block import AudioBlock, MediaRef

        messages = [
            {
                "role": "user",
                "content": [
                    AudioBlock(
                        media=MediaRef(
                            kind="url",
                            url="https://example.com/audio.mp3",
                            mime_type="audio/mp3",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert len(google_contents[0].parts) == 1
        assert hasattr(google_contents[0].parts[0], "file_data")
        assert google_contents[0].parts[0].file_data.mime_type == "audio/mp3"

    def test_convert_to_google_format_with_video(self, mixin):
        """Test conversion of message with video."""
        from agentflow.state.message_block import VideoBlock, MediaRef

        messages = [
            {
                "role": "user",
                "content": [
                    VideoBlock(
                        media=MediaRef(
                            kind="url",
                            url="https://example.com/video.mp4",
                            mime_type="video/mp4",
                        )
                    ),
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert len(google_contents[0].parts) == 1
        assert hasattr(google_contents[0].parts[0], "file_data")
        assert google_contents[0].parts[0].file_data.mime_type == "video/mp4"

    def test_convert_to_google_format_backwards_compat(self, mixin):
        """Test that text-only messages still work (backwards compatibility)."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 2
        assert google_contents[0].role == "user"
        assert google_contents[1].role == "model"
        assert google_contents[0].parts[0].text == "Hello, how are you?"
        assert google_contents[1].parts[0].text == "I'm doing well, thank you!"

    def test_convert_image_block_invalid_base64(self, mixin):
        """Test handling of invalid base64 data."""
        from agentflow.state.message_block import ImageBlock, MediaRef

        block = ImageBlock(
            media=MediaRef(
                kind="data",
                data_base64="invalid!!!base64",
                mime_type="image/jpeg",
            )
        )

        # Should return None and log warning
        result = mixin._convert_image_block_to_part(block)
        assert result is None

    def test_convert_to_google_format_assistant_with_tool_calls_and_content(self, mixin):
        """Test conversion of assistant message with both content and tool calls."""
        from agentflow.state.message_block import TextBlock

        messages = [
            {
                "role": "assistant",
                "content": [TextBlock(text="Let me search for that.")],
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "python"}),
                        },
                    }
                ],
            }
        ]

        system_instruction, google_contents = mixin._convert_to_google_format(messages)

        # Assertions
        assert len(google_contents) == 1
        assert google_contents[0].role == "model"
        assert len(google_contents[0].parts) == 2
        # First part should be text
        assert google_contents[0].parts[0].text == "Let me search for that."
        # Second part should be function_call
        assert hasattr(google_contents[0].parts[1], "function_call")
        assert google_contents[0].parts[1].function_call.name == "search"
