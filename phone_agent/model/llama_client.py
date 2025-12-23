import time
from typing import Any

import requests
import sseclient  # pip install sseclient-py
from .client import ModelConfig, ModelResponse, get_message
import json


class LlamaModelClient:
    """
    Client for interacting with llama-server (OpenAI-compatible chat API).

    This client is interface-compatible with ModelClient.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
        }

        # llama-server usually ignores api_key, but keep it for compatibility
        if self.config.api_key:
            self.headers["Authorization"] = f"Bearer {self.config.api_key}"

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        start_time = time.time()
        time_to_first_token = None
        time_to_thinking_end = None

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "stream": True,
        }

        # llama-server does not support extra_body; ignore safely
        url = f"{self.base_url}/v1/chat/completions"
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=None,
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)

        raw_content = ""
        buffer = ""

        action_markers = ["finish(message=", "do(action="]
        in_action_phase = False
        first_token_received = False

        for event in client.events():
            if event.data == "[DONE]":
                break

            try:
                data = json.loads(event.data)
            except Exception:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # llama-server implementations differ here
            content = delta.get("content") or delta.get("text") or ""

            if not content:
                continue

            raw_content += content

            if not first_token_received:
                time_to_first_token = time.time() - start_time
                first_token_received = True

            if in_action_phase:
                continue

            buffer += content

            marker_found = False
            for marker in action_markers:
                if marker in buffer:
                    thinking_part = buffer.split(marker, 1)[0]
                    print(thinking_part, end="", flush=True)
                    print()

                    in_action_phase = True
                    marker_found = True

                    if time_to_thinking_end is None:
                        time_to_thinking_end = time.time() - start_time
                    break

            if marker_found:
                continue

            # detect partial marker
            is_potential_marker = False
            for marker in action_markers:
                for i in range(1, len(marker)):
                    if buffer.endswith(marker[:i]):
                        is_potential_marker = True
                        break
                if is_potential_marker:
                    break

            if not is_potential_marker:
                print(buffer, end="", flush=True)
                buffer = ""

        total_time = time.time() - start_time

        thinking, action = self._parse_response(raw_content)

        lang = self.config.lang
        print()
        print("=" * 50)
        print(f"{get_message('performance_metrics', lang)}:")
        print("-" * 50)
        if time_to_first_token is not None:
            print(
                f"{get_message('time_to_first_token', lang)}: {time_to_first_token:.3f}s"
            )
        if time_to_thinking_end is not None:
            print(
                f"{get_message('time_to_thinking_end', lang)}:        {time_to_thinking_end:.3f}s"
            )
        print(
            f"{get_message('total_inference_time', lang)}:          {total_time:.3f}s"
        )
        print("=" * 50)

        return ModelResponse(
            thinking=thinking,
            action=action,
            raw_content=raw_content,
            time_to_first_token=time_to_first_token,
            time_to_thinking_end=time_to_thinking_end,
            total_time=total_time,
        )

    def _parse_response(self, content: str) -> tuple[str, str]:
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            return parts[0].strip(), "finish(message=" + parts[1]

        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            return parts[0].strip(), "do(action=" + parts[1]

        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action

        return "", content
