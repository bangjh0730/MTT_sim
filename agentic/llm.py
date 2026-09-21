import json
import os
import time
from google import genai
from google.genai import types

# Default model when a client doesn't specify one. Individual agents pick a model
# suited to their reasoning load: the PA does multi-factor spatial/energy reasoning
# over many UAV-target pairs and needs a stronger model; the MA does a cheap,
# frequent yes/no coverage check and can use a lighter/faster one.
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class LLMClient:
    """Google Gemini client using the native Google GenAI SDK."""

    def __init__(self, log_path: str = "llm_log.jsonl", model: str = DEFAULT_MODEL):
        if not os.environ.get("GEMINI_API_KEY"):
            raise EnvironmentError("Set the GEMINI_API_KEY environment variable.")

        # Initialize the official unified client
        self._client = genai.Client()
        self._log_path = log_path
        self._model = model
        self.latencies: list[float] = []   # wall-clock seconds per generate() call
        self.slots:     list[int]   = []   # sim slot of each generate() call, same order

        # Initialize/clear the log file
        with open(self._log_path, "w", encoding="utf-8") as f:
            pass

    def generate(self, system: str, user: str, max_tokens: int = 1024, agent: str = "unknown", slot: int = 0) -> str:
        # Configuration mapping for system instructions and generation limits
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        # Call the native API endpoint, timing how long the reasoning call itself takes.
        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=config
        )
        latency_s = time.perf_counter() - start
        self.latencies.append(latency_s)
        self.slots.append(slot)

        answer = response.text.strip() if response.text else ""

        # Log the transaction
        log_entry = {
            "slot": slot,
            "agent": agent,
            "latency_s": latency_s,
            "input": user,
            "output": answer
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, indent=2, ensure_ascii=False) + "\n")

        return answer
