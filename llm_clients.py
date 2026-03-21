# llm_clients.py -- LLM API Wrappers for The Clankers
# Each client takes conversation history and returns a response string.
# Uniform interface: send(system_prompt, messages) -> str
#
# Includes per-provider rate limiting and automatic retry on 429 errors.

import json
import time
import threading
from collections import deque

import config

# ---------------------------------------------------------------------------
# Rate limiter -- shared per provider, thread-safe
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Sliding window rate limiter. Thread-safe.
    Tracks timestamps of recent calls and sleeps if limit is reached.
    """
    
    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self.timestamps = deque()
        self.lock = threading.Lock()
    
    def wait(self):
        """Block until a call slot is available."""
        with self.lock:
            now = time.time()
            # Purge timestamps outside the window
            while self.timestamps and self.timestamps[0] <= now - self.window:
                self.timestamps.popleft()
            
            if len(self.timestamps) >= self.max_calls:
                # Need to wait until the oldest call exits the window
                sleep_time = self.timestamps[0] + self.window - now + 0.1
                if sleep_time > 0:
                    print(f"  [RateLimit] Throttling for {sleep_time:.1f}s...")
                    # Release lock while sleeping so other threads aren't blocked
                    self.lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self.lock.acquire()
                    # Re-purge after sleep
                    now = time.time()
                    while self.timestamps and self.timestamps[0] <= now - self.window:
                        self.timestamps.popleft()
            
            self.timestamps.append(time.time())


# Per-provider rate limiters (RPM limits)
# Adjust these to match your API tier
_RATE_LIMITERS = {
    "google":    RateLimiter(max_calls=getattr(config, "GEMINI_RPM", 140), window_seconds=60),
    "anthropic": RateLimiter(max_calls=getattr(config, "CLAUDE_RPM", 200), window_seconds=60),
    "openai":    RateLimiter(max_calls=getattr(config, "CHATGPT_RPM", 200), window_seconds=60),
}

def get_rate_limiter(provider: str) -> RateLimiter:
    return _RATE_LIMITERS.get(provider)


# ---------------------------------------------------------------------------
# Retry logic for rate limit (429) errors
# ---------------------------------------------------------------------------

MAX_RETRIES_429 = 5
INITIAL_BACKOFF = 2.0  # seconds

def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit (429) error."""
    err_str = str(e).lower()
    if "429" in err_str or "rate" in err_str or "quota" in err_str:
        return True
    # Check for specific SDK exception types
    if hasattr(e, "status_code") and e.status_code == 429:
        return True
    if hasattr(e, "code") and e.code == 429:
        return True
    return False


# ---------------------------------------------------------------------------
# BASE CLASS
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """All LLM clients implement this interface."""
    
    def __init__(self, name, provider):
        self.name = name
        self.provider = provider
        self.limiter = get_rate_limiter(provider)
    
    def send(self, system_prompt, messages):
        raise NotImplementedError
    
    def _rate_limited_call(self, call_fn):
        """
        Execute an API call with rate limiting and 429 retry.
        
        Args:
            call_fn: Zero-arg callable that makes the actual API call
        
        Returns:
            The API response
        """
        # Wait for rate limit slot
        if self.limiter:
            self.limiter.wait()
        
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES_429 + 1):
            try:
                return call_fn()
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < MAX_RETRIES_429:
                    print(f"  [429] {self.name} rate limited. Retrying in {backoff:.1f}s... (attempt {attempt + 1}/{MAX_RETRIES_429})")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                    # Re-wait for a rate limit slot
                    if self.limiter:
                        self.limiter.wait()
                else:
                    raise


# ---------------------------------------------------------------------------
# ANTHROPIC (Claude)
# ---------------------------------------------------------------------------

class ClaudeClient(BaseLLMClient):
    def __init__(self):
        super().__init__("Claude", "anthropic")
        self.client = None
    
    def _ensure_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    def send(self, system_prompt, messages):
        self._ensure_client()
        
        api_messages = []
        for msg in messages:
            if msg["role"] == self.name:
                api_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": f"[{msg['role']}]: {msg['content']}"})
        
        api_messages = self._merge_consecutive(api_messages)
        
        def _call():
            return self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=16384,
                system=system_prompt,
                messages=api_messages,
            )
        
        response = self._rate_limited_call(_call)
        if not response.content or len(response.content) == 0:
            raise RuntimeError("Claude returned no content (empty response)")
        return response.content[0].text
    
    def _merge_consecutive(self, messages):
        if not messages:
            return messages
        
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)
        
        if merged and merged[0]["role"] == "assistant":
            merged.insert(0, {"role": "user", "content": "[System]: The session begins. You speak first."})

        # Solo / consecutive-turn fix: Anthropic API requires the last message to be user.
        # In single-LLM mode Claude speaks multiple times in a row; inject a bridge prompt.
        if merged and merged[-1]["role"] == "assistant":
            merged.append({"role": "user", "content": "[Continue: build on your previous thoughts. When the Music Sheet is complete, output [SESSION COMPLETE] and the full JSON in a ```json block.]"})

        return merged


# ---------------------------------------------------------------------------
# GOOGLE (Gemini)
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    def __init__(self):
        super().__init__("Gemini", "google")
        self.client = None
    
    def _ensure_client(self):
        if self.client is None:
            from google import genai
            from google.genai import types
            timeout_ms = getattr(config, "GEMINI_TIMEOUT_MS", 120_000)
            self.client = genai.Client(
                api_key=config.GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
    
    def send(self, system_prompt, messages):
        self._ensure_client()
        from google.genai import types
        
        api_messages = []
        for msg in messages:
            if msg["role"] == self.name:
                api_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
            else:
                api_messages.append({"role": "user", "parts": [{"text": f"[{msg['role']}]: {msg['content']}"}]})
        
        api_messages = self._merge_consecutive(api_messages)
        
        if api_messages and api_messages[0]["role"] == "model":
            api_messages.insert(0, {"role": "user", "parts": [{"text": "[System]: The session begins."}]})
        
        def _call():
            return self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=api_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
        
        response = self._rate_limited_call(_call)

        try:
            return response.text
        except ValueError:
            if response.candidates and response.candidates[0].finish_reason:
                reason = response.candidates[0].finish_reason
                raise RuntimeError(f"Gemini response blocked (finish_reason={reason})")
            raise RuntimeError("Gemini returned no text (response may have been filtered)")
    
    def _merge_consecutive(self, messages):
        if not messages:
            return messages
        
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                existing_text = merged[-1]["parts"][0]["text"]
                new_text = msg["parts"][0]["text"]
                merged[-1]["parts"][0]["text"] = existing_text + "\n\n" + new_text
            else:
                merged.append(msg)
        
        return merged


# ---------------------------------------------------------------------------
# OPENAI (ChatGPT)
# ---------------------------------------------------------------------------

class ChatGPTClient(BaseLLMClient):
    def __init__(self):
        super().__init__("ChatGPT", "openai")
        self.client = None

    def _ensure_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def send(self, system_prompt, messages):
        self._ensure_client()

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] == self.name:
                api_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                api_messages.append({"role": "user", "content": f"[{msg['role']}]: {msg['content']}"})

        api_messages = self._merge_consecutive(api_messages)
        if api_messages and api_messages[0]["role"] == "assistant":
            api_messages.insert(0, {"role": "user", "content": "[System]: The session begins. You speak first."})

        kwargs = {
            "model": config.CHATGPT_MODEL,
            "messages": api_messages,
            "max_completion_tokens": 16384,
        }
        
        def _call():
            return self.client.chat.completions.create(**kwargs)
        
        response = self._rate_limited_call(_call)
        if not response.choices:
            raise RuntimeError("ChatGPT returned no choices (empty or filtered response)")
        return response.choices[0].message.content

    def _merge_consecutive(self, messages):
        if not messages:
            return messages
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)
        return merged


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def get_client(provider):
    """
    Get an LLM client by provider name.

    Args:
        provider (str): "anthropic", "google", or "openai"

    Returns:
        BaseLLMClient instance
    """
    clients = {
        "anthropic": ClaudeClient,
        "google": GeminiClient,
        "openai": ChatGPTClient,
    }
    
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(clients.keys())}")
    
    return clients[provider]()
