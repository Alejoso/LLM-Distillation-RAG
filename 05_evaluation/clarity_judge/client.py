import json
import re
import time
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional
from .models import OllamaConfig

class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def preflight_check(self):
        root = self._get_root_url()
        try:
            v_resp = requests.get(f"{root}/api/version", timeout=10)
            v_resp.raise_for_status()
            
            t_resp = requests.get(f"{root}/api/tags", timeout=10)
            t_resp.raise_for_status()
            
            models = [m.get("name") for m in (t_resp.json().get("models") or [])]
            if self.cfg.model not in models:
                raise RuntimeError(f"Model '{self.cfg.model}' not found. Run `ollama pull {self.cfg.model}`")
        except requests.RequestException as e:
            raise RuntimeError(f"Could not connect to Ollama at {root}: {e}")

    def request(self, messages: List[Dict[str, str]], fmt: Any) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "format": fmt,
            "options": {"temperature": self.cfg.temperature, "seed": self.cfg.seed},
        }
        
        r = requests.post(self.cfg.url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        
        data = r.json()
        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
            
        return ((data.get("message") or {}).get("content") or "").strip()

    def _get_root_url(self) -> str:
        u = urlparse(self.cfg.url)
        return f"{u.scheme}://{u.netloc}"

    def extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
            
        raise ValueError("No valid JSON found in model output")

    def sleep_backoff(self, attempt: int):
        time.sleep(self.cfg.base_backoff_s * (2 ** attempt))