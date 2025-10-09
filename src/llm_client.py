"""
LLM Client Module - Pure API interaction with multiple LLM providers.

Handles API calls, token counting, cost calculation, and usage tracking.
Does NOT handle reasoning strategies - that's in reasoning.py
"""

import time
from typing import Dict, Optional, Tuple
from .utils import load_api_keys, estimate_tokens


# Pricing per 1M tokens (update as needed)
PRICING = {
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
    'gemini-2.5-pro': {'input': 1.25, 'output': 5.00},
    'llama-3.1-8b': {'input': 0.0, 'output': 0.0},  # Free (on-device)
    'mistral-7b': {'input': 0.0, 'output': 0.0},
    'alpaca-7b': {'input': 0.0, 'output': 0.0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD based on token usage."""
    if model not in PRICING:
        return 0.0
    pricing = PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing['input']
    output_cost = (completion_tokens / 1_000_000) * pricing['output']
    return input_cost + output_cost


class LLMClient:
    """Unified LLM client for cloud and on-device models."""
    
    MODELS = {
        'gpt-5': 'openai',
        'gpt-5-nano': 'openai',
        'claude-4.0-sonnet': 'anthropic',
        'gemini-2.5-pro': 'google',
        'llama-3.1-8b': 'ollama',
        'mistral-7b': 'ollama',
        'alpaca-7b': 'ollama',
    }
    
    OLLAMA_MODELS = {
        'llama-3.1-8b': 'llama3.1:8b',
        'mistral-7b': 'mistral:7b',
        'alpaca-7b': 'alpaca:7b',
    }
    
    def __init__(self, model: str = "gpt-5-nano"):
        self.model = model.lower()
        if self.model not in self.MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        self.provider = self.MODELS[self.model]
        self.api_keys = load_api_keys()
        self.client = None
        self.usage_stats = {
            'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
            'total_latency': 0.0, 'total_cost': 0.0, 'num_requests': 0
        }
        self._gpt5_warning_shown = False  # Track if we've shown GPT-5 warnings
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize provider-specific client."""
        if self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_keys['openai'])
            
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_keys['anthropic'])
            
        elif self.provider == 'google':
            import google.generativeai as genai
            genai.configure(api_key=self.api_keys['google'])
            self.client = genai
            
        elif self.provider == 'ollama':
            import ollama
            self.client = ollama
            self.ollama_model = self.OLLAMA_MODELS[self.model]
            print(f"✅ Ollama: {self.model}")
    
    def call_api(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 1000, seed: Optional[int] = None) -> Tuple[Optional[str], Dict]:
        """Call LLM API and return response with usage info."""
        try:
            if self.provider == 'openai':
                return self._call_openai(prompt, temperature, max_tokens, seed)
            elif self.provider == 'anthropic':
                return self._call_anthropic(prompt, temperature, max_tokens)
            elif self.provider == 'google':
                return self._call_google(prompt, temperature, max_tokens)
            elif self.provider == 'ollama':
                return self._call_ollama(prompt, temperature, max_tokens)
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, {'error': str(e)}
    
    def _call_openai(self, prompt: str, temperature: float, max_tokens: int,
                     seed: Optional[int]) -> Tuple[str, Dict]:
        # GPT-5 models only support temperature=1.0 and use reasoning tokens
        if 'gpt-5' in self.model.lower():
            # Show warnings only once per client instance
            if not self._gpt5_warning_shown:
                if temperature != 1.0:
                    print(f"  ℹ️  Note: GPT-5 only supports temperature=1.0 (requested {temperature})")
                if max_tokens < 4000:
                    print(f"  ℹ️  Note: GPT-5 uses reasoning tokens internally, increasing max_tokens {max_tokens} → 4000")
                self._gpt5_warning_shown = True
            
            # Apply GPT-5 constraints
            temperature = 1.0
            if max_tokens < 4000:
                max_tokens = 4000
        
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
        if seed:
            params["seed"] = seed
        
        start = time.time()
        resp = self.client.chat.completions.create(**params)
        latency = time.time() - start
        
        pt, ct, total = resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens
        cost = calculate_cost(self.model, pt, ct)
        
        self._update_stats(pt, ct, total, latency, cost)
        
        content = resp.choices[0].message.content
        
        return content, {
            'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': total,
            'latency': latency, 'cost': cost, 'seed': seed,
            'provider': 'openai', 'deployment': 'cloud'
        }
    
    def _call_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        start = time.time()
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start
        
        pt, ct = resp.usage.input_tokens, resp.usage.output_tokens
        cost = calculate_cost(self.model, pt, ct)
        
        self._update_stats(pt, ct, pt + ct, latency, cost)
        
        return resp.content[0].text, {
            'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct,
            'latency': latency, 'cost': cost, 'provider': 'anthropic', 'deployment': 'cloud'
        }
    
    def _call_google(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        start = time.time()
        model = self.client.GenerativeModel(self.model)
        resp = model.generate_content(prompt, generation_config={
            'temperature': temperature, 'max_output_tokens': max_tokens
        })
        latency = time.time() - start
        
        text = resp.text
        pt, ct = estimate_tokens(prompt), estimate_tokens(text)
        cost = calculate_cost(self.model, pt, ct)
        
        self._update_stats(pt, ct, pt + ct, latency, cost)
        
        return text, {
            'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct,
            'latency': latency, 'cost': cost, 'provider': 'google', 'deployment': 'cloud'
        }
    
    def _call_ollama(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        start = time.time()
        resp = self.client.generate(
            model=self.ollama_model, prompt=prompt,
            options={'temperature': temperature, 'num_predict': max_tokens}
        )
        latency = time.time() - start
        
        text = resp['response']
        pt = resp.get('prompt_eval_count', estimate_tokens(prompt))
        ct = resp.get('eval_count', estimate_tokens(text))
        cost = 0.0  # Free for on-device models
        
        self._update_stats(pt, ct, pt + ct, latency, cost)
        
        return text, {
            'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct,
            'latency': latency, 'cost': cost, 'provider': 'ollama', 'deployment': 'on-device'
        }
    
    def _update_stats(self, pt: int, ct: int, total: int, latency: float, cost: float):
        """Update cumulative usage statistics."""
        self.usage_stats['prompt_tokens'] += pt
        self.usage_stats['completion_tokens'] += ct
        self.usage_stats['total_tokens'] += total
        self.usage_stats['total_latency'] += latency
        self.usage_stats['total_cost'] += cost
        self.usage_stats['num_requests'] += 1
    
    def get_usage_summary(self) -> Dict:
        """Get summary of API usage and costs."""
        n = max(self.usage_stats['num_requests'], 1)
        return {
            **self.usage_stats,
            'average_latency': self.usage_stats['total_latency'] / n,
            'tokens_per_second': self.usage_stats['total_tokens'] / max(self.usage_stats['total_latency'], 0.001),
            'provider': self.provider, 'model': self.model
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
            'total_latency': 0.0, 'total_cost': 0.0, 'num_requests': 0
        }