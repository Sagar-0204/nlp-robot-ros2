"""LLM backends that turn a prompt into a JSON plan.

Every backend implements `complete(system, user, schema) -> dict` and is forced
to return JSON matching PLAN_JSON_SCHEMA:
  * ollama    - local, offline. Uses Ollama's structured outputs (`format` = JSON schema).
  * anthropic - Claude via the Messages API, using a forced tool call for the schema.
  * openai    - any OpenAI-compatible /chat/completions endpoint (OpenAI, Groq, vLLM,
                LM Studio, llama.cpp server ...) with response_format json_schema.
  * gemini    - Google Gemini through that same OpenAI-compatible API (GEMINI_API_KEY).
  * rules     - no LLM at all: a small regex parser. Used for tests and as an
                offline fallback so a demo never dies because Wi-Fi did.

Only `requests` is needed; no vendor SDKs.
"""
import json
import os
import re
import time

import requests


class BackendError(RuntimeError):
    pass


def _extract_json(text):
    """Parse JSON even if a model wrapped it in prose or ``` fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise BackendError(f'model did not return JSON: {text[:200]!r}')
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise BackendError(f'invalid JSON from model: {e}') from e


class LLMBackend:
    name = 'base'

    def complete(self, system, user, schema):  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self):
        return self.name

    def check(self):
        """(ok, message): can this backend answer right now? Called once at startup."""
        return True, 'ready'

    def warmup(self):
        """Prepare for the first request. Returns (seconds, note). No-op by default."""
        return 0.0, ''


class OllamaBackend(LLMBackend):
    name = 'ollama'

    def __init__(self, model='qwen2.5:7b-instruct', base_url='http://localhost:11434',
                 temperature=0.0, timeout_s=180.0, keep_alive='60m', **_):
        self.model, self.base_url = model, base_url.rstrip('/')
        self.temperature, self.timeout_s = temperature, timeout_s
        # Ollama unloads an idle model after 5 min by default; the next command then waits
        # for a multi-GB reload on top of the answer. Keep it in memory for the session.
        self.keep_alive = keep_alive

    def describe(self):
        return f'ollama:{self.model}'

    def installed_models(self):
        r = requests.get(f'{self.base_url}/api/tags', timeout=3)
        r.raise_for_status()
        return [m['name'] for m in r.json().get('models', [])]

    def _has(self, names):
        want = self.model if ':' in self.model else self.model + ':latest'
        return self.model in names or want in names

    def check(self):
        try:
            names = self.installed_models()
        except requests.RequestException:
            return False, (f'Ollama is not reachable at {self.base_url}. '
                           f'Start it with `ollama serve` (or `sudo systemctl start ollama`).')
        if not self._has(names):
            have = ', '.join(names) if names else 'none'
            return False, (f'model "{self.model}" is not downloaded (installed: {have}). '
                           f'Run `ollama pull {self.model}`, or launch with llm_model:=<one of '
                           f'the installed models>.')
        return True, 'ready'

    def warmup(self):
        """Load the model into memory now, so the first command doesn't pay for it.
        Returns (seconds, where) - where is e.g. "100% GPU" or "35% GPU / 65% CPU"."""
        t0 = time.monotonic()
        try:
            requests.post(f'{self.base_url}/api/generate',
                          json={'model': self.model, 'prompt': '', 'keep_alive': self.keep_alive},
                          timeout=self.timeout_s).raise_for_status()
        except requests.RequestException as e:
            raise BackendError(f'could not load {self.model} into Ollama: {e}') from e
        return time.monotonic() - t0, self.placement()

    def placement(self):
        """How much of the loaded model sits in GPU memory, as `ollama ps` reports it.
        A model partly on the CPU is several times slower."""
        try:
            r = requests.get(f'{self.base_url}/api/ps', timeout=3)
            for m in r.json().get('models', []):
                if m.get('name') == self.model or m.get('model') == self.model:
                    size, vram = m.get('size') or 0, m.get('size_vram') or 0
                    if not size:
                        break
                    gpu = round(100 * vram / size)
                    return '100% GPU' if gpu >= 99 else ('100% CPU' if gpu <= 0
                                                          else f'{gpu}% GPU / {100 - gpu}% CPU')
        except (requests.RequestException, ValueError):
            pass
        return 'unknown'

    def complete(self, system, user, schema):
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'format': schema,
            'stream': False,
            'options': {'temperature': self.temperature},
            'keep_alive': self.keep_alive,
        }
        try:
            r = requests.post(f'{self.base_url}/api/chat', json=body, timeout=self.timeout_s)
        except requests.ConnectionError as e:
            raise BackendError(f'Ollama is not reachable at {self.base_url} - run `ollama serve`'
                               ) from e
        except requests.Timeout as e:
            raise BackendError(f'Ollama took longer than {self.timeout_s:.0f} s. The model may '
                               f'still be loading, or running on the CPU: check `ollama ps`, '
                               f'or try a smaller model') from e
        except requests.RequestException as e:
            raise BackendError(f'Ollama request failed: {e}') from e
        if r.status_code == 404:
            raise BackendError(f'Ollama has no model "{self.model}" - run '
                               f'`ollama pull {self.model}` (see `ollama list`)')
        if not r.ok:
            raise BackendError(f'Ollama error {r.status_code}: {r.text[:200]}')
        return _extract_json(r.json()['message']['content'])


class AnthropicBackend(LLMBackend):
    name = 'anthropic'
    TOOL = 'submit_plan'

    def __init__(self, model='claude-haiku-5-5', base_url='https://api.anthropic.com',
                 api_key_env='ANTHROPIC_API_KEY', temperature=0.0, timeout_s=30.0, **_):
        self.model, self.base_url = model, base_url.rstrip('/')
        self.api_key = os.environ.get(api_key_env, '')
        self.api_key_env = api_key_env
        self.temperature, self.timeout_s = temperature, timeout_s

    def describe(self):
        return f'anthropic:{self.model}'

    def check(self):
        if not self.api_key:
            return False, f'environment variable {self.api_key_env} is not set'
        return True, 'ready'

    def complete(self, system, user, schema):
        if not self.api_key:
            raise BackendError(f'environment variable {self.api_key_env} is not set')
        body = {
            'model': self.model,
            'max_tokens': 1024,
            'temperature': self.temperature,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
            'tools': [{'name': self.TOOL,
                       'description': 'Submit the robot plan for the user command.',
                       'input_schema': schema}],
            'tool_choice': {'type': 'tool', 'name': self.TOOL},
        }
        headers = {'x-api-key': self.api_key, 'anthropic-version': '2023-06-01',
                   'content-type': 'application/json'}
        try:
            r = requests.post(f'{self.base_url}/v1/messages', json=body, headers=headers,
                              timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as e:
            detail = getattr(getattr(e, 'response', None), 'text', '')[:300]
            raise BackendError(f'Anthropic request failed: {e} {detail}') from e
        for block in r.json().get('content', []):
            if block.get('type') == 'tool_use':
                return block['input']
        raise BackendError('Anthropic response contained no tool_use block')


class OpenAICompatBackend(LLMBackend):
    """Any server that speaks the OpenAI chat-completions API with json_schema output."""
    name = 'openai'
    DEFAULT_MODEL = ''
    DEFAULT_URL = 'https://api.openai.com/v1'
    DEFAULT_KEY_ENV = 'OPENAI_API_KEY'
    RETRY_STATUS = (429, 500, 502, 503)   # rate limited / temporarily unavailable
    RETRIES = 3

    def __init__(self, model='', base_url='', api_key_env='', temperature=0.0, timeout_s=60.0,
                 **_):
        self.model = model or self.DEFAULT_MODEL
        if not self.model:
            raise BackendError(f'set llm.model for the {self.name} backend')
        self.base_url = (base_url or self.DEFAULT_URL).rstrip('/')
        self.api_key_env = api_key_env or self.DEFAULT_KEY_ENV
        self.api_key = os.environ.get(self.api_key_env, '')
        self.temperature, self.timeout_s = temperature, timeout_s
        self.retries_used = 0      # rate-limit retries in the last call (for evaluation)

    def describe(self):
        return f'{self.name}:{self.model}'

    def check(self):
        if not self.api_key:
            return False, f'environment variable {self.api_key_env} is not set'
        return True, 'ready'

    def complete(self, system, user, schema):
        if not self.api_key:
            raise BackendError(f'environment variable {self.api_key_env} is not set')
        body = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
            'response_format': {'type': 'json_schema',
                                'json_schema': {'name': 'robot_plan', 'schema': schema}},
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        self.retries_used = 0
        for attempt in range(self.RETRIES + 1):
            try:
                r = requests.post(f'{self.base_url}/chat/completions', json=body,
                                  headers=headers, timeout=self.timeout_s)
            except requests.Timeout as e:
                raise BackendError(f'{self.name} took longer than {self.timeout_s:.0f} s') from e
            except requests.RequestException as e:
                raise BackendError(f'{self.name} request failed: {e}') from e
            if r.status_code in self.RETRY_STATUS and attempt < self.RETRIES:
                # Free tiers allow only a few requests per minute: wait and try again.
                try:
                    wait = float(r.headers.get('Retry-After', ''))
                except ValueError:
                    wait = 5.0 * 2 ** attempt
                self.retries_used += 1
                time.sleep(min(wait, 60.0))
                continue
            break
        if not r.ok:
            raise BackendError(f'{self.name} error {r.status_code}: {r.text[:300]}')
        try:
            content = r.json()['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise BackendError(f'unexpected {self.name} response: {r.text[:300]}') from e
        return _extract_json(content or '')


class GeminiBackend(OpenAICompatBackend):
    """Google Gemini through its OpenAI-compatible endpoint. Key: GEMINI_API_KEY.
    Models (Oct 2026): gemini-3.8-flash (default), gemini-3.5-flash-lite (smaller, faster)."""
    name = 'gemini'
    DEFAULT_MODEL = 'gemini-3.8-flash'
    DEFAULT_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
    DEFAULT_KEY_ENV = 'GEMINI_API_KEY'

    def __init__(self, timeout_s=120.0, **kw):
        # The free tier sometimes queues a request for most of a minute before answering.
        super().__init__(timeout_s=timeout_s, **kw)


_NUM = r'(-?\d+(?:\.\d+)?)'


class RuleBackend(LLMBackend):
    """Deterministic fallback for simple commands. Not an LLM; it understands only
    the patterns below, chained with "then", "and then", "and", "," or ";"."""
    name = 'rules'

    def __init__(self, **_):
        pass  # accepts (and ignores) the LLM settings so it can be the primary backend

    def complete(self, system, user, schema):
        # The user message ends with the raw command after a marker; see command_parser.
        command = user.rsplit('COMMAND:', 1)[-1].strip().strip('"').lower()
        parts = re.split(r'\s*(?:,\s*and then|,\s*then|\bthen\b|;|,|\band\b)\s*', command)
        steps = []
        for p in (x.strip(' .!') for x in parts):
            if not p:
                continue
            step = self._parse(p)
            if step is None:
                return {'reply': f'Sorry, I could not understand "{p}".', 'steps': []}
            steps.append(step)
        return {'reply': f'OK: {command}', 'steps': steps}

    @staticmethod
    def _parse(p):
        if re.fullmatch(r'(please )?(come|go|get) back( here)?|return( here| to start)?', p):
            return {'action': 'return_to_start'}
        m = re.fullmatch(r'(?:turn|rotate|spin)\s+(left|right)(?:\s+(?:by\s+)?' + _NUM +
                         r'\s*(?:deg(?:rees?)?)?)?', p)
        if m:
            ang = float(m.group(2) or 90)
            return {'action': 'rotate', 'angle_deg': ang if m.group(1) == 'left' else -ang}
        if re.fullmatch(r'turn around|spin around|do a u-?turn', p):
            return {'action': 'rotate', 'angle_deg': 180}
        m = re.fullmatch(r'(?:move|go|drive)\s+(forward|ahead|back(?:ward)?s?)\s+' + _NUM +
                         r'\s*(m|meters?|metres?|cm|centimeters?|centimetres?)?', p)
        if m:
            d = float(m.group(2)) / (100.0 if (m.group(3) or 'm').startswith('c') else 1.0)
            return {'action': 'move', 'distance_m': d if m.group(1) in ('forward', 'ahead') else -d}
        m = re.fullmatch(r'(?:back up|back off|reverse)\s+(?:by\s+)?' + _NUM +
                         r'\s*(m|meters?|metres?|cm|centimeters?|centimetres?)?', p)
        if m:
            d = float(m.group(1)) / (100.0 if (m.group(2) or 'm').startswith('c') else 1.0)
            return {'action': 'move', 'distance_m': -d}
        m = re.fullmatch(r'wait(?: for)?\s+' + _NUM + r'\s*(?:s|secs?|seconds?)?', p)
        if m:
            return {'action': 'wait', 'seconds': float(m.group(1))}
        m = re.fullmatch(r'(?:find|look for|search for|locate)\s+(.+)', p)
        if m:
            return {'action': 'find', 'target': m.group(1)}
        m = re.fullmatch(r'(?:go|drive|navigate|move|head|take me)\s+(?:to|towards?)\s+(.+)', p)
        if m:
            return {'action': 'navigate_to', 'target': m.group(1)}
        if p in ('go home', 'return home', 'go back home'):
            return {'action': 'navigate_to', 'target': 'home'}
        return None


BACKENDS = {
    'ollama': OllamaBackend,
    'anthropic': AnthropicBackend,
    'openai': OpenAICompatBackend,
    'gemini': GeminiBackend,
    'rules': RuleBackend,
}


def make_backend(name, **kwargs):
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise BackendError(f'unknown backend "{name}". Choose from {list(BACKENDS)}')
    kwargs = {k: v for k, v in kwargs.items() if v not in (None, '')}
    return cls(**kwargs)
