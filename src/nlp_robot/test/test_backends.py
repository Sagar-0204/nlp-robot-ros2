"""Backend request/response handling, with the HTTP layer mocked out."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_robot import llm_backends as B  # noqa: E402
from nlp_robot.plan_schema import PLAN_JSON_SCHEMA  # noqa: E402

PLAN = {'reply': 'ok', 'steps': [{'action': 'navigate_to', 'target': 'red_box'}]}


class FakeResp:
    status_code, ok, text = 200, True, ""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def capture(monkeypatch):
    calls = {}

    def install(payload):
        def fake_post(url, json=None, headers=None, timeout=None):
            calls.update(url=url, body=json, headers=headers or {})
            return FakeResp(payload)
        monkeypatch.setattr(B.requests, 'post', fake_post)
        return calls
    return install


def test_ollama(capture):
    calls = capture({'message': {'content': json.dumps(PLAN)}})
    out = B.OllamaBackend(model='m').complete('sys', 'user', PLAN_JSON_SCHEMA)
    assert out == PLAN
    assert calls['url'].endswith('/api/chat') and calls['body']['format'] == PLAN_JSON_SCHEMA
    assert calls['body']['stream'] is False
    assert calls['body']['keep_alive'] == '60m'      # stays loaded between commands


def test_anthropic_tool_use(capture, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'k')
    calls = capture({'content': [{'type': 'text', 'text': 'hi'},
                                 {'type': 'tool_use', 'name': 'submit_plan', 'input': PLAN}]})
    out = B.AnthropicBackend().complete('sys', 'user', PLAN_JSON_SCHEMA)
    assert out == PLAN
    assert calls['headers']['x-api-key'] == 'k'
    assert calls['body']['tool_choice'] == {'type': 'tool', 'name': 'submit_plan'}
    assert calls['body']['model'] == 'claude-haiku-5-5'


def test_anthropic_needs_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(B.BackendError):
        B.AnthropicBackend().complete('s', 'u', PLAN_JSON_SCHEMA)


def test_openai_compat(capture):
    calls = capture({'choices': [{'message': {'content': '```json\n' + json.dumps(PLAN) + '\n```'}}]})
    out = B.OpenAICompatBackend(model='x', base_url='http://localhost:8000/v1').complete(
        's', 'u', PLAN_JSON_SCHEMA)
    assert out == PLAN and calls['url'] == 'http://localhost:8000/v1/chat/completions'


def test_network_error_becomes_backend_error(monkeypatch):
    def boom(*a, **k):
        raise B.requests.ConnectionError('refused')
    monkeypatch.setattr(B.requests, 'post', boom)
    with pytest.raises(B.BackendError, match='ollama serve'):
        B.OllamaBackend().complete('s', 'u', PLAN_JSON_SCHEMA)


@pytest.mark.parametrize('name', list(B.BACKENDS))
def test_every_backend_accepts_commander_kwargs(name):
    # Exactly what commander_node passes; regression test for "RuleBackend() takes no arguments"
    kw = dict(model='m', base_url='', api_key_env='', temperature=0.0, timeout_s=60.0)
    assert B.make_backend(name, **kw).describe()


def test_make_backend_ignores_empty_strings():
    be = B.make_backend('ollama', model='', base_url='')
    assert be.model == 'qwen2.5:7b-instruct'
    with pytest.raises(B.BackendError):
        B.make_backend('gpt-9000')


# ---------------------------------------------------------------- "Ollama up, model missing"
class Resp404(FakeResp):
    status_code, ok, text = 404, False, '{"error":"model not found"}'


def test_ollama_404_names_the_missing_model(monkeypatch):
    monkeypatch.setattr(B.requests, 'post', lambda *a, **k: Resp404({}))
    with pytest.raises(B.BackendError, match=r'ollama pull qwen2.5:7b-instruct'):
        B.OllamaBackend().complete('s', 'u', PLAN_JSON_SCHEMA)


@pytest.mark.parametrize('installed,ok', [
    (['qwen2.5:7b-instruct'], True),
    (['qwen2.5:7b', 'llama3.2:latest'], False),
    ([], False),
])
def test_ollama_check(monkeypatch, installed, ok):
    monkeypatch.setattr(B.requests, 'get',
                        lambda *a, **k: FakeResp({'models': [{'name': n} for n in installed]}))
    good, why = B.OllamaBackend().check()
    assert good is ok
    if not ok:
        assert 'ollama pull qwen2.5:7b-instruct' in why


def test_ollama_check_server_down(monkeypatch):
    def boom(*a, **k):
        raise B.requests.ConnectionError('refused')
    monkeypatch.setattr(B.requests, 'get', boom)
    good, why = B.OllamaBackend().check()
    assert not good and 'ollama serve' in why


def test_ollama_timeout_explains_itself(monkeypatch):
    def slow(*a, **k):
        raise B.requests.ReadTimeout('read timed out')
    monkeypatch.setattr(B.requests, 'post', slow)
    with pytest.raises(B.BackendError, match='ollama ps'):
        B.OllamaBackend(timeout_s=5).complete('s', 'u', PLAN_JSON_SCHEMA)


def test_ollama_default_timeout_is_generous():
    assert B.OllamaBackend().timeout_s >= 180


@pytest.mark.parametrize('size,vram,expected', [
    (100, 100, '100% GPU'), (100, 0, '100% CPU'), (100, 40, '40% GPU / 60% CPU')])
def test_ollama_placement(monkeypatch, size, vram, expected):
    monkeypatch.setattr(B.requests, 'get', lambda *a, **k: FakeResp(
        {'models': [{'name': 'qwen2.5:7b-instruct', 'size': size, 'size_vram': vram}]}))
    assert B.OllamaBackend().placement() == expected


def test_ollama_warmup(capture, monkeypatch):
    calls = capture({})
    monkeypatch.setattr(B.requests, 'get', lambda *a, **k: FakeResp({'models': []}))
    secs, where = B.OllamaBackend().warmup()
    assert calls['url'].endswith('/api/generate') and calls['body']['keep_alive'] == '60m'
    assert secs >= 0 and where == 'unknown'


def test_fallback_is_reported():
    from nlp_robot.command_parser import CommandParser
    from nlp_robot.plan_schema import WorldContext

    class Dead(B.LLMBackend):
        name = 'dead'

        def complete(self, *a):
            raise B.BackendError('Ollama has no model "x"')
    ctx = WorldContext(places={'home': {'pose': [0, 0, 0]}})
    r = CommandParser(Dead(), lambda: ctx, fallback=B.RuleBackend()).parse('go home')
    assert r.ok and r.backend == 'rules' and 'no model' in r.fallback_reason
