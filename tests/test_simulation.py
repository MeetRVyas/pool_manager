"""
Chaos simulation: a fake provider that ENFORCES real rate limits, a 503 outage and a revoked key,
driven through the public MultiProviderChatLLM API on a fake clock.
"""
import collections

from pool_manager.llm import MultiProviderChatLLM
from pool_manager.scheduler import LLMScheduler
from tests.fakes import (
    FakeTime, gemini_429, gemini_client_error, gemini_server_error, make_ep, run,
)
from langchain_core.messages import AIMessage

LATENCY = 0.4   # simulated seconds per provider call


class EnforcingProvider:
    """Behaves like a real API: true per-minute limit per (model, key), outages, revoked keys."""

    def __init__(self, ft, ep_id, true_rpm, outage=None, revoked=False):
        self.ft, self.id, self.true_rpm, self.outage, self.revoked = ft, ep_id, true_rpm, outage, revoked
        self.window = []
        self.calls = self.ok = self.got_429 = 0

    async def ainvoke(self, messages, **kw):
        self.calls += 1
        self.ft.t += LATENCY
        now = self.ft.t
        if self.revoked:
            raise gemini_client_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")
        if self.outage and self.outage[0] <= now < self.outage[1]:
            raise gemini_server_error(503, "UNAVAILABLE", "This model is currently experiencing high demand.")
        self.window = [t for t in self.window if t > now - 60]
        if len(self.window) >= self.true_rpm:
            self.got_429 += 1
            raise gemini_429(retry="25s")
        self.window.append(now)
        self.ok += 1
        return AIMessage(content=self.id, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


def build(ft):
    spec = [  # (id, priority, true_rpm, yaml_rpm or None, extra provider behaviour)
        ("google:flash:KEY_1", 0, 10, 10, {}),
        ("google:flash:KEY_2", 0, 10, 10, {"outage": (1030.0, 1075.0)}),   # 45 s of 503s
        ("google:flash:KEY_3", 0, 10, 10, {}),
        ("google:lite:KEY_1", 1, 15, 15, {}),
        ("google:lite:KEY_2", 1, 15, 15, {}),
        ("google:lite:KEY_3", 1, 15, 15, {"revoked": True}),               # dead key
        ("groq:llama:KEY_1", 2, 20, None, {}),                              # NO yaml limit -> reactive only
    ]
    eps, providers = [], {}
    for ep_id, prio, true_rpm, yaml_rpm, extra in spec:
        prov = EnforcingProvider(ft, ep_id, true_rpm, **extra)
        providers[ep_id] = prov
        eps.append(make_ep(ep_id, prov, priority=prio, rpm_limit=yaml_rpm))
    sched = LLMScheduler(eps, max_wait_seconds=60, clock=ft.now, sleep=ft.sleep, asleep=ft.asleep)
    return MultiProviderChatLLM(scheduler=sched, max_attempts=6), providers, eps


def test_pool_serves_every_call_through_limits_an_outage_and_a_dead_key():
    ft = FakeTime()
    llm, providers, eps = build(ft)
    served, failures, per_call = collections.Counter(), [], []

    async def workload():
        for i in range(400):
            t0 = ft.now()
            try:
                out = await llm.ainvoke(f"q{i}")
                served[out.content] += 1
            except Exception as exc:   # anything reaching the caller is a failure of the pool
                failures.append(repr(exc))
            per_call.append(ft.now() - t0)

    run(workload())

    assert failures == []                                              # never stopped, never crashed
    assert sum(served.values()) == 400

    # PRE-STOP: models with a YAML limit never drew a 429 from the provider.
    yaml_limited = [p for k, p in providers.items() if not k.startswith("groq")]
    assert sum(p.got_429 for p in yaml_limited) == 0

    # REACTIVE: the model with no YAML limit hit real 429s, handled by cooling down + switching.
    assert providers["groq:llama:KEY_1"].got_429 >= 1

    # Dead key: probed once, then quarantined (not hammered).
    assert providers["google:lite:KEY_3"].calls == 1

    # Outage: the 503 key was tried, backed off, and traffic kept flowing elsewhere.
    assert 1 <= providers["google:flash:KEY_2"].calls - providers["google:flash:KEY_2"].ok <= 6

    # All healthy keys carried traffic.
    used = {k for k, p in providers.items() if p.ok > 0}
    assert {"google:flash:KEY_1", "google:flash:KEY_2", "google:flash:KEY_3",
            "google:lite:KEY_1", "google:lite:KEY_2", "groq:llama:KEY_1"} <= used

    # Bounded waiting: no single call ever stalled longer than the wait budget (+ its own latency).
    assert max(per_call) <= 60 + 6 * LATENCY
    # and the pool never slept in big blocks: waits are exact "until the earliest key frees up"
    assert max(ft.sleeps) <= 1.0 + 1e-9
