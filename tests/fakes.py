"""Fake HTTP session for exercising broker adapters without network access."""
import json


class Resp:
    def __init__(self, status=200, body=None, ctype="application/json", text=None):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self._body = body
        self.text = text if text is not None else json.dumps(body)
        self.content = self.text.encode()

    def json(self):
        return self._body


class FakeSession:
    """Routes (method, path-suffix) -> Resp or callable(params, data, json) -> Resp."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, params=None, data=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "data": data,
                           "json": json, "headers": headers or {}})
        for (m, suffix), handler in self.routes.items():
            if m == method and url.split("?")[0].endswith(suffix):
                return handler(params, data, json) if callable(handler) else handler
        return Resp(404, {"status": "error", "message": f"no route {method} {url}"})

    def find(self, method, suffix):
        return [c for c in self.calls if c["method"] == method and c["url"].endswith(suffix)]
