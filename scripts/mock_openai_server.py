"""Minimal fake OpenAI-compatible server for tracing verification.

Implements just enough of the OpenAI Chat Completions API (streaming) so that
SQLBot's real langchain-openai client can drive the full `run_task` pipeline,
while we control exactly what each LLM call returns.

The server inspects the prompt to decide which pipeline step is calling and
returns the appropriate canned JSON:
  * SQL generation  -> {"success": true, "sql": "...", "tables": [...], ...}
  * chart config    -> {"type": "table", "columns": [...]}
  * datasource pick -> {"id": <MOCK_DS_ID>}
  * analysis/predict/recommend -> a short text

Stdlib only (http.server) so it runs anywhere with no extra deps. Configure via
env:
  MOCK_PORT          listen port (default 8888)
  MOCK_SQL_TABLE     table name used in the canned SQL (default "chat")
  MOCK_DS_ID         datasource id returned by the select_datasource step
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_PORT = int(os.environ.get("MOCK_PORT", "8888"))
MOCK_SQL_TABLE = os.environ.get("MOCK_SQL_TABLE", "chat")
MOCK_DS_ID = os.environ.get("MOCK_DS_ID", "0")
MODEL = os.environ.get("MOCK_MODEL", "mock-gpt")


def _detect_step(messages):
    """Return one of: sql, chart, datasource, analysis, predict, recommend, text."""
    blob = ""
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):  # multimodal
            c = json.dumps(c, ensure_ascii=False)
        blob += " " + str(c)
    low = blob.lower()
    # init_messages injects distinct AI-confirmation strings per step. Use the
    # most specific ones (the SQL and chart prompts both embed the DB schema,
    # so "【Schema】"/表结构 alone is NOT enough to tell them apart).
    if "符合要求的json" in blob:           # chart init AI confirmation
        return "chart"
    if "表结构schema" in blob or "表結構schema" in blob:  # SQL init AI confirmation
        return "sql"
    if "数据源" in blob or "數據源" in blob or "datasource" in low:
        return "datasource"
    if "预测" in blob or "預測" in blob or "predict" in low:
        return "predict"
    if "分析" in blob or "analysis" in low:
        return "analysis"
    if "推荐" in blob or "推薦" in blob or "recommend" in low:
        return "recommend"
    return "sql"  # default to SQL JSON (most common step)


def _build_answer(step):
    if step == "sql":
        return json.dumps({
            "success": True,
            "sql": f"SELECT * FROM {MOCK_SQL_TABLE} LIMIT 10",
            "tables": [MOCK_SQL_TABLE],
            "chart-type": "table",
            "brief": f"查询 {MOCK_SQL_TABLE}",
        }, ensure_ascii=False)
    if step == "chart":
        return json.dumps({
            "type": "table",
            "columns": [
                {"name": "id", "value": "id"},
                {"name": "name", "value": "name"},
            ],
        }, ensure_ascii=False)
    if step == "datasource":
        return json.dumps({"id": int(MOCK_DS_ID)} if str(MOCK_DS_ID).isdigit() else {"id": 0},
                          ensure_ascii=False)
    if step == "predict":
        return "[]"
    if step in ("analysis", "recommend"):
        return json.dumps([f"{step}-item-1", f"{step}-item-2"], ensure_ascii=False) \
            if step == "recommend" else f"这是 {step} 结果文本。"
    return "ok"


def _sse_chunks(answer):
    """Yield OpenAI-style SSE strings, chunking the answer into a few pieces."""
    cid = "chatcmpl-mock"
    created = int(time.time())
    pieces = [answer[i:i + 12] for i in range(0, len(answer), 12)] or [""]
    for piece in pieces:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    # final chunk with finish_reason + usage
    final = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default logging
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if "/chat/completions" not in self.path:
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        messages = req.get("messages", [])
        step = _detect_step(messages)
        answer = _build_answer(step)
        stream = req.get("stream", True)

        if not stream:
            self._send(200, {
                "id": "chatcmpl-mock", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for chunk in _sse_chunks(answer):
            self.wfile.write(chunk.encode())
            self.wfile.flush()


def main():
    server = ThreadingHTTPServer(("0.0.0.0", MOCK_PORT), Handler)
    print(f"[mock-openai] listening on :{MOCK_PORT} (sql_table={MOCK_SQL_TABLE}, ds_id={MOCK_DS_ID})")
    server.serve_forever()


if __name__ == "__main__":
    main()
