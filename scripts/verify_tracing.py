"""In-container verification driver for Langfuse tracing.

Runs INSIDE the sqlbot-langfuse:dev container. It exercises the REAL
LLMService.run_task code path (real instrumented llm.py) end-to-end against a
mock OpenAI server + the real Langfuse instance, then asserts the resulting
traces in Langfuse.

Usage (from host):
  docker cp scripts/verify_tracing.py sqlbot-langfuse-dev:/tmp/verify_tracing.py
  docker exec -e PYTHONPATH=/opt/sqlbot/app sqlbot-langfuse-dev \\
    /opt/sqlbot/app/.venv/bin/python /tmp/verify_tracing.py

It proves:
  * each request -> exactly one trace
  * internal calls -> generations (LLM) + spans (tools) under that trace
  * multiple requests in the same chat -> different traces, same session_id
"""
import asyncio
import base64
import json
import os
import time
import urllib.request

from sqlmodel import Session, select

from apps.chat.models.chat_model import Chat, ChatQuestion, ChatFinishStep
from apps.chat.task.llm import LLMService
from apps.datasource.models.datasource import CoreDatasource, CoreTable, CoreField
from apps.datasource.utils.utils import aes_encrypt
from apps.system.crud.user import get_user_info
from apps.system.models.system_model import AiModelDetail
from common.core.db import engine

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://host.docker.internal:3000").rstrip("/")
LANGFUSE_PK = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SK = os.environ.get("LANGFUSE_SECRET_KEY", "")
DS_NAME = "lf_test_ds"
MODEL_NAME = "lf_test_model"
TABLE = "chat"


def _basic_auth_header():
    token = base64.b64encode(f"{LANGFUSE_PK}:{LANGFUSE_SK}".encode()).decode()
    return {"Authorization": "Basic " + token}


def query_traces(session_id):
    url = f"{LANGFUSE_HOST}/api/public/traces?sessionId={session_id}&limit=50"
    req = urllib.request.Request(url, headers=_basic_auth_header())
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read()).get("data", [])


def fetch_trace_detail(trace_id):
    url = f"{LANGFUSE_HOST}/api/public/traces/{trace_id}"
    req = urllib.request.Request(url, headers=_basic_auth_header())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def seed(session):
    # clean old test data (idempotent re-runs)
    old_ds_ids = [d.id for d in session.exec(select(CoreDatasource).where(CoreDatasource.name == DS_NAME))]
    if old_ds_ids:
        session.exec(select(CoreField).where(CoreField.ds_id.in_(old_ds_ids)))
        for f in session.exec(select(CoreField).where(CoreField.ds_id.in_(old_ds_ids))):
            session.delete(f)
        for t in session.exec(select(CoreTable).where(CoreTable.ds_id.in_(old_ds_ids))):
            session.delete(t)
    for ds in session.exec(select(CoreDatasource).where(CoreDatasource.name == DS_NAME)):
        session.delete(ds)
    for c in session.exec(select(Chat).where(Chat.brief == "lf-test-chat")):
        session.delete(c)
    for am in session.exec(select(AiModelDetail).where(AiModelDetail.name == MODEL_NAME)):
        session.delete(am)
    # reset any other default models so ours is the unique default
    for am in session.exec(select(AiModelDetail).where(AiModelDetail.default_model == True)):
        am.default_model = False
        session.add(am)
    session.commit()

    # 1) AI model -> mock OpenAI server (default, so get_default_config picks it)
    model = AiModelDetail(name=MODEL_NAME, base_model="mock-gpt", model_type=0, supplier=0,
                          protocol=1, api_domain="http://host.docker.internal:8888",
                          api_key="sk-mock", default_model=True, config="[]")
    session.add(model)
    session.commit()
    session.refresh(model)

    # 2) datasource -> the container's own postgres (has the `chat` table)
    conf = {"host": "127.0.0.1", "port": 5432, "username": "root",
            "password": "Password123@pg", "database": "sqlbot", "timeout": 30}
    ds = CoreDatasource(name=DS_NAME, description="lf test", type="pg",
                        type_name="PostgreSQL", configuration=aes_encrypt(json.dumps(conf)).decode(),
                        oid=1, status="1")
    session.add(ds)
    session.commit()
    session.refresh(ds)

    table = CoreTable(ds_id=ds.id, checked=True, table_name=TABLE)
    session.add(table)
    session.commit()
    session.refresh(table)
    for fname, ftype in [("id", "int4"), ("brief", "varchar")]:
        session.add(CoreField(ds_id=ds.id, table_id=table.id, checked=True,
                              field_name=fname, field_type=ftype, field_index=0))
    session.commit()

    # 3) chat bound to this datasource
    chat = Chat(oid=1, datasource=ds.id, engine_type="PostgreSQL", chat_type="chat",
                brief="lf-test-chat")
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat, ds


async def ask(session, user, chat_id, ds_id, question):
    q = ChatQuestion(chat_id=chat_id, question=question, datasource_id=ds_id)
    svc = await LLMService.create(session, user, q, None)
    svc.init_record(session)
    svc.run_task_async(in_chat=False, stream=False, finish_step=ChatFinishStep.GENERATE_CHART)
    # consume the async generator (drives run_task in the worker thread)
    chunks = list(svc.await_result())
    record_id = svc.get_record().id
    return record_id, chunks


def summarize(trace):
    obs = trace.get("observations", [])
    gens = [o for o in obs if o.get("type") == "GENERATION" or "generation" in str(o.get("type", "")).lower()]
    spans = [o for o in obs if o.get("type") in ("SPAN", "span")]
    names = [o.get("name") for o in obs]
    return {
        "id": trace.get("id"),
        "name": trace.get("name"),
        "sessionId": trace.get("sessionId"),
        "userId": trace.get("userId"),
        "gen_names": [g.get("name") for g in gens],
        "span_names": [s.get("name") for s in spans],
        "all_obs_names": names,
    }


async def main():
    with Session(engine) as session:
        chat, ds = seed(session)
        user = await get_user_info(session=session, user_id=1)
        user.oid = 1
        chat_id, ds_id = chat.id, ds.id

        print(f"[seed] chat_id={chat_id} ds_id={ds_id} (session_id will be {chat_id})")

        # record traces that already exist for this session (from prior runs) so
        # we only assert on the NEW traces produced by this run
        existing_ids = {t["id"] for t in query_traces(str(chat_id))}

        # --- Scenario 1 & 2: two questions in the SAME chat -----------------
        r1 = await ask(session, user, chat_id, ds_id, "查询chat表有多少条记录")
        print(f"[ask#1] record_id={r1[0]}")
        r2 = await ask(session, user, chat_id, ds_id, "再查一次chat表")
        print(f"[ask#2] record_id={r2[0]}")

    # let langfuse worker ingest
    print("[wait] flushing langfuse...")
    time.sleep(6)

    session_id = str(chat_id)
    # only consider traces produced by THIS run (exclude prior runs' leftovers)
    traces = [t for t in query_traces(session_id) if t["id"] not in existing_ids]
    print(f"\n===== Langfuse traces for session_id={session_id}: {len(traces)} =====")
    summaries = []
    for t in traces:
        detail = fetch_trace_detail(t["id"])
        s = summarize(detail or t)
        summaries.append(s)
        print(json.dumps(s, ensure_ascii=False))

    # ----- assertions -----
    print("\n===== ASSERTIONS =====")
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    check(">=2 traces produced (one per request)", len(traces) >= 2)
    check("all traces share the SAME session_id (the chat id)",
          len({t.get("sessionId") for t in traces}) == 1 and traces[0].get("sessionId") == session_id)
    check("traces have DISTINCT ids", len({t.get("id") for t in traces}) >= 2)
    gen_names_all = [n for s in summaries for n in s["gen_names"]]
    check("has a generate_sql generation", any("sql" in (n or "").lower() for n in gen_names_all))
    span_names_all = [n for s in summaries for n in s["span_names"]]
    check("has tool spans (e.g. choose_table_schema/check_connection/execute_sql)",
          any(n in span_names_all for n in
              ("choose_table_schema", "check_connection", "execute_sql", "filter_terminology",
               "filter_sql_example", "filter_custom_prompt", "render_chart_picture")))
    # at least one trace should have reached chart generation (full path)
    check("at least one trace reached generate_chart",
          any("chart" in (n or "").lower() for n in gen_names_all))

    # ----- input/output completeness on EVERY observation ----------------
    TOOL_SPAN_NAMES = {"choose_table_schema", "check_connection", "execute_sql",
                       "filter_terminology", "filter_sql_example", "filter_custom_prompt",
                       "render_chart_picture"}
    missing_io = []  # (trace_id, name, which) where input or output absent
    for t in traces:
        detail = fetch_trace_detail(t["id"])
        for o in detail.get("observations", []):
            name = o.get("name")
            otype = str(o.get("type", "")).upper()
            has_in = o.get("input") is not None
            has_out = o.get("output") is not None
            if otype == "GENERATION":
                if not has_in:
                    missing_io.append((t["id"][:8], name, "input"))
                if not has_out:
                    missing_io.append((t["id"][:8], name, "output"))
            elif name in TOOL_SPAN_NAMES:
                if not has_in:
                    missing_io.append((t["id"][:8], name, "input"))
                if not has_out:
                    missing_io.append((t["id"][:8], name, "output"))
    if missing_io:
        print("  missing input/output details: " + json.dumps(missing_io, ensure_ascii=False))
    check("every generation & key tool span has BOTH input and output", not missing_io)

    # ----- generation input roles use industry-standard names ------------
    STD_ROLES = {"system", "user", "assistant", "tool", "function"}
    bad_roles = []
    for t in traces:
        detail = fetch_trace_detail(t["id"])
        for o in detail.get("observations", []):
            if str(o.get("type", "")).upper() != "GENERATION":
                continue
            inp = o.get("input")
            if not isinstance(inp, list):
                continue
            for m in inp:
                if isinstance(m, dict) and m.get("role") and m.get("role") not in STD_ROLES:
                    bad_roles.append((t["id"][:8], o.get("name"), m.get("role")))
    if bad_roles:
        print("  non-standard roles: " + json.dumps(bad_roles, ensure_ascii=False))
    check("generation input roles are industry-standard (user/assistant, not human/ai)", not bad_roles)

    print(f"\nRESULT: {'ALL PASS ✅' if ok else 'SOME FAILED ❌'}")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    asyncio.run(main())
