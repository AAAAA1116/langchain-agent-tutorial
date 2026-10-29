"""
2.2 流式输出  ——  在 2.1 的基础上只引进一个新概念：流式（stream）

图的拓扑、工具、审批逻辑和 2.1 完全一样，唯一改动的是最后这段 __main__：
  invoke()   —— 整张图跑完才一次性返回，等待期间屏幕一片黑
  stream()   —— 边跑边吐：模型每生成一个 token、工具每返回一条结果，立刻推出来

三条必须先记住的结论（在本机 langgraph 1.2.16 上实测）：
  1. stream_mode="messages" 给出的是「过程」：(AIMessageChunk, 元数据) 元组，一个 token 一个 chunk；
     stream_mode="updates"   给出的是「结果」：某个节点跑完后写回状态的增量 dict。
     两个可以一起订阅：stream_mode=["messages", "updates"]，产出统一的 (模式名, payload) 元组。
  2. 中断（interrupt）不会被抛异常！它是以特殊键 __interrupt__ 出现在 updates 的 payload 里。
     只订阅 messages 时流会**静默结束**，你什么都收不到 —— 所以这两种模式必须同时订阅。
  3. 恢复运行就把 Command(resume=...) 当作输入传给 stream()，和 invoke() 的用法完全一致。
"""
import os
import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, ToolMessage


@tool
def get_weather(city: str) -> str:
    "查询指定城市的天气"
    return f"城市：{city}, 气温25度，雨天"

@tool
def get_time(city: str) -> str:
    "查询指定城市的时间"
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"{city}现在{time_str}"

REMINDERS_FILE = Path("reminders.json")


def load_reminders() -> list:
    """读提醒列表，文件不存在就返回空列表"""
    if not REMINDERS_FILE.exists():
        return []
    data = json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    #兼容没 id 字段的旧数据，按顺序补上编号
    for i, r in enumerate(data, 1):
        r.setdefault("id", i)
    return data


def save_reminders(data: list) -> None:
    REMINDERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(item: dict) -> str:
    return f"#{item['id']} {item['remind_at']}（{item['repeat']}）{item['content']}"


def dump_all(data: list) -> str:
    return "\n".join(fmt(r) for r in data) or "（无）"


@tool
def reminder(content: str, remind_at: str, repeat: str = "none") -> str:
    """把用户话里的待办事项存成提醒。只要用户表达了“要做什么”+“什么时候”就必须调用本工具。

    Args:
        content: 提醒内容，去掉时间词之后的动作，例如“买双鞋”
        remind_at: 提醒时间，格式 YYYY-MM-DD HH:MM；只说了日期没说钟点时默认 09:00
        repeat: 重复频率，只能是 none、daily、weekly、monthly 之一
    """
    #参数是模型填的，不可信，进函数先校验
    if repeat not in ("none", "daily", "weekly", "monthly"):
        return f"repeat 只能是 none/daily/weekly/monthly，收到的是 {repeat}"
    try:
        datetime.strptime(remind_at, "%Y-%m-%d %H:%M")
    except ValueError:
        return f"remind_at 格式不对，要求 YYYY-MM-DD HH:MM，收到的是 {remind_at}"

    data = load_reminders()
    #用业务 id 而不是列表下标：下标会随删除前移，模型很容易删错条
    new_id = max([r["id"] for r in data], default=0) + 1
    data.append({"id": new_id, "content": content, "remind_at": remind_at, "repeat": repeat})
    save_reminders(data)
    return f"已记录提醒 {fmt(data[-1])}，当前共 {len(data)} 条"


@tool
def list_reminders() -> str:
    """列出当前所有提醒及其编号。用户问“我有哪些提醒”，或要删除、修改某个提醒时，先调本工具。"""
    data = sorted(load_reminders(), key=lambda r: r["remind_at"])
    if not data:
        return "当前没有任何提醒"
    return dump_all(data)


@tool
def cancel_reminder(reminder_id: int = 0, keyword: str = "") -> str:
    """删除指定提醒。知道编号就用 reminder_id，只知道用户说的是什么事就用 keyword。"""
    data = load_reminders()
    if not data:
        return "当前没有任何提醒，无需删除"

    if reminder_id:
        hit = [r for r in data if r["id"] == reminder_id]
        if not hit:
            return f"找不到编号 {reminder_id}。现有提醒：\n{dump_all(data)}"
    elif keyword:
        hit = [r for r in data if keyword in r["content"]]
        if not hit:
            return f"没有内容包含“{keyword}”的提醒。现有提醒：\n{dump_all(data)}"
        if len(hit) > 1:
            #歧义时不自作主张删第一条，把候选列表回给模型让用户选
            return f"“{keyword}”匹配到 {len(hit)} 条，请让用户确认编号后再删：\n{dump_all(hit)}"
    else:
        return f"需要 reminder_id 或 keyword 其中之一。现有提醒：\n{dump_all(data)}"

    target = hit[0]
    save_reminders([r for r in data if r["id"] != target["id"]])
    return f"已删除提醒 {fmt(target)}"


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

tools = [get_weather, get_time, reminder, list_reminders, cancel_reminder]

# 绑好工具的模型，节点里直接调它就能带出 tool_calls
llm_with_tools = llm.bind_tools(tools)


# model 节点：进来完整历史，出去一条新 AIMessage
def repair_dangling(messages: list) -> list:
    """给没人回复的 tool_calls 补上占位 ToolMessage，否则下一次请求直接被 API 拒 400"""
    answered = {m.tool_call_id for m in messages if m.type == "tool"}
    fixed = []
    for m in messages:
        if m.type == "ai":
            for tc in m.tool_calls:
                if tc["id"] not in answered:
                    fixed.append(ToolMessage(content="(本次工具调用被中断，未执行)", tool_call_id=tc["id"]))
    return fixed


def call_model(state: MessagesState) -> dict:
    #模型不知道今天几号，不告知就解不了“明天”“下周三”这类相对时间
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = (f"今天是 {today}。回答前先调用工具查信息；用户说到待办和时间时调用 reminder 存提醒，"
              f"问现有提醒时调 list_reminders，要删除时调 cancel_reminder。")
    messages = [SystemMessage(content=prompt)] + state["messages"] + repair_dangling(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": response}


# 条件边：看最后一条消息有没有 tool_calls，决定去 tools 还是结束
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "approve" if last.tool_calls else END


def approve(state):
    last = state["messages"][-1]
    answer = interrupt(f"want use to {[tc['name'] for tc in last.tool_calls]}")
    if answer.strip().lower() in ("y", "yes"):
        return {}  #什么都不改
    return {"messages":[
        ToolMessage(content = "reject to answer", tool_call_id = tc["id"])
        for tc in last.tool_calls
    ]}

def after_approve(state) -> str:
    last = state["messages"][-1]
    return "model" if last.type == "tool" else "tools"


#入口 → model ⇄ tools，直到 model 不再调工具 → END
graph = (
    StateGraph(MessagesState)
    .add_node("model", call_model)
    .add_node("tools", ToolNode(tools))
    .add_node("approve", approve)
    .add_edge(START, "model")
    .add_conditional_edges("model", should_continue, ["approve", END])
    .add_conditional_edges("approve", after_approve, ["tools", "model"])
    .add_edge("tools", "model")   # 这条边就是循环本身
    .compile(checkpointer=MemorySaver())
)


# ============================ 以下是本版新增的全部内容 ============================

# 同时订阅两种模式：messages 拿过程（token），updates 拿结果（节点写回了什么）＋中断
STREAM_MODES = ["messages", "updates"]


def render_once(inp, config):
    """消费一次 stream，把过程实时打印出来，返回没处理完的中断（没有就返回 None）"""
    pending = None          # 本趟撞到的 interrupt
    last_node = None        # 上一个 chunk 来自哪个节点，用来决定要不要换行打标题

    for mode, payload in graph.stream(inp, config=config, stream_mode=STREAM_MODES):

        # ---- updates：某个节点刚跑完，payload 是 {节点名: 写回状态的增量} ----
        if mode == "updates":
            for key, value in payload.items():
                if key == "__interrupt__":
                    # 中断不是一个节点，而是挂在 payload 上的特殊键。value 是 Interrupt 元组
                    pending = value[0]
                elif key == "model" and value:
                    # 模型节点跑完了，这里拿到的是**完整的** AIMessage，
                    # 所以 tool_calls 在这里打印，避免去解析 messages 里那些残缺的 tool_call 分片
                    msg = value["messages"]
                    msg = msg[-1] if isinstance(msg, list) else msg
                    for tc in getattr(msg, "tool_calls", None) or []:
                        print(f"\n[调用工具] {tc['name']}({tc['args']})", flush=True)
            continue

        # ---- messages：(消息块, 元数据)，token 级的流式就靠它 ----
        chunk, meta = payload
        node = meta.get("langgraph_node")

        if node == "tools":
            # 工具节点吐的是完整 ToolMessage，不是 token，因为工具本身没有逐字输出的概念
            print(f"\n[工具返回] {chunk.content}", flush=True)
            last_node = node
        elif node == "model" and chunk.content:
            if last_node != "model":
                print("\n[助手] ", end="", flush=True)
            print(chunk.content, end="", flush=True)   # 边生成边打印的关键：end="" + flush=True
            last_node = node

    return pending


def turn(user_input, config):
    """跑完一整轮：用户一句话进去，内部可能经历多次「中断 → 人工审批 → 继续」"""
    inp = {"messages": [{"role": "user", "content": user_input}]}
    while True:
        pending = render_once(inp, config)
        if pending is None:      # render_once 返回 None 说明流正常走到 END，换个行美观一下。
            print("\n")
            return
        # 有中断：把 interrupt 的提示语问给用户，再用 Command(resume=答案) 接着往下流
        answer = input(f"\n{pending.value}(y/n)")
        inp = Command(resume=answer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10}
    while True:
        user_input = input("你:")
        if user_input in ("q", "exit"):
            break
        turn(user_input, config)


# ---------------------------------------------------------------- 小结备忘
# stream_mode 全家桶（想换就改 STREAM_MODES）：
#   "values"   每次写状态后，吐出**完整快照**（state 全量），最省心但数据量最大
#   "updates"  只吐本步的**增量**，格式 {节点名: 增量}，是做日志/落库/接 UI 的首选
#   "messages" 吐 LLM 的 token 流 (AIMessageChunk, meta)，做打字机效果用这个
#   "debug"    连 Pregel 的任务调度细节都吐出来，排查死循环时用
#   "custom"   节点里自己 get_stream_writer().write(...) 的东西，比如给前端推进度条
#
# 三个易踩的点：
#   1. 多种模式一起订阅时，产出统一变成 (模式名, payload) 的二元组，必须先拆模式
#   2. messages 模式下每个 chunk 是 AIMessageChunk，可以直接相加合并（chunk1 + chunk2），
#      LangGraph 靠它把工具调用的碎片拼回完整的 tool_calls；最后还会来一个 content 为空的收尾 chunk
#   3. 节点里写 llm.invoke(...) 也能拿到 token 流 —— LangGraph 内部挂了回调监听模型事件，
#      不需要你改成 llm.stream(...)
