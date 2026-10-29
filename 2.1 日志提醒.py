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
    # json的四个基础函数：
    # json.dumps(obj) python对象 -> json
    # json.loads(s) json -> Python对象
    # json.dump(obj, f) Python对象 -> 写入文件
    # json.load(f) 读文件 -> Python对象 这两行隐含一个转换

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

@tool
def update_reminder(reminder_id: int, content: str = "", remind_at: str = "", repeat: str = "") -> str:
    """更新指定编号的提醒。只传要改的字段。
    Args：
        reminder_id: id
        content:
        reminder_at:
        repeat:

    """
    data = load_reminders()
    target = next((r for r in data if r["id"] == reminder_id), None)
    if not target:
        return f"cant find id{reminder_id}."

    if repeat:
        #枚举和 reminder 保持一致，否则 yearly 能改出来却建不出来
        if repeat not in ("none", "daily", "weekly", "monthly"):
            return f"repeat 只能是 none/daily/weekly/monthly，收到的是 {repeat}"
    #时间和 repeat 无关，单独传也要校验，否则“明天”会被直接写进文件
    if remind_at:
        try:
            datetime.strptime(remind_at, "%Y-%m-%d %H:%M")
        except ValueError:
            return f"remind_at 格式不对，要求 YYYY-MM-DD HH:MM，收到的是 {remind_at}"
        if remind_at == target["remind_at"]:
            remind_at = ""  #时间和原来一样，视为没改

    if not (content or remind_at or repeat):
        return f"{fmt(target)} 没有任何改动，检查是不是字段都留空了"

    old = fmt(target)
    if content:
        target["content"] = content
    if remind_at:
        target["remind_at"] = remind_at
    if repeat:
        target["repeat"] = repeat
    save_reminders(data)
    return f"已把提醒 {old} 更新为 {fmt(target)}"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

tools = [get_weather, get_time, reminder, list_reminders, cancel_reminder, update_reminder]

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
              f"问现有提醒时调 list_reminders，要删除时调 cancel_reminder，要修改时调 update_reminder，"
              f"别用先删再建的方式改提醒——两步之间被拒绝会只剩一半。")
    messages = [SystemMessage(content=prompt)] + state["messages"] + repair_dangling(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": response}

# state = {"messages": [...]}     # 就一个键

# 条件边：看最后一条消息有没有 tool_calls，决定去 tools 还是结束
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "approve" if last.tool_calls else END


def approve(state):
    print(">>> approve 执行了")
    last = state["messages"][-1]
    answer = interrupt(f"agent want use{[tc['name'] for tc in last.tool_calls]}")
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
# 用户输入 → model节点 ⇄ approve节点(人工审批) ⇄ tools节点(调工具)
#                 ↘ (无工具调用 → END)       ↘ (拒绝 → 回model)

if __name__ == "__main__":
    #一次工具往返吃 4 个超步（model→approve→tools→model），10 只够两轮半，当保险丝太紧
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 40}
    while True:
        user_input = input("你:")
        if user_input in ("q", "exit"):
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        #一轮里可能连续暂停多次（模型分批调工具），必须循环处理到没有暂停为止
        while "__interrupt__" in result:
            answer = input(result["__interrupt__"][0].value + "(y/n)")
            result = graph.invoke(Command(resume=answer), config)
        messages = result["messages"]
        # 测试repair_dangling
        # from langchain_core.messages import AIMessage
        # messages = list(messages) + [
        #     AIMessage(content="", tool_calls=[{"id": "orphan-test", "name": "reminder", "args": {}}])
        # ]
        start = next(i for i in range(len(messages) - 1, -1, -1)
                     if messages[i].type == "human")
        for msg in messages[start:]:
            if msg.type == "ai" and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[调用工具] {tc['name']}({tc['args']})")
            elif msg.type == "tool":
                print(f"[工具返回] {msg.content}")
            elif msg.type == "ai":
                print(f"[最终回答] {msg.content}")

        if repair_dangling(messages):
            print("有工具调用但是未被回复，本轮可能被未处理的暂停打断")
