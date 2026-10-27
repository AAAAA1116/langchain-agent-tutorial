import os
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage


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

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

tools = [get_weather, get_time]

# 绑好工具的模型，节点里直接调它就能带出 tool_calls
llm_with_tools = llm.bind_tools(tools)

system_prompt = "回答前先调用工具查信息"


# model 节点：进来完整历史，出去一条新 AIMessage
def call_model(state: MessagesState) -> dict:
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
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
    from langchain_core.messages import ToolMessage
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
# START → model ⇄ approve → tools → model
#               ↘ (拒绝) → model        ↘ END (无 tool_calls)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10}
    while True:
        user_input = input("你:")
        if user_input in ("q", "exit"):
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        if "__interrupt__" in result:
            answer = input(result["__interrupt__"][0].value + "(y/n)")
            result = graph.invoke(Command(resume=answer),config)
        messages = result["messages"]
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
