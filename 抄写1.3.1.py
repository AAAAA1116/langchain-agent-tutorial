import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

#在抄写1.3.py 多轮对话版本上加人工审批：create_agent 传 middleware=[HumanInTheLoopMiddleware(...)]，工具执行前图暂停问人，resume 传决议字典 {"decisions": [{"type": "approve"}]} 或 reject，对比 抄写 1.4 langraph重构.py 里手写 approve 节点的等价实现。

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

#  创建agent
agent = create_agent(
    model=llm,
    tools=[get_weather,get_time],
    system_prompt="回答前先调用工具查信息",
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True, "get_time": True})],
)

if __name__ == "__main__":
    config = {"configurable":{"thread_id":"1"}, "recursion_limit": 10}
    while True:
        user_input = input("你:")
        if user_input in ("q","exit"):
            break

        result = agent.invoke(
            {"messages":[{"role": "user", "content": user_input}]},
            config=config,
        )
        #工具执行前图会暂停，问完人再用 Command 恢复；resume 要传决议字典，不是裸字符串
        if "__interrupt__" in result:
            #HITL 暂停的 payload 是 dict，工具请求列表在 action_requests 键里
            actions = result["__interrupt__"][0].value["action_requests"]
            print(actions)
            for a in actions:
                print(f"[请求调用] {a['name']}({a['args']})")
            answer = input("允许吗？(y/n)")
            if answer.strip().lower() in ("y", "yes"):
                resume_value = {"decisions": [{"type": "approve"}] * len(actions)}
            else:
                resume_value = {"decisions": [{"type": "reject", "message": "用户不同意调用工具"}] * len(actions)}
            result = agent.invoke(Command(resume=resume_value), config)
        messages = result["messages"]
        start = next(i for i in range(len(messages) -1, -1, -1) #第一个是len减一
                if messages[i].type == "human") #next拿到第一个命中就停止，这是在找最新用户消息
        for msg in messages[start:]:
            if msg.type == "ai" and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[调用工具] {tc['name']}({tc['args']})")
            elif msg.type == "tool":
                print(f"[工具返回] {msg.content}")
            elif msg.type == "ai":
                print(f"[最终回答] {msg.content}")
