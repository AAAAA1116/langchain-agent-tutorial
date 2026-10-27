import os
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

#用 create_agent 写最简 agent 的单轮对话版本：两个 @tool 装饰的工具（查天气、查时间），ChatOpenAI 接 DeepSeek，invoke 一次后遍历 result["messages"]，按 ai+tool_calls / tool / ai 三种类型打印调用过程与最终回答，config 里设 recursion_limit=10。


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
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages":[{"role": "user", "content": "现在北京几点，适合出门吗"}]},
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[调用工具] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"[工具返回] {msg.content}")
        elif msg.type == "ai":
            print(f"[最终回答] {msg.content}")
