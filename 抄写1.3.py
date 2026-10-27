import os
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

#把 C:\Users\A\WorkBuddy\2026-10-17-13-18-20\langchain-agent-tutorial\抄写1.3.py 改写成完整的多轮对话版本：config 中补上 configurable.thread_id="1"，在 if __name__ == "__main__" 下套 while True + input 循环，把 next 生成器表达式的 if 收回括号内，并保留 messages[start:] 切片打印逻辑，确保脚本可直接运行。

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

# C:\Users\A\WorkBuddy\2026-10-17-13-18-20\langchain-agent-tutorial\.venv\Scripts\python.exe "C:\Users\A\WorkBuddy\2026-10-17-13-18-20\langchain-agent-tutorial\抄写1.3.py"
# 你:北京天气
# [调用工具] get_weather({'city': '北京'})
# [工具返回] 城市：北京, 气温25度，雨天
# [最终回答] 北京当前天气：气温 25℃，雨天。出门记得带伞哦 ☔
# 你:几点了
# [调用工具] get_time({'city': '北京'})
# [工具返回] 北京现在2026-10-27 09:27:19
# [最终回答] 北京时间现在是 2026年10月27日 09:27:19。
# 你:q
#
# 进程已结束，退出代码为 0
