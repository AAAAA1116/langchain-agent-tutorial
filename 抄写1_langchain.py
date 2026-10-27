import os
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

"""
用 LangChain create_agent 重写 抄写1.py 的同等功能
"""

# 定义工具
# 手写：tools = [{"type": "function", "function": {...}}] + tool_map = {...}
# @tool：docstring 就是 description，参数签名自动生成 parameters schema
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当天天气信息。""" #这是docstring
    return f"城市：{city}，气温：25℃，天气：雨天"


@tool
def get_time(city: str) -> str:
    """查询指定城市的当前时间。"""
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"城市：{city}，当前时间：{time_str}"

# 初始化 LLM
# 手写：client = OpenAI(...) 然后在循环里手动指定 model、messages、tools
# create_agent：传入 llm 对象，自动组装参数调用 API
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 创建 agent：封装了以下逻辑
#        a：while 循环 + 手动 messages.append() 追加 user/ai/tool 消息 -> 自动循环 + 自动维护历史
#        b：手动 tool_map[name](**args) 查表调用 -> 自动通过工具名找 @tool 函数并调用
#        c：手动 messages.append({"role": "tool", ...}) 塞回结果 -> 自动包装成 ToolMessage
agent = create_agent(
    model=llm,
    tools=[get_weather, get_time],
    system_prompt="你是一个中文生活助手，可以查天气和查时间。回答时请结合工具返回的信息给出实用建议。",
)

# 调用
# 手写：if not message.tool_calls: return（自己判断是否结束）
# create_agent：没有 tool_calls 自动走 END，recursion_limit 等价于 max_rounds
#        a：手动 try/except 捕获工具报错 + 返回 error dict 给 LLM -> 内置捕获自动返回 ToolMessage
#        b：手动 call_history 连续 3 次相同则终止 -> 只有 recursion_limit 兜底
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "现在北京几点了，适合出门吗"}]},
        config={"recursion_limit": 10},  # 超过最大轮次强制停止
    )

    # 打印完整轨迹
    # 手写：name = call.function.name  args = json.loads(call.function.arguments)
    # result["messages"]：LangChain 已解析好，直接用 tc["name"]、tc["args"]
    for msg in result["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [调用工具] {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"  [工具返回] {msg.content}")
        elif msg.type == "ai":
            print(f"  [回答] {msg.content}")
