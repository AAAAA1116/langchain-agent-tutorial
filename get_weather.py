import json
import os
from datetime import datetime

from openai import OpenAI
from rich import print as rprint

# 工具定义：告诉模型有哪些函数可以调用
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {     # 参数定义（JSON Schema 格式）
            "type": "object",   # 参数是一个对象
            "properties": {   # 对象有哪些属性
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"],  # 必填参数列表
        },
    },
}

# 时间工具：获取当前时间
time_tool = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

# 记忆工具：存储和读取信息
store_set_tool = {
    "type": "function",
    "function": {
        "name": "store_set",
        "description": "存储键值对到记忆，用于记住用户偏好或重要信息",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "键名"},
                "value": {"type": "string", "description": "要存储的值"}
            },
            "required": ["key", "value"],
        },
    },
}

store_get_tool = {
    "type": "function",
    "function": {
        "name": "store_get",
        "description": "从记忆中读取之前存储的信息",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "要读取的键名"}
            },
            "required": ["key"],
        },
    },
}

def get_weather(city):
    return {"city": city, "temp": "12°C", "cond": "多云"}

def get_time():
    return {"time": datetime.now().strftime("%H:%M")}

# 记忆存储：模块级字典
STORE = {}

def store_set(key: str, value: str) -> str:
    """存储键值对到记忆"""
    STORE[key] = value
    return f"已存储: {key} = {value}"

def store_get(key: str) -> str:
    """从记忆中读取键值"""
    if key in STORE:
        return STORE[key]
    return f"未找到键: {key}"

# 工具函数注册表：函数名 -> 实际函数对象，用于动态调用
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_time": get_time,
    "store_set": store_set,
    "store_get": store_get,
}

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

def agent(user_input: str, user_id: str) -> str:
    """
    Agent 函数：接收用户输入，自动循环调用工具直到模型返回纯文本。
    
    Args:
        user_input: 用户的问题或指令
        user_id: 用户 ID（可用于维护独立会话历史）
    
    Returns:
        模型的最终回答（字符串）
    """
    # 初始化对话历史
    messages = [{"role": "user", "content": user_input}]

    # 自动循环：模型要调工具就执行，直到返回纯文本
    while True:
        # 发起 API 调用
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[weather_tool, time_tool, store_set_tool, store_get_tool],  # 传入可用工具列表
        )
        message = response.choices[0].message
        messages.append(message)  # 把模型返回的消息追加到历史

        # 检查是否有工具调用
        if not message.tool_calls:
            # 没有 tool_calls，说明模型回了纯文本，返回回答
            return message.content

        # 有 tool_calls，遍历执行每个工具
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name  # 从模型返回中取出函数名
            func_args = json.loads(tool_call.function.arguments)  # 解析参数（JSON 字符串 -> dict）
            
            # 执行工具，捕获异常
            try:
                result = TOOL_REGISTRY[func_name](**func_args)  # 查表拿到函数对象，解包参数后调用
                result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                # 工具执行失败，把错误信息返回给模型
                result_str = json.dumps({"error": str(e)}, ensure_ascii=False)
            
            # 把工具执行结果追加到消息历史，role 必须是 "tool"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # 用 id 关联到对应的 tool_call
                "content": result_str,  # 结果序列化成 JSON 字符串
            })


if __name__ == "__main__":
    # 测试 agent 函数（带记忆功能）
    print("=== 测试 1: 记住用户偏好 ===")
    result1 = agent("记住：我的名字是小明，我喜欢吃苹果", "user_001")
    rprint(result1)
    
    print("\n=== 测试 2: 调用天气和时间 ===")
    result2 = agent("北京现在适合穿什么出门？考虑天气和时间", "user_001")
    rprint(result2)
    
    print("\n=== 测试 3: 读取记忆 ===")
    result3 = agent("我叫什么名字？我喜欢吃什么？", "user_001")
    rprint(result3)
    
    print("\n=== 当前 STORE 内容 ===")
    rprint(STORE)
