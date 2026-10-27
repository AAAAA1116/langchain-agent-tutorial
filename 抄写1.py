import json
import os
from openai import OpenAI
from datetime import datetime

""" 
1. 准备：messages(对话历史) + tools(工具清单)
2. 循环：
        a：把messages和tools发给llm
        b：如果toolcall为空就意味着不用调工具 直接返回结果
        c：toolcall不为空就执行代码
        d：再把结果塞到messages ->a发给llm
 """

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 定义工具
def get_weather(city: str):
    # raise ValueError("moni")
    return {"city" :city, "temperature": 25, "weather": "rain"}

def get_time(city:str):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"city": city,"time": time_str}


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当天天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
},

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "查询指定城市的当前时间",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

#工具名称 -> Python 函数
tool_map = {"get_weather": get_weather, "get_time": get_time}

def agent(user_input: str, max_rounds: int = 10):
    messages = [{"role": "user", "content": user_input}]
    round_count = 0
    call_history = []  # 记录每次工具调用，用于死循环检测

    while round_count < max_rounds:
        round_count += 1
        print(f"[Round {round_count}]")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print(message.content)
            return message.content

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            call_record = (name, json.dumps(args, sort_keys=True))

            # 死循环检测：同一调用连续出现 3 次则终止
            call_history.append(call_record)
            if len(call_history) >= 3 and call_history[-1] == call_history[-2] == call_history[-3]:
                print(f"[死循环检测] 工具 {name} 连续调用 3 次，强制停止")
                return "检测到重复调用，已强制停止"

            # 工具报错处理
            try:
                result = tool_map[name](**args)
            except Exception as e:
                result = {"error": f"工具 {name} 执行失败: {str(e)}"}
                print(f"[工具错误] {name}({args}): {e}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # 超过最大轮次强制停止
    print(f"达到最大轮次 {max_rounds} 已停止")
    return None


agent("现在北京几点了，适合出门吗") #      ,max_rounds=2