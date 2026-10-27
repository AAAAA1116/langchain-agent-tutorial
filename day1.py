import json
from openai import OpenAI

client = OpenAI()

# 1. 定义工具
def get_weather(city: str):
    return {"city": city, "temperature": 25, "weather": "晴"}

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "查询指定城市的天气",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False
    },
    "strict": True
}]

# 工具名称 -> Python 函数
tool_map = {"get_weather": get_weather}


# 2. Agent 核心循环
def agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]

    while True:
        response = client.responses.create(
            model="gpt-6",
            input=messages,
            tools=tools
        )

        # 保存模型本轮输出，供下一轮继续对话
        messages += response.output

        # 没有工具调用，说明模型已经完成回答
        if not response.output:
            break

        # 3. 执行工具，并把结果交还给模型
        calls = [item for item in response.output
                 if item.type == "function_call"]

        if not calls:
            print(response.output_text)
            break

        for call in calls:
            name = call.name
            args = json.loads(call.arguments)
            result = tool_map[name](**args)

            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False)
            })

    return response.output_text


# 运行
print(agent("查询北京的天气，并告诉我是否适合出门。"))