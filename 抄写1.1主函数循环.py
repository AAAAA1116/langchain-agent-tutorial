import json
import os
from openai import OpenAI
from datetime import datetime

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

def get_weather(city: str):
    # raise ValueError("moni")
    return {"city" :city, "temperature": 25, "weather": "rain"}

def get_time(city:str):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"city": city,"time": time_str}


tools = [...]


def agent(user_input: str):
    messages = []

    while True:
        response = client.messagechat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
    )

        message = response.choice[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            call_record = (name, json.dumps(args, sort_keys=True))

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
            })

