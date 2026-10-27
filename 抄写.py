import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# ---- 定义一个外部工具: 四则运算计算器 ----
@tool
def cal(exp: str) -> str:
    """计算一个四则运算表达式并返回结果。输入示例: (3+5)*12"""
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Exp):
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return (allowed_ops[type(node.op)](
                _eval(node.left),
                _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式: {exp!r}")

    result = _eval(ast.parse(exp, mode="eval"))
    return str(result)

TOOLS = [cal]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="你是一个中文问答助手。遇到数学计算时必须调用 calculator 工具，不要心算。",
)

if __name__ =="__main__":
    questions = [
        "LangChain 是做什么的？一句话回答。",
        "请问 (128 + 512) * 3 / 4 等于多少？",
    ]
    for q in questions:
        print(f"[Q] {q}")
        result = agent.invoke({"message": [{"role": "user","content": q}]})

        for msg in result["message"]:
            role = msg.type
            if role == "ai" and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[工具调用] {tc['name']} ({tc['args']})")
            elif role == "ai" and msg.content:
                print(f"[A] {msg.content}")
            elif role == "tool":
                print(f"[工具调用] {msg['name']} ({msg['args']})")
