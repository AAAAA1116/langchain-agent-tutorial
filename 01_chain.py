"""
LangChain 官方教程: chains 基础 (LCEL)
对应官方文档: https://docs.langchain.com/oss/python/langchain/quickstart
链 = Prompt | Model | Parser, 用 | 管道符串起来
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# DeepSeek 走 OpenAI 兼容接口
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)

# 1. Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个简洁的中文问答助手，回答尽量不超过三句话。"),
        ("user", "{question}"),
    ]
)

# 2. 用 LCEL 把 组件串成链
chain = prompt | llm | StrOutputParser()

# 3. 调用
if __name__ == "__main__":
    question = "用一句话说明 LangChain 中 chain 的作用。"
    print(f"[Q] {question}")
    answer = chain.invoke({"question": question})
    print(f"[A] {answer}")
