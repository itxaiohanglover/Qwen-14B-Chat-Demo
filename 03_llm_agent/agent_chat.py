from langchain.memory import ConversationBufferWindowMemory
from agent.tools_select import tool_names
from agent.tools_select import tools
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from agent.custom_template import CustomOutputParser, CustomPromptTemplate
from fastapi import Body
from configs.model_config import LLM_MODEL, TEMPERATURE, HISTORY_LEN
from agent.utils import get_prompt_template
from langchain.chains import LLMChain
from typing import List
from agent.chat.history import History
from langchain.chat_models import ChatOpenAI
from agent.model_contain import model_container


def agent_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: List[History] = Body([],
                                             description="历史对话",
                                             examples=[[
                                                 {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                 {"role": "assistant", "content": "虎头虎脑"}]]
                                             ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
               prompt_name: str = Body("agent_chat",
                                       description="使用的prompt模板名称"),
               ):
    # 创建 model
    model = ChatOpenAI(
        streaming=stream,
        verbose=True,
        callbacks=[],
        openai_api_key="none",
        openai_api_base="http://127.0.0.1:8000/v1",
        model_name=model_name,
        temperature=temperature
    )

    # 配置全局model
    model_container.MODEL = model

    # 配置prompt模板
    prompt_template = CustomPromptTemplate(
        template=get_prompt_template("agent_chat", 'Qwen'),
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"]
    )

    # 配置输出格式化
    output_parser = CustomOutputParser()

    # 创建LLMChain
    llm_chain = LLMChain(llm=model, prompt=prompt_template)

    # 创建agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["Observation:", "Observation:\n", "<|im_end|>"],
        allowed_tools=tool_names,
    )

    # 把history转成agent的memory
    memory = ConversationBufferWindowMemory(k=HISTORY_LEN * 2)

    for message in history:
        # 检查消息的角色
        if message.role == 'user':
            # 添加用户消息
            memory.chat_memory.add_user_message(message.content)
        else:
            # 添加AI消息
            memory.chat_memory.add_ai_message(message.content)

    # 创建Executor
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True,
                                                        memory=memory,
                                                        )

    # 执行Executor
    response = agent_executor(query, include_run_info=True)
    return response


if __name__ == '__main__':
    data = {
        # "query": '使用合适的工具翻译：我用周末的时间学习人工智能',
        # "query": '北京市朝阳区的温度',
        "query": '100的平方根结果的三次方',
        "history": [],
        "stream": True,
        "model_name": 'Qwen-14B-Chat',
        "temperature": 0.01,
    }
    response = agent_chat(**data)
