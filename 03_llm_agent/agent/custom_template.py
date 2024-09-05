from __future__ import annotations
from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from typing import List
from langchain.schema import AgentAction, AgentFinish


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # {'history': '', 'input': '北京市朝阳区的天气', 'intermediate_steps': []}
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # # {'agent_scratchpad': '', 'history': '', 'input': '北京市朝阳区的天气'}
        kwargs["agent_scratchpad"] = thoughts
        # {'agent_scratchpad': '', 'history': '', 'input': '北京市朝阳区的天气', 'tools':
        # '计算器工具: 进行简单的数学运算\n翻译工具: 如果你无法访问互联网，并且需要翻译各种语言，应该使用这个工具\n天气查询工具:
        # 无需访问互联网，使用这个工具查询中国各地未来24小时的天气'}
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # {'agent_scratchpad': '', 'history': '', 'input': '北京市朝阳区的天气',
        # 'tool_names': '计算器工具, 翻译工具, 天气查询工具', 'tools': '计算器工具: 进行简单的数学运算\n
        # 翻译工具: 如果你无法访问互联网，并且需要翻译各种语言，应该使用这个工具\n
        # 天气查询工具: 无需访问互联网，使用这个工具查询中国各地未来24小时的天气'}
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # text='
        # Answer the following questions as best you can. If it is in order,
        # you can use some tools appropriately.You have access to the following tools:
        #
        # 计算器工具: 进行简单的数学运算
        # 翻译工具: 如果你无法访问互联网，并且需要翻译各种语言，应该使用这个工具
        # 天气查询工具: 无需访问互联网，使用这个工具查询中国各地未来24小时的天气
        #
        # Please note that the "天气查询工具" can only be used once since Question begin.
        #
        # Use the following format:
        # Question: the input question you must answer1
        # Thought: you should always think about what to do and what tools to use.
        # Action: the action to take, should be one of [计算器工具, 翻译工具, 天气查询工具]
        # Action Input: the input to the action
        # Observation: the result of the action
        # ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        # Thought: I now know the final answer
        # Final Answer: the final answer to the original input question
        #
        # Begin!
        # history:
        # Question: 北京市朝阳区的天气
        # Thought:'
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    begin: bool = False

    def __init__(self):
        super().__init__()
        self.begin = True

    def parse(self, llm_output: str) -> AgentFinish | tuple[dict[str, str], str] | AgentAction:
        # 封装 LLM 的结果
        if "Final Answer:" in llm_output:
            self.begin = True
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:", 1)[-1].strip()},
                log=llm_output,
            )

        parts = llm_output.split("Action:")
        if len(parts) < 2:
            return AgentFinish(
                return_values={"output": f"调用agent失败: `{llm_output}`"},
                log=llm_output,
            )

        action = parts[1].split("Action Input:")[0].strip()
        action_input = parts[1].split("Action Input:")[1].strip()

        # 失败后再次尝试调用
        try:
            ans = AgentAction(
                tool=action,
                tool_input=action_input.strip(" ").strip('"'),
                log=llm_output
            )
            return ans
        except:
            return AgentFinish(
                return_values={"output": f"调用agent失败: `{llm_output}`"},
                log=llm_output,
            )
