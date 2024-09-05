from langchain.tools import Tool
from agent.tools import *

tools = [
    Tool.from_function(
        func=calculate,
        name="计算器工具",
        description="进行简单的数学运算"
    ),
    Tool.from_function(
        func=translate,
        name="翻译工具",
        description="如果你无法访问互联网，并且需要翻译各种语言，应该使用这个工具"
    ),
    Tool.from_function(
        func=weathercheck,
        name="天气查询工具",
        description="无需访问互联网，使用这个工具查询中国各地未来24小时的天气",
    )
]

tool_names = [tool.name for tool in tools]
