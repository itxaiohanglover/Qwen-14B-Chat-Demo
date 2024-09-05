PROMPT_TEMPLATES = {}

PROMPT_TEMPLATES["agent_chat"] = {
    "Qwen":
        """
        Answer the following questions as best you can. If it is in order, you can use some tools appropriately.You have access to the following tools:

        {tools}

        Use the following format:
        Question: the input question you must answer1
        Thought: you should always think about what to do and what tools to use.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question


        Begin!
        history:
        {history}
        Question: {input}
        Thought: {agent_scratchpad}
        """
}
