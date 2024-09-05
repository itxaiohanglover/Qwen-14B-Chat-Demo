from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import warnings
from typing import Dict
from common.constants import *

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import requests
from typing import List, Any, Optional
from langchain.prompts import PromptTemplate
from agent.model_contain import model_container

_PROMPT_TEMPLATE = """
用户会提出一个关于天气的问题，你的目标是拆分出用户问题中的区，市 并按照我提供的工具回答。
例如 用户提出的问题是: 上海浦东未来1小时天气情况？
则 提取的市和区是: 上海 浦东
如果用户提出的问题是: 上海未来1小时天气情况？
则 提取的市和区是: 上海 None
请注意以下内容:
1. 如果你没有找到区的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定市 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的市和区，中间用空格隔开}}
```
... weather(市 区)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：o
问题: 上海浦东未来1小时天气情况？


```text
上海 浦东
```
...weather(上海 浦东)...

```output
预报时间: 1小时后
具体时间: 今天 18:00
温度: 24°C
天气: 多云
风向: 西南风
风速: 7级
湿度: 88%
降水概率: 16%

Answer: 上海浦东一小时后的天气是多云。

现在，这是我的问题：

问题: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def get_city_info(location, adm, key):
    base_url = 'https://geoapi.qweather.com/v2/city/lookup?'
    params = {'location': location, 'adm': adm, 'key': key}
    response = requests.get(base_url, params=params)
    data = response.json()
    return data


def format_weather_data(data):
    daily_forecast = data['daily']
    formatted_data = ''
    for daily_info in daily_forecast[:1]:
        date = daily_info['fxDate']
        text_day = daily_info['textDay']
        min_temp = daily_info['tempMin']
        max_temp = daily_info['tempMax']
        # 获取预报时间的时区
        formatted_data += '日期: ' + date + '°C\n'
        formatted_data += '天气: ' + text_day + '°C\n'
        formatted_data += '最低温度: ' + min_temp + '°C\n'
        formatted_data += '最高温度: ' + min_temp + '°C\n'
        formatted_data += '\n\n'
    return formatted_data


def get_weather(key, location_id, place):
    url = "https://api.qweather.com/v7/weather/3d?"
    params = {
        'location': location_id,
        'key': key,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return format_weather_data(data)


def split_query(query):
    parts = query.split()
    adm = parts[0]
    location = parts[1] if parts[1] != 'None' else adm
    return location, adm


def weather(query):
    location, adm = split_query(query)
    key = WEATHER_KEY
    if key == "":
        return "请先在代码中填入和风天气API Key"
    try:
        # 获取 city_info
        city_info = get_city_info(location=location, adm=adm, key=key)
        location_id = city_info['location'][0]['id']
        place = adm + "市" + location + "区"

        # 根据 city_info 查询天气情况
        weather_data = get_weather(key=key, location_id=location_id, place=place)
        print('###' * 100)
        print(weather_data)
        return weather_data + "以上是查询到的天气信息，请你查收\n"
    except KeyError:
        try:
            city_info = get_city_info(location=adm, adm=adm, key=key)
            location_id = city_info['location'][0]['id']
            place = adm + "市"
            weather_data = get_weather(key=key, location_id=location_id, place=place)
            return weather_data + "重要提醒：用户提供的市和区中，区的信息不存在，或者出现错别字，因此该信息是关于市的天气，请你查收\n"
        except KeyError:
            return "输入的地区不存在，无法提供天气预报"


class LLMWeatherChain(Chain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = PROMPT
    input_key: str = "question"
    output_key: str = "answer"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMWeatherChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        try:
            # 调用和风天气的接口获取天气
            output = weather(expression)
        except Exception as e:
            output = "输入的信息有误，请再次尝试"
        return output

    def _process_llm_result(
            self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        llm_output = llm_output.strip()
        # 解析结果
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            # 执行表达式
            output = self._evaluate_expression(expression)
            # 输出打印信息
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            # 拼接结果
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: f"输入的格式不对: {llm_output},应该输入 (市 区)的组合"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)

        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # CallbackManagerForChainRun.get_noop_manager(): 返回一个不执行任何操作的 manager
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # run_manager.on_text: 当收到 text 时就执行
        _run_manager.on_text(inputs[self.input_key])
        # 获取模型预测结果
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        # 调用 LLM 结果处理方法
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_weather_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMWeatherChain:
        # 基于传入模型创建 Chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # 调用自身的 call 方法
        return cls(llm_chain=llm_chain, **kwargs)


def weathercheck(query: str):
    model = model_container.MODEL
    llm_weather = LLMWeatherChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_weather.run(query)
    return ans


if __name__ == '__main__':
    result = weathercheck("苏州姑苏区今晚热不热？")
    print(result)
