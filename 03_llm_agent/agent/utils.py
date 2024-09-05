from typing import Optional


def get_prompt_template(type: str, name: str) -> Optional[str]:
    from configs import prompt_config
    import importlib
    importlib.reload(prompt_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)
