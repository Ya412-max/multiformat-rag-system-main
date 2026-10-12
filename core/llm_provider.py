from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from config.settings import settings

PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "Alibaba Qwen (Tongyi)",
    },
    "custom": {
        "base_url": "",
        "description": "Custom OpenAI‑Compatible API",
    },
}


class LLMProvider:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        # ` ** kwargs`：接收其他额外参数
        **kwargs,
    ):
        self.provider = provider or settings.llm_provider
        self.model_name = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

        if self.provider not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported: {list(PROVIDER_CONFIGS.keys())}"
            )
        if self.provider == "custom":
            # custom不能依赖配置文件兜底，调用方必须手动传入base_url、model
            if not base_url:
                raise ValueError("provider=custom 必须手动传入 base_url 参数，不允许复用全局配置")
            if not model:
                raise ValueError("provider=custom 必须手动传入 model 参数，不允许复用全局配置")

    def get_llm(self) -> BaseChatModel:
        provider_config = PROVIDER_CONFIGS[self.provider]
        effective_base_url = self.base_url or provider_config["base_url"]
        return self._create_openai_compatible_llm(effective_base_url)

    def _create_openai_compatible_llm(self, base_url: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.extra_kwargs,
        )


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> BaseChatModel:
    llm_provider = LLMProvider(
        provider=provider,
        model=model,
        temperature=temperature,
        **kwargs,
    )
    return llm_provider.get_llm()
