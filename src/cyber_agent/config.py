from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    openai_api_key: str = Field(default='sk-default', validation_alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-5.4', validation_alias='OPENAI_MODEL')   # 默认值
    openai_base_url: str | None = Field(default=None, validation_alias='OPENAI_BASE_URL')

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    
    def get_service(self) -> str:
        # 从环境变量读取，例如 "openai"
        return Field(default='openai', validation_alias='SERVICE_NAME')

    def get_chat_openai_kwargs(self, service_name: str) -> dict:
        base_kwargs = {
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        if service_name == "openai":
            kwargs = {
                "model": self.openai_model,
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
            }
        else:
            raise ValueError(f"Unknown service: {service_name}")
        # 合并公共参数
        kwargs.update(base_kwargs)
        # 过滤 None
        return {k: v for k, v in kwargs.items() if v is not None}
        
        

# 创建全局单例供其他模块使用
settings = Settings()