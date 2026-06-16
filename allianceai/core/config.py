"""
Application-level configuration via environment variables or a .env file.

Pydantic-settings validates types at startup so misconfigured environments
fail loudly rather than silently producing wrong results.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional paid API keys — the system works without them (falls back to yfinance/EDGAR).
    alpha_vantage_key: str = Field(default="", description="Alpha Vantage API key (optional).")
    fred_api_key: str = Field(default="", description="FRED macroeconomic data key (optional).")

    # Forecasting
    forecast_horizon_quarters: int = Field(default=8, ge=1, le=40)

    # ------------------------------------------------------------------
    # LLM highlights — short metric callouts.
    # Provider is modular:
    #   "anthropic"  → Claude API directly (needs ANTHROPIC_API_KEY)
    #   "openrouter" → any model via OpenRouter (needs OPENROUTER_API_KEY)
    #   "auto"       → anthropic if its key is set, else openrouter, else fallback
    #   "none"       → always use the rule-based fallback (no API calls)
    # ------------------------------------------------------------------
    llm_provider: str = Field(default="auto")
    anthropic_api_key: str = Field(default="")
    highlights_model: str = Field(default="claude-haiku-4-5")
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="anthropic/claude-haiku-4.5")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # LLM narrative generation
    # A small open-source model that runs on CPU.  Replace with a larger model
    # (e.g. 'mistralai/Mistral-7B-Instruct-v0.3') if you have a GPU.
    narrative_model_id: str = Field(default="facebook/opt-125m")
    narrative_max_new_tokens: int = Field(default=300)

    # DuckDB persistence
    db_path: str = Field(default="allianceai.duckdb")

    # Data staleness: re-fetch if cached data is older than this many hours.
    cache_ttl_hours: int = Field(default=24)

    # SEC EDGAR — a descriptive User-Agent with contact info is required by
    # SEC fair-access rules (https://www.sec.gov/os/accessing-edgar-data).
    edgar_user_agent: str = Field(
        default="AllianceAI research tool (contact: user@example.com)",
        description="User-Agent header sent to SEC EDGAR endpoints.",
    )
    # Extend statements with EDGAR history when yfinance returns fewer periods.
    edgar_min_periods: int = Field(default=12)


settings = Settings()
