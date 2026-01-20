"""
Services configuration module for easy STT, LLM, and TTS initialization.
Handles environment variable loading and provides factory functions.
"""

import os
from typing import Optional

# Import plugins at module level to ensure registration on main thread
from livekit.plugins import assemblyai, cartesia, mistralai, silero, openai

IGNORE_WORDS = [
    "yeah", "ok", "hmm", "aha", "right", "sure", "yes", "yep", "uh-huh"
]


class ServicesConfig:
    """Configuration for AI services with environment variable fallbacks."""
    
    def __init__(self):
        # STT Configuration
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        
        # LLM Configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "mistral").lower()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # TTS Configuration
        self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        self.cartesia_voice = os.getenv("CARTESIA_VOICE", "79a125e8-cd45-4c13-8a67-188112422040")
        
        # VAD Configuration
        self.vad_model = os.getenv("VAD_MODEL", "silero")
        
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that required API keys are configured."""
        missing_keys = []
        
        if not self.assemblyai_api_key:
            missing_keys.append("ASSEMBLYAI_API_KEY")
        if self.llm_provider == "mistral" and not self.mistral_api_key:
            missing_keys.append("MISTRAL_API_KEY")
        if not self.cartesia_api_key:
            missing_keys.append("CARTESIA_API_KEY")
        
        if missing_keys:
            print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
            print("Set these as environment variables for the agent to work properly.")
    
    def create_stt(self):
        """Create and return STT instance."""
        return assemblyai.STT()
    
    def create_llm(self):
        """Create and return LLM instance (Mistral or OpenAI)."""
        if self.llm_provider == "mistral":
            if not self.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY must be set for provider 'mistral'")
            return mistralai.LLM(model=self.mistral_model, api_key=self.mistral_api_key)

        if self.llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY must be set for provider 'openai'")
            return openai.LLM(model=self.openai_model)

        raise ValueError(f"Unknown LLM_PROVIDER: {self.llm_provider}")
    
    def create_tts(self):
        """Create and return TTS instance."""
        return cartesia.TTS()
    
    def create_vad(self):
        """Create and return VAD instance."""
        if self.vad_model.lower() == "silero":
            return silero.VAD.load()
        else:
            raise ValueError(f"Unknown VAD model: {self.vad_model}")


# Global instance for easy access
_config: Optional[ServicesConfig] = None


def get_config() -> ServicesConfig:
    """Get or create the global services configuration."""
    global _config
    if _config is None:
        _config = ServicesConfig()
    return _config


def create_services() -> tuple:
    """
    Convenience function to create all services at once.
    
    Returns:
        tuple: (vad, stt, llm, tts)
    """
    config = get_config()
    return (
        config.create_vad(),
        config.create_stt(),
        config.create_llm(),
        config.create_tts(),
    )


def create_agent_session_kwargs() -> dict:
    """
    Get ready-to-use kwargs for AgentSession initialization.
    
    Returns:
        dict: Keyword arguments for AgentSession
    """
    vad, stt, llm, tts = create_services()
    
    return {
        "vad": vad,
        "stt": stt,
        "llm": llm,
        "tts": tts,
        # Interruption handling configuration
        "allow_interruptions": True,
        "discard_audio_if_uninterruptible": True,  # Discard buffered audio when interrupted
        "min_interruption_duration": 0.05,  # 50ms minimum speech to interrupt (aggressive)
        "min_interruption_words": 1,        # Allow single-word interrupts
        "false_interruption_timeout": None, # Disable auto-resume
        "resume_false_interruption": False, # Don't automatically resume
        "user_away_timeout": None,          # Don't auto-timeout user sessions
        "min_endpointing_delay": 0.2,       # Faster end-of-turn detection
    }

