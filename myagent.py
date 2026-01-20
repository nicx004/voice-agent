from dotenv import load_dotenv
load_dotenv()  # reads .env and sets environment variables

import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents import llm
from services import create_agent_session_kwargs

class MyAgent(Agent):
    
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            # LLM is configured separately in entrypoint via services module
        )

    async def user_fn(self, user_msg: llm.ChatMessage) -> str:
        """Handle user input. If user says 'stop', return empty to halt."""
        text = user_msg.content.lower().strip()
        
        # If user says "stop" or "halt" or similar, immediately return empty
        # This tells the framework not to generate a response
        if text in ("stop", "halt", "quit", "exit", "stop talking", "shut up"):
            return ""
        
        # Otherwise, let the LLM handle it normally by returning None
        # which causes the framework to use the LLM
        return None

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Get all services pre-configured with interruption handling
    session = AgentSession(**create_agent_session_kwargs())
    
    await session.start(agent=MyAgent(), room=ctx.room)

if __name__ == "__main__":
    # Set default LiveKit credentials for local development if not in environment
    # For console mode with fake_job=True, these are only used for validation
    # and don't need to connect to a real server
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=os.getenv("LIVEKIT_URL") or "ws://localhost:7880",
        api_key=os.getenv("LIVEKIT_API_KEY") or "devkey",
        api_secret=os.getenv("LIVEKIT_API_SECRET") or "secret",
    )
    cli.run_app(worker_options)
