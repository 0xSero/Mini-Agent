"""ACP (Agent Client Protocol) integration for Mini-Agent."""
import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    AgentSideConnection, InitializeRequest, InitializeResponse, NewSessionRequest,
    NewSessionResponse, PromptRequest, PromptResponse, CancelNotification,
    session_notification, text_block, update_agent_message, update_agent_thought,
    start_tool_call, update_tool_call, tool_content, PROTOCOL_VERSION, stdio_streams,
)
from acp.schema import AgentCapabilities, Implementation
from mini_agent.agent import Agent
from mini_agent.llm import LLMClient
from mini_agent.schema import Message

logger = logging.getLogger(__name__)


def acp_to_text(content: list[Any]) -> str:
    """Convert ACP content blocks to plain text."""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


class MiniMaxACPAgent:
    """Minimal ACP agent wrapping Mini-Agent core."""
    def __init__(self, conn: AgentSideConnection, llm: LLMClient, tools: list, system_prompt: str):
        self._conn, self._llm, self._tools, self._system_prompt = conn, llm, tools, system_prompt
        self._sessions: dict[str, Agent] = {}

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(supportsLoadSession=False, supportsSetMode=False),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent (MiniMax M2)", version="0.1.0"),
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        self._sessions[session_id] = Agent(self._llm, self._tools, self._system_prompt, workspace_dir=params.cwd or ".")
        logger.info("Created session: %s", session_id)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        agent = self._sessions.get(params.sessionId)
        if not agent:
            return PromptResponse(stopReason="error")

        user_text = acp_to_text(params.prompt)
        agent.messages.append(Message(role="user", content=user_text))
        await self._send(params.sessionId, update_agent_message(text_block(f"User: {user_text}")))

        for _ in range(10):
            tool_schemas = [t.to_schema() for t in agent.tools.values()]

            try:
                response = await agent.llm.generate(messages=agent.messages, tools=tool_schemas)
            except Exception as e:
                await self._send(params.sessionId, update_agent_message(text_block(f"Error: {e}")))
                return PromptResponse(stopReason="error")

            if response.thinking:
                await self._send(params.sessionId, update_agent_thought(text_block(response.thinking)))

            if response.content:
                await self._send(params.sessionId, update_agent_message(text_block(response.content)))

            agent.messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:
                return PromptResponse(stopReason="end_turn")

            for tc in response.tool_calls:
                await self._send(
                    params.sessionId,
                    start_tool_call(tc.id, f"Executing {tc.function.name}", name=tc.function.name, arguments=tc.function.arguments),
                )

                tool = agent.tools.get(tc.function.name)
                if not tool:
                    error = f"Unknown tool: {tc.function.name}"
                    await self._send(params.sessionId, update_tool_call(tc.id, status="failed", content=[tool_content(text_block(error))]))
                    agent.messages.append(Message(role="tool", content=error, tool_call_id=tc.id, name=tc.function.name))
                    continue

                try:
                    result = await tool.execute(**tc.function.arguments)
                    status = "completed" if result.success else "failed"
                    text = result.content if result.success else result.error or "Unknown error"
                    await self._send(params.sessionId, update_tool_call(tc.id, status=status, content=[tool_content(text_block(text))]))
                    agent.messages.append(Message(role="tool", content=text, tool_call_id=tc.id, name=tc.function.name))
                except Exception as e:
                    error = f"Tool error: {e}"
                    await self._send(params.sessionId, update_tool_call(tc.id, status="failed", content=[tool_content(text_block(error))]))
                    agent.messages.append(Message(role="tool", content=error, tool_call_id=tc.id, name=tc.function.name))

        return PromptResponse(stopReason="max_steps")

    async def cancel(self, params: CancelNotification) -> None:
        logger.info("Cancel requested: %s", params.sessionId)
    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))


async def run_acp_server(config=None):
    """Run Mini-Agent as an ACP server."""
    from mini_agent.config import Config
    from mini_agent.tools.bash_tool import BashTool, BashOutputTool, BashKillTool
    from mini_agent.tools.file_tools import ReadTool, WriteTool, EditTool
    config = config or Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting Mini-Agent ACP server")
    llm = LLMClient(api_key=config.llm.api_key, api_base=config.llm.api_base, model=config.llm.model, retry_config=config.llm.retry)
    tools = []
    if config.tools.enable_bash:
        tools.extend([BashTool(), BashOutputTool(), BashKillTool()])
    if config.tools.enable_file_tools:
        workspace = Path(config.agent.workspace_dir).expanduser()
        tools.extend([ReadTool(workspace), WriteTool(workspace), EditTool(workspace)])
    try:
        with open(config.agent.system_prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        system_prompt = "You are a helpful AI assistant."
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, llm, tools, system_prompt), writer, reader)
    logger.info("ACP server running")
    await asyncio.Event().wait()

def main():
    """Entry point for mini-agent-acp CLI command."""
    asyncio.run(run_acp_server())


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
