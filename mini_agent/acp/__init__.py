"""ACP (Agent Client Protocol) integration for Mini-Agent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp.schema import AgentCapabilities, Implementation, McpCapabilities
from mini_agent.agent import Agent
from mini_agent.cli import add_workspace_tools, initialize_base_tools
from mini_agent.config import Config
from mini_agent.llm import LLMClient
from mini_agent.retry import RetryConfig as RetryConfigBase
from mini_agent.schema import Message
from mini_agent.tools.mcp_loader import MCPServerConnection

logger = logging.getLogger(__name__)


def acp_to_text(blocks: Iterable[Any]) -> str:
    return "\n".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in blocks)


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool args so ACP clients can display useful context."""
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        text = repr(value)
        if len(text) > 48:
            text = f"{text[:45]}..."
        parts.append(f"{key}={text}")
        if len(parts) == 3:  # avoid spamming UI
            break
    return ", ".join(parts)


def _load_system_prompt(config: Config, skill_loader) -> str:
    path = Config.find_config_file(config.agent.system_prompt_path)
    text = path.read_text(encoding="utf-8") if path and path.exists() else "You are a helpful AI assistant."
    if skill_loader:
        meta = skill_loader.get_skills_metadata_prompt()
        if meta:
            text = f"{text.rstrip()}\n\n{meta}"
    return text


@dataclass
class SessionState:
    agent: Agent
    mcp_connections: list[MCPServerConnection] = field(default_factory=list)
    cancelled: bool = False


class MiniMaxACPAgent:
    """ACP bridge for Mini-Agent."""

    def __init__(
        self,
        conn: AgentSideConnection,
        config: Config,
        llm: LLMClient,
        base_tools: list,
        system_prompt: str,
    ):
        self._conn = conn
        self._config = config
        self._llm = llm
        self._base_tools = base_tools
        self._system_prompt = system_prompt
        self._sessions: dict[str, SessionState] = {}

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(loadSession=False, mcpCapabilities=McpCapabilities(http=False, sse=False)),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent (MiniMax M2)", version="0.1.0"),
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        workspace = Path(params.cwd or self._config.agent.workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        requested_mcp = getattr(params, "mcpServers", [])
        logger.info(
            "ACP newSession cwd=%s (resolved=%s). MCP servers requested: %d",
            params.cwd,
            workspace,
            len(requested_mcp),
        )
        tools = list(self._base_tools)
        add_workspace_tools(tools, self._config, workspace)
        extra_tools, connections = await self._connect_mcp_servers(requested_mcp)
        tools.extend(extra_tools)
        agent = Agent(llm_client=self._llm, system_prompt=self._system_prompt, tools=tools, max_steps=self._config.agent.max_steps, workspace_dir=str(workspace))
        self._sessions[session_id] = SessionState(agent=agent, mcp_connections=connections)
        if connections:
            logger.info("Activated %d MCP servers providing %d tools", len(connections), len(extra_tools))
        elif requested_mcp:
            logger.warning("ACP client requested MCP servers but none connected successfully")
        logger.info("Created ACP session %s (workspace=%s)", session_id, workspace)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            return PromptResponse(stopReason="refusal")
        state.cancelled = False
        user_text = acp_to_text(params.prompt)
        state.agent.messages.append(Message(role="user", content=user_text))
        stop_reason = await self._run_turn(state, params.sessionId)
        return PromptResponse(stopReason=stop_reason)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True

    async def _run_turn(self, state: SessionState, session_id: str) -> str:
        agent = state.agent
        for _ in range(agent.max_steps):
            if state.cancelled:
                return "cancelled"
            tool_schemas = [tool.to_schema() for tool in agent.tools.values()]
            try:
                response = await agent.llm.generate(messages=agent.messages, tools=tool_schemas)
            except Exception as exc:
                logger.exception("LLM error")
                await self._send(session_id, update_agent_message(text_block(f"Error: {exc}")))
                return "refusal"
            if state.cancelled:
                return "cancelled"
            if response.thinking:
                await self._send(session_id, update_agent_thought(text_block(response.thinking)))
            if response.content:
                await self._send(session_id, update_agent_message(text_block(response.content)))
            agent.messages.append(Message(role="assistant", content=response.content, thinking=response.thinking, tool_calls=response.tool_calls))
            if not response.tool_calls:
                return "end_turn"
            for call in response.tool_calls:
                name, args = call.function.name, call.function.arguments
                arg_summary = _format_tool_args(args)
                label = f"{name}({arg_summary})" if arg_summary else f"{name}()"
                await self._send(session_id, start_tool_call(call.id, label, kind="execute", raw_input=args))
                tool = agent.tools.get(name)
                if not tool:
                    text, status = f"Unknown tool: {name}", "failed"
                else:
                    try:
                        result = await tool.execute(**args)
                        status = "completed" if result.success else "failed"
                        text = result.content if result.success else result.error or "Tool execution failed"
                    except Exception as exc:
                        status, text = "failed", f"Tool error: {exc}"
                await self._send(session_id, update_tool_call(call.id, status=status, content=[tool_content(text_block(text))], raw_output=text))
                agent.messages.append(Message(role="tool", content=text, tool_call_id=call.id, name=name))
        return "max_turn_requests"

    async def _connect_mcp_servers(self, servers) -> tuple[list, list[MCPServerConnection]]:
        tools, connections = [], []
        if not servers:
            logger.info("ACP client did not request any MCP servers")
            return tools, connections
        for server in servers:
            name = getattr(server, "name", "unknown")
            if getattr(server, "type", "stdio") != "stdio":
                logger.warning("Unsupported MCP server type for %s: %s", name, getattr(server, "type", None))
                continue
            env = {item.name: item.value for item in getattr(server, "env", [])}
            conn = MCPServerConnection(name, server.command, list(server.args or []), env)
            logger.info("Connecting to MCP server '%s' via %s %s", name, server.command, server.args or [])
            if await conn.connect():
                connections.append(conn)
                tools.extend(conn.tools)
                logger.info("Loaded %d tools from MCP server '%s'", len(conn.tools), name)
            else:
                logger.warning("Failed to connect to MCP server '%s'", name)
        return tools, connections

    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))


async def run_acp_server(config: Config | None = None):
    """Run Mini-Agent as an ACP-compatible stdio server."""
    config = config or Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_tools, skill_loader = await initialize_base_tools(config)
    system_prompt = _load_system_prompt(config, skill_loader)
    rcfg = config.llm.retry
    retry = RetryConfigBase(enabled=rcfg.enabled, max_retries=rcfg.max_retries, initial_delay=rcfg.initial_delay, max_delay=rcfg.max_delay, exponential_base=rcfg.exponential_base)
    llm = LLMClient(api_key=config.llm.api_key, api_base=config.llm.api_base, model=config.llm.model, retry_config=retry)
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, config, llm, base_tools, system_prompt), writer, reader)
    logger.info("Mini-Agent ACP server running")
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(run_acp_server())


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
