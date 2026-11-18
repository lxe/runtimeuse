"""
Orble Quickstart -- Run Claude Code in an Orble sandbox using runtimeuse.

Orble speaks the E2B API, so this uses the e2b SDK pointed at Orble. Orble's
default template already has Node.js, Claude Code and Docker, so there is no
template to build. The client reaches runtimeuse through a public port, which
Orble opens for accounts that have added credit.

Setup:
  pip install runtimeuse-client e2b

Environment variables:
  E2B_API_KEY        - your Orble API key (https://orble.tech/settings?tab=api)
  E2B_DOMAIN         - orble.tech
  ANTHROPIC_API_KEY  - your Anthropic API key

Usage:
  python orble-quickstart.py
"""

from __future__ import annotations

import asyncio
import os

from e2b import Sandbox

from runtimeuse_client import (
    RuntimeEnvironmentDownloadableInterface,
    RuntimeUseClient,
    QueryOptions,
    AssistantMessageInterface,
    TextResult,
)


def _get_env_or_fail(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def create_sandbox() -> tuple[Sandbox, str]:
    """Create an Orble sandbox, start runtimeuse + Claude Code in it, and return (sandbox, ws_url)."""

    _get_env_or_fail("E2B_API_KEY")
    _get_env_or_fail("E2B_DOMAIN")
    anthropic_api_key = _get_env_or_fail("ANTHROPIC_API_KEY")

    print("Creating Orble sandbox...")
    sandbox = Sandbox.create(timeout=600)

    sandbox.commands.run(
        "npx -y runtimeuse@latest --agent claude",
        background=True,
        envs={
            "ANTHROPIC_API_KEY": anthropic_api_key,
            "IS_SANDBOX": "1",
            "CLAUDE_SKIP_ROOT_CHECK": "1",
        },
    )

    print("Waiting for runtimeuse server to start...")
    sandbox.commands.run(
        "timeout 120 sh -c 'until curl -s -o /dev/null localhost:8080; do sleep 1; done'",
        timeout=130,
    )

    ws_url = f"wss://{sandbox.get_host(8080)}"
    print(f"Sandbox ready at {ws_url}")

    return sandbox, ws_url


async def main() -> None:
    sandbox, ws_url = create_sandbox()
    try:
        client = RuntimeUseClient(ws_url=ws_url)
        print(f"Connected to {ws_url}")

        async def on_message(msg: AssistantMessageInterface) -> None:
            for block in msg.text_blocks:
                print(f"[assistant] {block}")

        prompt = "Summarize the contents of the codex repository and list your favorite file in the repository."
        print(f"Sending query: {prompt}")

        result = await client.query(
            prompt=prompt,
            options=QueryOptions(
                system_prompt="You are a helpful assistant.",
                model="claude-sonnet-4-20250514",
                on_assistant_message=on_message,
                pre_agent_downloadables=[
                    RuntimeEnvironmentDownloadableInterface(
                        download_url="https://github.com/openai/codex/archive/refs/heads/main.zip",
                        working_dir=".",
                    )
                ],
            ),
        )

        print("\n--- Final Result ---")
        assert isinstance(result.data, TextResult)
        print(result.data.text)
    finally:
        sandbox.kill()
        print("Sandbox terminated.")


if __name__ == "__main__":
    asyncio.run(main())
