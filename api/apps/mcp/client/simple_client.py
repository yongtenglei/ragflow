#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED = "Session is not initialized. Call connect() first"
ERROR_MCP_CONNECT_SERVER = "Cannot connect to server"
ERROR_MCP__SERVER_SCRIPT_TYPE_NOT_SUPPORT = "Server script must be a .py or .js file"


class SimpleMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError(ERROR_MCP__SERVER_SCRIPT_TYPE_NOT_SUPPORT)

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write, read_timeout_seconds=timedelta(seconds=10)))
        self.server_script_path = server_script_path

        if self.session:
            print("connect to server ok")
            print(f"{self.session=}")

        try:
            await self.session.initialize()
            tool_list = await self.session.list_tools()
            self.tools = tool_list.tools
        except Exception:
            raise ValueError(ERROR_MCP_CONNECT_SERVER)

    async def list_prompts(self):
        if self.session:
            return await self.session.list_prompts()
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def get_prompt(self, prompt_name, arguments=None):
        if self.session:
            return await self.session.get_prompt(prompt_name, arguments or {})
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def list_resources(self):
        if self.session:
            return await self.session.list_resources()
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def list_tools(self):
        if self.session:
            return await self.session.list_tools()
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def read_resource(self, resource_path):
        if self.session:
            return await self.session.read_resource(resource_path)
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def call_tool(self, tool_name, arguments=None):
        if self.session:
            return await self.session.call_tool(tool_name, arguments or {})
        else:
            raise RuntimeError(ERROR_MCP_CLIENT_SESSION_NOT_INITIALIZED)

    async def cleanup(self):
        await self.exit_stack.aclose()


if __name__ == "__main__":
    import argparse

    import trio

    async def main(server_script_path=None):
        if not server_script_path:
            server_script_path = input("Enter the path to the server script (e.g., weather_server.py): ").strip()
            if not server_script_path:
                print("Error: Server script path cannot be empty.")
                return

        try:
            mcp_client = SimpleMCPClient()
            await mcp_client.connect_to_server(server_script_path=server_script_path)

            prompts = await mcp_client.list_prompts()
            print("Prompts:", prompts)

            tools = await mcp_client.list_tools()
            print("Tools:", tools)

            result = await mcp_client.call_tool("get_current_weather", arguments={"location": "Shanghai"})
            print("Tool result:", result)
        finally:
            await mcp_client.cleanup()

    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--server-script-path", type=str, help="Path to the server script (e.g., weather_server.py)")
    args = parser.parse_args()

    trio.run(main, args.server_script_path)
