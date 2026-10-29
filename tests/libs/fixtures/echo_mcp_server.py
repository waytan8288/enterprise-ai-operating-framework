"""Minimal stdio MCP server used by tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def echo(text: str) -> str:
    """Echo text back."""
    return f"echo: {text}"


@server.tool()
def delete_everything() -> str:
    """A tool that should never be allowlisted."""
    return "deleted"


if __name__ == "__main__":
    server.run(transport="stdio")
