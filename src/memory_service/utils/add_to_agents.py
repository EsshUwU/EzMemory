from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON from a file, returning an empty dict on any error."""
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            return json.loads(text or "{}")
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _ensure_parent_dir(path: Path) -> None:
    """Ensure parent directory exists for a path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def add_cursor_mcp() -> bool:
    """
    Add EzMemory MCP configuration for Cursor.

    Returns:
        bool: True if the entry was newly added, False if it already existed.
    """
    cursor_dir = Path.home() / ".cursor"
    mcp_path = cursor_dir / "mcp.json"

    _ensure_parent_dir(mcp_path)

    data = _load_json_file(mcp_path)
    if not isinstance(data, dict):
        data = {}

    mcp_servers = data.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    data["mcpServers"] = mcp_servers

    # If already configured, do not overwrite
    if "ezmemory" in mcp_servers:
        return False

    mcp_servers["ezmemory"] = {
        "url": "http://localhost:8080/mcp",
    }

    mcp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def _get_vscode_mcp_path() -> Path:
    """
    Get the path to VS Code's MCP configuration file.

    On Windows this is typically:
        %APPDATA%\\Code\\User\\mcp.json
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        # Fallback best-effort for non-standard environments
        base = Path.home() / "AppData" / "Roaming"

    return base / "Code" / "User" / "mcp.json"


def add_vscode_mcp() -> bool:
    """
    Add EzMemory MCP configuration for VS Code.

    VS Code expects a structure like:
    {
        "servers": {
            "ezmemory": {
                "url": "http://localhost:8080/mcp",
                "type": "http"
            }
        },
        "inputs": []
    }

    Returns:
        bool: True if the entry was newly added, False if it already existed.
    """
    mcp_path = _get_vscode_mcp_path()
    _ensure_parent_dir(mcp_path)

    data = _load_json_file(mcp_path)
    if not isinstance(data, dict):
        data = {}

    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    data["servers"] = servers

    inputs = data.get("inputs")
    if not isinstance(inputs, list):
        inputs = []
    data["inputs"] = inputs

    # If already configured, do not overwrite
    if "ezmemory" in servers:
        return False

    servers["ezmemory"] = {
        "url": "http://localhost:8080/mcp",
        "type": "http",
    }

    mcp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True

