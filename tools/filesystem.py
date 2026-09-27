from datetime import datetime, timezone
import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from tools.base import BaseTool


def resolve_safe_path(path_str: str, workspace_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Safely resolve a path string against the workspace root.
    Supports relative paths, absolute paths, and '~'.
    Enforces that the resolved target is strictly contained within workspace_root.
    Blocks access to virtual/system filesystems (/proc, /sys, /dev, /etc).

    Returns:
        (resolved_path, error_message)
    """
    if not path_str or not path_str.strip():
        path_str = "."

    raw = Path(path_str).expanduser()
    ws_resolved = workspace_root.resolve()

    if not raw.is_absolute():
        candidate = (ws_resolved / raw).resolve()
    else:
        candidate = raw.resolve()

    # Workspace confinement check: candidate must be relative to workspace_root
    try:
        candidate.relative_to(ws_resolved)
    except ValueError:
        return None, (
            f"Access denied: Path '{path_str}' resolves to '{candidate}', "
            f"which is outside the allowed workspace '{ws_resolved}'."
        )

    # Prevent traversal into dangerous virtual/system filesystems
    cand_str = str(candidate)
    for forbidden in ("/proc", "/sys", "/dev", "/etc"):
        if cand_str == forbidden or cand_str.startswith(forbidden + "/"):
            return None, f"Access denied: Access to system directory '{forbidden}' is forbidden."

    return candidate, None


def is_binary_file(file_path: Path) -> bool:
    """Check if a file appears to be binary by inspecting initial bytes for null bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return False


class FilesystemTool(BaseTool):
    """
    Unified read-only filesystem tool for Zia.
    Provides safe, structured access to list directories, read text files,
    search for files, and retrieve file metadata within the allowed workspace.
    """

    name = "filesystem"
    description = (
        "Read-only filesystem tool. Supports operations: 'list_directory', 'read_file', "
        "'search_files', and 'file_info' within the allowed workspace."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list_directory", "read_file", "search_files", "file_info"],
                "description": "The filesystem read operation to perform."
            },
            "path": {
                "type": "string",
                "description": "Target file or directory path (relative to workspace or absolute within workspace)."
            },
            "pattern": {
                "type": "string",
                "description": "Optional search pattern (e.g. '*.py') used with 'search_files'."
            }
        },
        "required": ["operation", "path"]
    }

    # Directories to automatically skip during recursive search to avoid token blowout
    IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".tox", "node_modules"}

    def __init__(
        self,
        workspace_root: Path = config.WORKSPACE_ROOT,
        max_read_size: int = config.MAX_READ_FILE_SIZE,
        max_search_results: int = config.MAX_SEARCH_RESULTS,
        max_search_depth: int = config.MAX_SEARCH_DEPTH,
    ):
        self.workspace_root = workspace_root
        self.max_read_size = max_read_size
        self.max_search_results = max_search_results
        self.max_search_depth = max_search_depth

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """List contents of a single directory level."""
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "list_directory", "path": path, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "list_directory",
                "path": str(target),
                "error": f"Directory '{path}' does not exist."
            }

        if not target.is_dir():
            return {
                "success": False,
                "operation": "list_directory",
                "path": str(target),
                "error": f"Path '{path}' is a file, not a directory."
            }

        try:
            entries = []
            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                entry_type = "directory" if item.is_dir() else ("file" if item.is_file() else "other")
                size = item.stat().st_size if item.is_file() else None
                entries.append({
                    "name": item.name,
                    "type": entry_type,
                    "size": size,
                    "path": str(item.relative_to(self.workspace_root))
                })

            return {
                "success": True,
                "operation": "list_directory",
                "path": str(target),
                "entries": entries[:100],
                "total_entries": len(entries),
                "truncated": len(entries) > 100,
            }
        except Exception as e:
            return {"success": False, "operation": "list_directory", "path": str(target), "error": str(e)}

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read the text content of a file within the workspace with safety bounds."""
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "read_file", "path": path, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "read_file",
                "path": str(target),
                "error": f"File '{path}' does not exist."
            }

        if not target.is_file():
            return {
                "success": False,
                "operation": "read_file",
                "path": str(target),
                "error": f"Path '{path}' is a directory, not a file."
            }

        try:
            stat = target.stat()
            file_size = stat.st_size

            # Enforce file size limit to prevent context explosion
            if file_size > self.max_read_size:
                return {
                    "success": False,
                    "operation": "read_file",
                    "path": str(target),
                    "size": file_size,
                    "error": (
                        f"File size ({file_size} bytes) exceeds maximum allowed size "
                        f"({self.max_read_size} bytes)."
                    )
                }

            # Check for binary file
            if is_binary_file(target):
                return {
                    "success": False,
                    "operation": "read_file",
                    "path": str(target),
                    "size": file_size,
                    "error": f"Refusing to read binary file '{target.name}' as text."
                }

            content = target.read_text(encoding="utf-8", errors="replace")
            return {
                "success": True,
                "operation": "read_file",
                "path": str(target),
                "content": content,
                "size": file_size
            }
        except Exception as e:
            return {"success": False, "operation": "read_file", "path": str(target), "error": str(e)}

    def search_files(self, path: str = ".", pattern: str = "*") -> Dict[str, Any]:
        """Recursively search for files matching a pattern within workspace limits."""
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "search_files", "path": path, "pattern": pattern, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "search_files",
                "path": str(target),
                "pattern": pattern,
                "error": f"Search path '{path}' does not exist."
            }

        if not pattern or not pattern.strip():
            pattern = "*"

        matches: List[Dict[str, Any]] = []
        target_depth = len(target.resolve().parts)

        try:
            for root, dirs, files in os.walk(target):
                # Prune ignored virtual environment / cache directories
                dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

                current_depth = len(Path(root).resolve().parts) - target_depth
                if current_depth > self.max_search_depth:
                    dirs.clear()
                    continue

                for f in files:
                    if fnmatch.fnmatch(f, pattern):
                        file_path = Path(root) / f
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        matches.append({
                            "name": f,
                            "path": str(file_path.relative_to(self.workspace_root)),
                            "size": size
                        })
                        if len(matches) >= self.max_search_results:
                            break

                if len(matches) >= self.max_search_results:
                    break

            return {
                "success": True,
                "operation": "search_files",
                "path": str(target),
                "pattern": pattern,
                "results": matches,
                "count": len(matches),
                "truncated": len(matches) >= self.max_search_results
            }
        except Exception as e:
            return {
                "success": False,
                "operation": "search_files",
                "path": str(target),
                "pattern": pattern,
                "error": str(e)
            }

    def file_info(self, path: str) -> Dict[str, Any]:
        """Retrieve structured metadata for a file or directory."""
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "file_info", "path": path, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "file_info",
                "path": str(target),
                "error": f"Path '{path}' does not exist."
            }

        try:
            stat = target.stat()
            item_type = "directory" if target.is_dir() else ("file" if target.is_file() else "symlink")
            mod_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            mod_iso = mod_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            perms = oct(stat.st_mode)[-3:]

            return {
                "success": True,
                "operation": "file_info",
                "path": str(target),
                "name": target.name,
                "type": item_type,
                "size": stat.st_size,
                "modification_time": mod_iso,
                "permissions": perms,
                "is_readable": os.access(target, os.R_OK)
            }
        except Exception as e:
            return {"success": False, "operation": "file_info", "path": str(target), "error": str(e)}

    def execute(
        self,
        operation: str = "",
        path: str = ".",
        pattern: str = "*",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a read-only filesystem operation.
        Supported operations: 'list_directory', 'read_file', 'search_files', 'file_info'.
        """
        op = (operation or kwargs.get("action") or "").lower().strip()
        p = path or kwargs.get("file_path") or kwargs.get("dir_path") or "."
        pat = pattern or kwargs.get("query") or "*"

        if op == "list_directory":
            return self.list_directory(p)
        elif op == "read_file":
            return self.read_file(p)
        elif op == "search_files":
            return self.search_files(p, pat)
        elif op == "file_info":
            return self.file_info(p)
        else:
            supported = ["list_directory", "read_file", "search_files", "file_info"]
            return {
                "success": False,
                "operation": op,
                "error": f"Unknown filesystem operation '{op}'. Supported: {supported}"
            }


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    name = "filesystem.list_directory"
    aliases = ["list_directory"]
    description = "List files and directories in a given folder within the workspace. Use when asked what files or folders exist in a directory or project."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory '.')."
            }
        },
        "required": ["path"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = ".", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.list_directory(path)


class ReadFileTool(BaseTool):
    """Tool for reading text file content."""
    name = "filesystem.read_file"
    aliases = ["read_file"]
    description = "Read the text content of a file within the workspace. Use when asked to inspect or read the contents of a text file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the text file to read."
            }
        },
        "required": ["path"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.read_file(path)


class SearchFilesTool(BaseTool):
    """Tool for searching files by pattern."""
    name = "filesystem.search_files"
    aliases = ["search_files"]
    description = "Recursively search for files matching a pattern (e.g. '*.py', '*.md') within the workspace. Use when asked to find files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Starting directory to search (default: '.')."
            },
            "pattern": {
                "type": "string",
                "description": "Pattern to match (e.g. '*.py', '*.md')."
            }
        },
        "required": ["path", "pattern"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = ".", pattern: str = "*", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.search_files(path, pattern)


class FileInfoTool(BaseTool):
    """Tool for inspecting file metadata."""
    name = "filesystem.file_info"
    aliases = ["file_info"]
    description = "Get metadata (size, type, modification time, permissions) for a file or directory. Use when asked about file info, size, or properties."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory."
            }
        },
        "required": ["path"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.file_info(path)
