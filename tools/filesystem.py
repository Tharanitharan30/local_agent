from datetime import datetime, timezone
import fnmatch
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import config
from tools.base import BaseTool


def resolve_safe_path(path_str: str, workspace_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Safely resolve a path string against the workspace root.
    Supports relative paths, absolute paths, and '~'.
    Enforces that the resolved target is strictly contained within workspace_root.
    Blocks access to virtual/system filesystems (/proc, /sys, /dev, /etc, /boot, /usr).

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

    # Prevent traversal into sensitive system/virtual directories
    cand_str = str(candidate)
    for forbidden in ("/proc", "/sys", "/dev", "/etc", "/boot", "/usr", "/root"):
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
    Unified filesystem tool for Zia.
    Provides safe, structured read and write operations strictly within the workspace.
    Supported operations:
      Read: 'list_directory', 'read_file', 'search_files', 'file_info'
      Write: 'create_file', 'write_file', 'edit_file'
    """

    name = "filesystem"
    description = (
        "Filesystem tool for workspace file management. "
        "Supports operations: 'list_directory', 'read_file', 'search_files', 'file_info', "
        "'create_file', 'write_file', and 'edit_file'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "list_directory",
                    "read_file",
                    "search_files",
                    "file_info",
                    "create_file",
                    "write_file",
                    "edit_file",
                ],
                "description": "The filesystem operation to perform."
            },
            "path": {
                "type": "string",
                "description": "Target file or directory path within workspace."
            },
            "content": {
                "type": "string",
                "description": "Text content to write (for 'create_file' and 'write_file')."
            },
            "old_text": {
                "type": "string",
                "description": "Exact text snippet to replace (for 'edit_file')."
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text snippet (for 'edit_file')."
            },
            "pattern": {
                "type": "string",
                "description": "Search pattern (e.g. '*.py') used with 'search_files'."
            }
        },
        "required": ["operation", "path"]
    }

    IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".tox", "node_modules", ".zia_backups"}

    def __init__(
        self,
        workspace_root: Path = config.WORKSPACE_ROOT,
        max_read_size: int = config.MAX_READ_FILE_SIZE,
        max_write_size: int = config.MAX_WRITE_FILE_SIZE,
        max_search_results: int = config.MAX_SEARCH_RESULTS,
        max_search_depth: int = config.MAX_SEARCH_DEPTH,
        enable_backups: bool = config.ENABLE_BACKUPS,
        backup_dir_name: str = config.BACKUP_DIR_NAME,
        max_backups_per_file: int = config.MAX_BACKUPS_PER_FILE,
    ):
        self.workspace_root = workspace_root
        self.max_read_size = max_read_size
        self.max_write_size = max_write_size
        self.max_search_results = max_search_results
        self.max_search_depth = max_search_depth
        self.enable_backups = enable_backups
        self.backup_dir_name = backup_dir_name
        self.max_backups_per_file = max_backups_per_file

    # --------------------------------------------------------------------------
    # Backup Management
    # --------------------------------------------------------------------------

    def _create_backup(self, target_path: Path) -> Optional[Path]:
        """Create a timestamped backup of a file before modifying it."""
        if not self.enable_backups or not target_path.exists() or not target_path.is_file():
            return None

        backup_root = self.workspace_root / self.backup_dir_name
        try:
            backup_root.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rel_name = target_path.name
            backup_name = f"{rel_name}.{timestamp}.bak"
            backup_file = backup_root / backup_name
            shutil.copy2(target_path, backup_file)

            # Prune older backups for this file if exceeding limit
            existing = sorted(
                backup_root.glob(f"{rel_name}.*.bak"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(existing) > self.max_backups_per_file:
                oldest = existing.pop(0)
                try:
                    oldest.unlink()
                except OSError:
                    pass

            return backup_file
        except Exception:
            return None

    # --------------------------------------------------------------------------
    # Atomic Write Utility
    # --------------------------------------------------------------------------

    def _atomic_write(self, target_path: Path, content: str) -> Tuple[bool, Optional[str], int]:
        """
        Atomically write text content to target_path:
        1. Write to temporary file in the same directory.
        2. Flush and fsync.
        3. Atomically replace target.
        4. Verify result.
        Returns (success, error_message, bytes_written).
        """
        encoded = content.encode("utf-8")
        bytes_count = len(encoded)

        if bytes_count > self.max_write_size:
            return (
                False,
                f"Content size ({bytes_count} bytes) exceeds maximum allowed write size ({self.max_write_size} bytes).",
                0,
            )

        parent_dir = target_path.parent
        if not parent_dir.exists():
            return False, f"Parent directory '{parent_dir}' does not exist.", 0

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=parent_dir,
                mode="wb",
                delete=False,
                prefix=".zia_tmp_",
            ) as f:
                temp_file = Path(f.name)
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename / replace
            os.replace(temp_file, target_path)

            # Verification
            if not target_path.exists():
                return False, f"Verification failed: '{target_path}' does not exist after write.", 0
            if target_path.stat().st_size != bytes_count:
                return False, "Verification failed: File size mismatch after atomic write.", 0

            return True, None, bytes_count
        except Exception as e:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            return False, f"Atomic write failed: {str(e)}", 0

    # --------------------------------------------------------------------------
    # Read Operations
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Write & Modification Operations (Milestone 3)
    # --------------------------------------------------------------------------

    def create_file(self, path: str, content: str = "") -> Dict[str, Any]:
        """
        Create a new text file within workspace.
        Refuses to overwrite existing files.
        Uses atomic writing.
        """
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "create_file", "path": path, "error": err}

        if target.exists():
            return {
                "success": False,
                "operation": "create_file",
                "path": str(target),
                "error": "File already exists. Use 'write_file' or 'edit_file' to modify existing files."
            }

        if not target.parent.exists():
            return {
                "success": False,
                "operation": "create_file",
                "path": str(target),
                "error": f"Parent directory '{target.parent}' does not exist."
            }

        success, write_err, bytes_written = self._atomic_write(target, content)
        if not success:
            return {
                "success": False,
                "operation": "create_file",
                "path": str(target),
                "error": write_err or "Failed to create file."
            }

        return {
            "success": True,
            "operation": "create_file",
            "path": str(target.relative_to(self.workspace_root)),
            "message": "File created successfully",
            "bytes_written": bytes_written
        }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Replace the contents of an existing text file within workspace.
        Refuses if file does not exist (does not create silently).
        Creates automatic backup and performs atomic write.
        """
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "write_file", "path": path, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "write_file",
                "path": str(target),
                "error": f"File '{path}' does not exist. Use 'create_file' to create a new file."
            }

        if not target.is_file():
            return {
                "success": False,
                "operation": "write_file",
                "path": str(target),
                "error": f"Path '{path}' is a directory, not a file."
            }

        if is_binary_file(target):
            return {
                "success": False,
                "operation": "write_file",
                "path": str(target),
                "error": f"Refusing to overwrite binary file '{target.name}' with text."
            }

        # Create backup prior to modification
        self._create_backup(target)

        success, write_err, bytes_written = self._atomic_write(target, content)
        if not success:
            return {
                "success": False,
                "operation": "write_file",
                "path": str(target),
                "error": write_err or "Failed to write file."
            }

        return {
            "success": True,
            "operation": "write_file",
            "path": str(target.relative_to(self.workspace_root)),
            "message": "File written successfully",
            "bytes_written": bytes_written
        }

    def edit_file(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """
        Perform a deterministic replacement of 'old_text' with 'new_text' in an existing file.
        Fails cleanly if 'old_text' is not found or occurs multiple times (ambiguity).
        Creates automatic backup and performs atomic write.
        """
        target, err = resolve_safe_path(path, self.workspace_root)
        if err:
            return {"success": False, "operation": "edit_file", "path": path, "error": err}

        if not target.exists():
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": f"File '{path}' does not exist."
            }

        if not target.is_file():
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": f"Path '{path}' is a directory, not a file."
            }

        if is_binary_file(target):
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": f"Refusing to edit binary file '{target.name}'."
            }

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": f"Failed to read file for editing: {str(e)}"
            }

        if old_text not in content:
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": f"Target snippet 'old_text' was not found in '{path}'."
            }

        count = content.count(old_text)
        if count > 1:
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": (
                    f"Ambiguous match: Target snippet occurs {count} times in '{path}'. "
                    "Please provide more surrounding context to match a unique occurrence."
                )
            }

        new_content = content.replace(old_text, new_text, 1)

        # Create backup prior to modification
        self._create_backup(target)

        success, write_err, bytes_written = self._atomic_write(target, new_content)
        if not success:
            return {
                "success": False,
                "operation": "edit_file",
                "path": str(target),
                "error": write_err or "Failed to edit file."
            }

        return {
            "success": True,
            "operation": "edit_file",
            "path": str(target.relative_to(self.workspace_root)),
            "message": "File edited successfully",
            "bytes_written": bytes_written,
            "replacements": 1
        }

    def execute(
        self,
        operation: str = "",
        path: str = ".",
        pattern: str = "*",
        content: str = "",
        old_text: str = "",
        new_text: str = "",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute any supported filesystem operation.
        """
        op = (operation or kwargs.get("action") or "").lower().strip()
        p = path or kwargs.get("file_path") or kwargs.get("dir_path") or "."
        pat = pattern or kwargs.get("query") or "*"
        c = content or kwargs.get("text") or ""
        ot = old_text or kwargs.get("target_text") or ""
        nt = new_text or kwargs.get("replacement") or ""

        if op == "list_directory":
            return self.list_directory(p)
        elif op == "read_file":
            return self.read_file(p)
        elif op == "search_files":
            return self.search_files(p, pat)
        elif op == "file_info":
            return self.file_info(p)
        elif op == "create_file":
            return self.create_file(p, c)
        elif op == "write_file":
            return self.write_file(p, c)
        elif op == "edit_file":
            return self.edit_file(p, ot, nt)
        else:
            supported = [
                "list_directory",
                "read_file",
                "search_files",
                "file_info",
                "create_file",
                "write_file",
                "edit_file",
            ]
            return {
                "success": False,
                "operation": op,
                "error": f"Unknown filesystem operation '{op}'. Supported: {supported}"
            }


# ------------------------------------------------------------------------------
# Modular Tool Wrappers for Registration
# ------------------------------------------------------------------------------

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


class CreateFileTool(BaseTool):
    """Tool for creating a new text file within workspace."""
    name = "filesystem.create_file"
    aliases = ["create_file"]
    description = (
        "Create a new text file within the workspace. Refuses if the file already exists. "
        "Use when asked to create a new file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the new file to create."
            },
            "content": {
                "type": "string",
                "description": "Initial text content of the file."
            }
        },
        "required": ["path", "content"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = "", content: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.create_file(path, content)


class WriteFileTool(BaseTool):
    """Tool for replacing content of an existing text file."""
    name = "filesystem.write_file"
    aliases = ["write_file"]
    description = (
        "Replace the entire content of an existing text file within workspace. "
        "Refuses if file does not exist. Use when completely rewriting a file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the existing file to overwrite."
            },
            "content": {
                "type": "string",
                "description": "New text content to replace the file with."
            }
        },
        "required": ["path", "content"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = "", content: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.write_file(path, content)


class EditFileTool(BaseTool):
    """Tool for targeted replacement of text in an existing file."""
    name = "filesystem.edit_file"
    aliases = ["edit_file"]
    description = (
        "Perform a targeted replacement of 'old_text' with 'new_text' in an existing file. "
        "Target file must exist, and 'old_text' must occur exactly once to avoid ambiguity."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit."
            },
            "old_text": {
                "type": "string",
                "description": "Exact text snippet in the file to be replaced."
            },
            "new_text": {
                "type": "string",
                "description": "New replacement text snippet."
            }
        },
        "required": ["path", "old_text", "new_text"]
    }

    def __init__(self, fs_tool: Optional[FilesystemTool] = None):
        self.fs = fs_tool or FilesystemTool()

    def execute(self, path: str = "", old_text: str = "", new_text: str = "", **kwargs: Any) -> Dict[str, Any]:
        return self.fs.edit_file(path, old_text, new_text)
