from pathlib import Path
import pytest

from tools.filesystem import (
    FilesystemTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    FileInfoTool,
    resolve_safe_path,
    is_binary_file,
)


@pytest.fixture
def mock_workspace(tmp_path: Path):
    """Creates a temporary workspace with sample files and directories for testing."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Text files
    (ws / "hello.txt").write_text("Hello, Zia filesystem!", encoding="utf-8")
    (ws / "config.py").write_text("DEBUG = True\nPORT = 8080\n", encoding="utf-8")
    (ws / "script.py").write_text("print('test script')\n", encoding="utf-8")

    # Subdirectory with files
    subdir = ws / "subdir"
    subdir.mkdir()
    (subdir / "sub_script.py").write_text("# Subdir script\n", encoding="utf-8")
    (subdir / "notes.md").write_text("# Notes\nSome content\n", encoding="utf-8")

    # Ignored directory (.venv)
    venv = ws / ".venv"
    venv.mkdir()
    (venv / "ignored.py").write_text("# Ignored\n", encoding="utf-8")

    # Large file (> 1KB for low limit test)
    (ws / "large.txt").write_text("A" * 2000, encoding="utf-8")

    # Binary file (contains null bytes)
    (ws / "binary.bin").write_bytes(b"\x00\x01\x02\x03\x04\xff")

    return ws


# ==============================================================================
# list_directory tests
# ==============================================================================

def test_list_directory_valid(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.list_directory(".")
    assert res["success"] is True
    assert res["operation"] == "list_directory"
    names = [e["name"] for e in res["entries"]]
    assert "hello.txt" in names
    assert "config.py" in names
    assert "subdir" in names
    assert res["total_entries"] > 0


def test_list_directory_subdirectory(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.list_directory("subdir")
    assert res["success"] is True
    names = [e["name"] for e in res["entries"]]
    assert "sub_script.py" in names
    assert "notes.md" in names


def test_list_directory_nonexistent(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.list_directory("nonexistent_dir_123")
    assert res["success"] is False
    assert "does not exist" in res["error"]


def test_list_directory_on_file(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.list_directory("hello.txt")
    assert res["success"] is False
    assert "is a file, not a directory" in res["error"]


# ==============================================================================
# read_file tests
# ==============================================================================

def test_read_file_valid(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.read_file("hello.txt")
    assert res["success"] is True
    assert res["operation"] == "read_file"
    assert res["content"] == "Hello, Zia filesystem!"
    assert res["size"] == len("Hello, Zia filesystem!")


def test_read_file_nonexistent(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.read_file("does_not_exist.txt")
    assert res["success"] is False
    assert "does not exist" in res["error"]


def test_read_file_oversized(mock_workspace: Path):
    # Set limit to 1000 bytes (large.txt is 2000 bytes)
    fs = FilesystemTool(workspace_root=mock_workspace, max_read_size=1000)
    res = fs.read_file("large.txt")
    assert res["success"] is False
    assert "exceeds maximum allowed size" in res["error"]


def test_read_file_binary(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.read_file("binary.bin")
    assert res["success"] is False
    assert "Refusing to read binary file" in res["error"]


def test_read_file_on_directory(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.read_file("subdir")
    assert res["success"] is False
    assert "is a directory, not a file" in res["error"]


# ==============================================================================
# search_files tests
# ==============================================================================

def test_search_files_matching(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.search_files(".", "*.py")
    assert res["success"] is True
    assert res["operation"] == "search_files"
    names = [r["name"] for r in res["results"]]
    assert "config.py" in names
    assert "script.py" in names
    assert "sub_script.py" in names
    # .venv should be ignored
    assert "ignored.py" not in names


def test_search_files_no_matches(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.search_files(".", "*.rust")
    assert res["success"] is True
    assert len(res["results"]) == 0
    assert res["count"] == 0


def test_search_files_result_limit(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace, max_search_results=1)
    res = fs.search_files(".", "*.py")
    assert res["success"] is True
    assert len(res["results"]) == 1
    assert res["truncated"] is True


def test_search_files_nonexistent_dir(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.search_files("ghost_directory", "*.py")
    assert res["success"] is False
    assert "does not exist" in res["error"]


# ==============================================================================
# file_info tests
# ==============================================================================

def test_file_info_existing_file(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.file_info("hello.txt")
    assert res["success"] is True
    assert res["type"] == "file"
    assert res["size"] == len("Hello, Zia filesystem!")
    assert "modification_time" in res
    assert res["is_readable"] is True


def test_file_info_existing_directory(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.file_info("subdir")
    assert res["success"] is True
    assert res["type"] == "directory"
    assert res["name"] == "subdir"


def test_file_info_nonexistent(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.file_info("fake.txt")
    assert res["success"] is False
    assert "does not exist" in res["error"]


# ==============================================================================
# Security & Workspace Policy tests
# ==============================================================================

def test_path_traversal_rejected(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    # Attempt to escape workspace using relative traversal
    res = fs.read_file("../../etc/passwd")
    assert res["success"] is False
    assert "outside the allowed workspace" in res["error"]


def test_outside_workspace_rejected(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.list_directory("/etc")
    assert res["success"] is False
    assert "outside the allowed workspace" in res["error"]


def test_system_directories_forbidden(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    for forbidden in ["/proc", "/sys", "/dev"]:
        res = fs.list_directory(forbidden)
        assert res["success"] is False
        assert "outside the allowed workspace" in res["error"] or "forbidden" in res["error"]


def test_read_only_nature():
    """Verify that filesystem tool only exposes read-oriented methods and no write/delete."""
    fs = FilesystemTool()
    disallowed_methods = ["write", "create", "delete", "remove", "rename", "chmod", "execute_file"]
    for m in disallowed_methods:
        assert not hasattr(fs, m), f"FilesystemTool should not expose write/destructive method: {m}"


# ==============================================================================
# Dedicated Tool Wrappers tests
# ==============================================================================

def test_modular_wrappers(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)

    list_tool = ListDirectoryTool(fs_tool=fs)
    assert list_tool.name == "filesystem.list_directory"
    res_list = list_tool.execute(path=".")
    assert res_list["success"] is True

    read_tool = ReadFileTool(fs_tool=fs)
    assert read_tool.name == "filesystem.read_file"
    res_read = read_tool.execute(path="hello.txt")
    assert res_read["success"] is True

    search_tool = SearchFilesTool(fs_tool=fs)
    assert search_tool.name == "filesystem.search_files"
    res_search = search_tool.execute(path=".", pattern="*.txt")
    assert res_search["success"] is True

    info_tool = FileInfoTool(fs_tool=fs)
    assert info_tool.name == "filesystem.file_info"
    res_info = info_tool.execute(path="hello.txt")
    assert res_info["success"] is True
