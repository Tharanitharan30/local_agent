from pathlib import Path
import pytest

from tools.filesystem import (
    FilesystemTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    FileInfoTool,
    CreateFileTool,
    WriteFileTool,
    EditFileTool,
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

    # Ambiguous file with multiple occurrences
    (ws / "ambiguous.txt").write_text("apple\nbanana\napple\norange\n", encoding="utf-8")

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
# create_file tests (Milestone 3)
# ==============================================================================

def test_create_file_new(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.create_file("new_script.py", "print('hello from new file')")
    assert res["success"] is True
    assert res["operation"] == "create_file"
    assert (mock_workspace / "new_script.py").exists()
    assert (mock_workspace / "new_script.py").read_text(encoding="utf-8") == "print('hello from new file')"
    assert res["bytes_written"] > 0


def test_create_file_already_exists(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.create_file("hello.txt", "Overwrite attempt")
    assert res["success"] is False
    assert "already exists" in res["error"]
    # Ensure original was not modified
    assert (mock_workspace / "hello.txt").read_text(encoding="utf-8") == "Hello, Zia filesystem!"


def test_create_file_workspace_escape(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.create_file("../outside.txt", "Escaping workspace")
    assert res["success"] is False
    assert "outside the allowed workspace" in res["error"]
    assert not (mock_workspace.parent / "outside.txt").exists()


def test_create_file_oversized(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace, max_write_size=100)
    res = fs.create_file("too_large.txt", "B" * 200)
    assert res["success"] is False
    assert "exceeds maximum allowed write size" in res["error"]
    assert not (mock_workspace / "too_large.txt").exists()


def test_create_file_nonexistent_parent(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.create_file("missing_parent_dir/new.txt", "Content")
    assert res["success"] is False
    assert "Parent directory" in res["error"]


# ==============================================================================
# write_file tests (Milestone 3)
# ==============================================================================

def test_write_file_existing(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.write_file("hello.txt", "Completely rewritten content!")
    assert res["success"] is True
    assert res["operation"] == "write_file"
    assert (mock_workspace / "hello.txt").read_text(encoding="utf-8") == "Completely rewritten content!"


def test_write_file_nonexistent_rejection(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.write_file("nonexistent.txt", "Some content")
    assert res["success"] is False
    assert "does not exist" in res["error"]
    assert "Use 'create_file'" in res["error"]


def test_write_file_workspace_escape(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.write_file("../escape.txt", "Content")
    assert res["success"] is False
    assert "outside the allowed workspace" in res["error"]


def test_write_file_on_directory_rejection(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.write_file("subdir", "Content")
    assert res["success"] is False
    assert "is a directory" in res["error"]


# ==============================================================================
# edit_file tests (Milestone 3)
# ==============================================================================

def test_edit_file_exact_replacement(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    # config.py has: "DEBUG = True\nPORT = 8080\n"
    res = fs.edit_file("config.py", old_text="DEBUG = True", new_text="DEBUG = False")
    assert res["success"] is True
    assert res["operation"] == "edit_file"
    assert res["replacements"] == 1
    content = (mock_workspace / "config.py").read_text(encoding="utf-8")
    assert "DEBUG = False" in content
    assert "PORT = 8080" in content


def test_edit_file_old_text_not_found(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.edit_file("config.py", old_text="NONEXISTENT_KEY = 123", new_text="REPLACED")
    assert res["success"] is False
    assert "was not found" in res["error"]


def test_edit_file_ambiguous_rejection(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    # ambiguous.txt contains 'apple' twice
    res = fs.edit_file("ambiguous.txt", old_text="apple", new_text="pear")
    assert res["success"] is False
    assert "Ambiguous match" in res["error"]
    assert "occurs 2 times" in res["error"]
    # Ensure file was not modified
    content = (mock_workspace / "ambiguous.txt").read_text(encoding="utf-8")
    assert content.count("apple") == 2


def test_edit_file_nonexistent_rejection(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.edit_file("missing.py", old_text="foo", new_text="bar")
    assert res["success"] is False
    assert "does not exist" in res["error"]


def test_edit_file_workspace_escape(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    res = fs.edit_file("../../etc/issue", old_text="foo", new_text="bar")
    assert res["success"] is False
    assert "outside the allowed workspace" in res["error"]


# ==============================================================================
# Security & Backup tests
# ==============================================================================

def test_backup_creation_on_edit(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace, enable_backups=True)
    res = fs.edit_file("config.py", old_text="PORT = 8080", new_text="PORT = 9090")
    assert res["success"] is True

    backup_dir = mock_workspace / ".zia_backups"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("config.py.*.bak"))
    assert len(backups) == 1
    # Check that backup contains original text
    assert "PORT = 8080" in backups[0].read_text(encoding="utf-8")


def test_system_directories_forbidden_writes(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)
    for forbidden in ["/proc/foo", "/sys/foo", "/dev/foo", "/etc/foo"]:
        res_create = fs.create_file(forbidden, "test")
        assert res_create["success"] is False
        assert "outside the allowed workspace" in res_create["error"] or "forbidden" in res_create["error"]

        res_write = fs.write_file(forbidden, "test")
        assert res_write["success"] is False
        assert "outside the allowed workspace" in res_write["error"] or "forbidden" in res_write["error"]


def test_no_destructive_methods():
    """Verify that filesystem tool does not expose delete or chmod."""
    fs = FilesystemTool()
    for method in ["delete", "remove", "unlink", "rmdir", "chmod", "chown"]:
        assert not hasattr(fs, method), f"Destructive method '{method}' must not exist."


# ==============================================================================
# Modular Tool Wrappers tests
# ==============================================================================

def test_modular_wrappers(mock_workspace: Path):
    fs = FilesystemTool(workspace_root=mock_workspace)

    create_tool = CreateFileTool(fs_tool=fs)
    assert create_tool.name == "filesystem.create_file"
    res_c = create_tool.execute(path="wrapper_test.txt", content="created via wrapper")
    assert res_c["success"] is True

    edit_tool = EditFileTool(fs_tool=fs)
    assert edit_tool.name == "filesystem.edit_file"
    res_e = edit_tool.execute(path="wrapper_test.txt", old_text="created", new_text="edited")
    assert res_e["success"] is True

    write_tool = WriteFileTool(fs_tool=fs)
    assert write_tool.name == "filesystem.write_file"
    res_w = write_tool.execute(path="wrapper_test.txt", content="overwritten via wrapper")
    assert res_w["success"] is True

    read_tool = ReadFileTool(fs_tool=fs)
    res_r = read_tool.execute(path="wrapper_test.txt")
    assert res_r["content"] == "overwritten via wrapper"
