"""Tests for file upload manager utility."""

import pytest
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile

from oai_agent_server.utils.file_manager import FileUploadManager


class TestFileUploadManager:
    """Test suite for FileUploadManager class."""
    
    def test_validate_files_empty_list(self):
        """Empty file list should pass validation."""
        FileUploadManager.validate_files([])
    
    def test_validate_files_within_limits(self):
        """Valid file count and sizes should pass."""
        files = [
            MagicMock(filename="file1.txt", size=1024),
            MagicMock(filename="file2.txt", size=2048),
        ]
        FileUploadManager.validate_files(files)
    
    def test_validate_files_exceeds_max_count(self):
        """Exceeding max file count should raise ValueError."""
        max_files = FileUploadManager.MAX_FILES
        files = [MagicMock(filename=f"file{i}.txt", size=100) for i in range(max_files + 1)]
        
        with pytest.raises(ValueError, match="Maximum.*files allowed"):
            FileUploadManager.validate_files(files)
    
    def test_validate_files_exceeds_size_limit(self):
        """File exceeding MAX_FILE_SIZE should raise ValueError."""
        max_size = FileUploadManager.MAX_FILE_SIZE
        files = [MagicMock(filename="large.bin", size=max_size + 1)]
        
        with pytest.raises(ValueError, match="exceeds maximum size"):
            FileUploadManager.validate_files(files)
    
    @pytest.mark.asyncio
    async def test_save_files_empty_list(self):
        """Empty file list should return empty paths and None temp_dir."""
        file_paths, temp_dir = await FileUploadManager.save_files([])
        assert file_paths == []
        assert temp_dir is None
    
    @pytest.mark.asyncio
    async def test_save_files_single_file(self):
        """Single file should be saved correctly."""
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.size = 12
        mock_file.read = AsyncMock(return_value=b"test content")
        
        file_paths, temp_dir = await FileUploadManager.save_files([mock_file])
        
        try:
            assert len(file_paths) == 1
            assert temp_dir is not None
            assert os.path.exists(temp_dir)
            assert os.path.exists(file_paths[0])
            assert file_paths[0].endswith("test.txt")
            
            with open(file_paths[0], "rb") as f:
                assert f.read() == b"test content"
        finally:
            if temp_dir:
                FileUploadManager.cleanup(temp_dir)
    
    @pytest.mark.asyncio
    async def test_save_files_validation_before_save(self):
        """Validation should happen before any files are saved."""
        max_files = FileUploadManager.MAX_FILES
        mock_files = [MagicMock(filename=f"file{i}.txt", size=100) for i in range(max_files + 1)]
        
        with pytest.raises(ValueError, match="Maximum.*files allowed"):
            await FileUploadManager.save_files(mock_files)
    
    def test_cleanup_existing_directory(self):
        """cleanup should remove directory and contents."""
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "test.txt")
        with open(temp_file, "w") as f:
            f.write("test")
        
        assert os.path.exists(temp_dir)
        FileUploadManager.cleanup(temp_dir)
        assert not os.path.exists(temp_dir)
    
    def test_cleanup_nonexistent_directory(self):
        """cleanup should not raise error for nonexistent directory."""
        FileUploadManager.cleanup("/nonexistent/path")  # Should not raise
    
    def test_cleanup_none(self):
        """cleanup should handle None gracefully."""
        FileUploadManager.cleanup(None)  # Should not raise
    
    def test_append_files_to_message_string_with_files(self):
        """Files should be appended to string message."""
        message = "Hello"
        files = ["/path/to/file1.txt", "/path/to/file2.txt"]
        result = FileUploadManager.append_files_to_message(message, files)
        
        assert "Hello" in result
        assert "/path/to/file1.txt" in result
        assert "/path/to/file2.txt" in result
    
    def test_append_files_to_message_string_with_session(self):
        """Session ID should be appended to string message."""
        message = "Hello"
        result = FileUploadManager.append_files_to_message(message, [], session_id="session123")
        
        assert "Hello" in result
        assert "session123" in result
    
    def test_append_files_to_message_dict_with_content_key(self):
        """Files should be appended to dict message with 'content' key."""
        message = {"content": "Hello"}
        files = ["/path/to/file.txt"]
        result = FileUploadManager.append_files_to_message(message, files)
        
        assert "Hello" in result["content"]
        assert "/path/to/file.txt" in result["content"]
    
    def test_append_files_to_message_no_changes_needed(self):
        """Message should be unchanged if no files or session_id."""
        message = "Hello"
        result = FileUploadManager.append_files_to_message(message, [])
        assert result == message
