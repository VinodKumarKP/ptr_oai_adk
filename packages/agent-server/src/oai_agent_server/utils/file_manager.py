"""File upload management utilities for handling multipart file uploads."""

import os
import shutil
import tempfile
from typing import List, Optional, Tuple, Union
from fastapi import UploadFile


class FileUploadManager:
    """Manages file uploads with validation, storage, and cleanup."""
    
    # Configuration constants
    MAX_FILES = int(os.environ.get("MAX_FILES", "10"))
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10MB default
    
    @classmethod
    def validate_files(cls, files: List[UploadFile]) -> None:
        """
        Validate file count and size constraints.
        
        Args:
            files: List of UploadFile objects to validate
            
        Raises:
            ValueError: If file count exceeds MAX_FILES or any file exceeds MAX_FILE_SIZE
        """
        if len(files) > cls.MAX_FILES:
            raise ValueError(
                f"Maximum {cls.MAX_FILES} files allowed, received {len(files)}"
            )
        
        for file in files:
            if file.size is not None and file.size > cls.MAX_FILE_SIZE:
                raise ValueError(
                    f"File '{file.filename}' exceeds maximum size "
                    f"({file.size} > {cls.MAX_FILE_SIZE} bytes)"
                )
    
    @classmethod
    async def save_files(
        cls, 
        files: List[UploadFile]
    ) -> Tuple[List[str], Optional[str]]:
        """
        Save uploaded files to a temporary directory with validation.
        
        Args:
            files: List of UploadFile objects to save
            
        Returns:
            Tuple of (file_paths, temp_dir) where:
            - file_paths: List of absolute paths to saved files
            - temp_dir: Path to temporary directory containing files, or None if no files
            
        Raises:
            ValueError: If validation fails (too many files, size exceeded, etc.)
            IOError: If file save operation fails
        """
        if not files:
            return [], None
        
        # Validate before saving
        cls.validate_files(files)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            for file in files:
                # Sanitize filename to prevent directory traversal
                safe_filename = os.path.basename(file.filename or "file")
                if not safe_filename:
                    safe_filename = "file"
                
                file_path = os.path.join(temp_dir, safe_filename)
                
                # Avoid overwriting existing files by appending counter
                counter = 1
                base_path = file_path
                while os.path.exists(file_path):
                    name, ext = os.path.splitext(base_path)
                    file_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                # Save file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                file_paths.append(file_path)
        
        except Exception as e:
            # Clean up temp directory on error
            cls.cleanup(temp_dir)
            raise IOError(f"Failed to save files: {str(e)}") from e
        
        return file_paths, temp_dir
    
    @classmethod
    def cleanup(cls, temp_dir: Optional[str]) -> None:
        """
        Safely remove temporary directory and all contained files.
        
        Args:
            temp_dir: Path to temporary directory to clean up, or None
        """
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                # Log error but don't raise - cleanup failures shouldn't crash the server
                import logging
                logging.error(f"Failed to cleanup temporary directory {temp_dir}: {e}")
    
    @classmethod
    def append_files_to_message(
        cls, 
        message: Union[str, dict], 
        file_paths: List[str],
        session_id: Optional[str] = None
    ) -> Union[str, dict]:
        """
        Append file paths and session ID information to message content.
        
        Args:
            message: Original message content (string or dict)
            file_paths: List of file paths to append
            session_id: Optional session ID to include
            
        Returns:
            Message with appended file paths and session ID
        """
        extra_content = ""
        
        if file_paths:
            extra_content += "\nUploaded Files:\n" + "\n".join(file_paths)
        
        if session_id:
            extra_content += f"\nSession ID: {session_id}"
        
        if not extra_content:
            return message
        
        if isinstance(message, str):
            return message + extra_content
        elif isinstance(message, dict):
            if "content" in message and isinstance(message["content"], str):
                message["content"] += extra_content
            elif "text" in message and isinstance(message["text"], str):
                message["text"] += extra_content
        
        return message
