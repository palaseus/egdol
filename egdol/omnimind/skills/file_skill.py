"""
File Skill for OmniMind
Handles file analysis and operations.
"""

import os
import re
from typing import Dict, Any, List, Optional
from .base import BaseSkill


class FileSkill(BaseSkill):
    """Handles file-related tasks."""
    
    def __init__(self):
        super().__init__()
        self.description = "Handles file analysis and operations"
        self.capabilities = [
            "file analysis",
            "text processing",
            "file reading",
            "content summarization",
            "file operations"
        ]
        
    def can_handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> bool:
        """Check if this is a file-related input."""
        user_input_lower = user_input.lower()
        
        # File keywords
        file_keywords = [
            'file', 'read', 'analyze', 'open', 'load', 'process',
            'content', 'text', 'document', 'txt', 'md', 'json', 'csv'
        ]
        
        # File operations
        file_operations = [
            'read', 'write', 'create', 'delete', 'copy', 'move',
            'list', 'show', 'display', 'find', 'search'
        ]
        
        # File extensions
        file_extensions = ['.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.css']
        
        # Check for file keywords
        if any(keyword in user_input_lower for keyword in file_keywords):
            return True
            
        # Check for file operations
        if any(operation in user_input_lower for operation in file_operations):
            return True
            
        # Check for file extensions
        if any(ext in user_input_lower for ext in file_extensions):
            return True
            
        # Check for file paths
        if '/' in user_input or '\\' in user_input:
            return True
            
        return False
        
    def handle(self, user_input: str, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file-related input."""
        try:
            # Detect file operation type
            operation_type = self._detect_operation_type(user_input)
            
            if operation_type == 'read':
                return self._read_file(user_input)
            elif operation_type == 'analyze':
                return self._analyze_file(user_input)
            elif operation_type == 'list':
                return self._list_files(user_input)
            elif operation_type == 'search':
                return self._search_files(user_input)
            else:
                return self._general_file_response(user_input)
                
        except Exception as e:
            return {
                'content': f"I encountered an error with the file operation: {str(e)}",
                'reasoning': [f"Error: {str(e)}"],
                'metadata': {'skill': 'file', 'error': str(e)}
            }
            
    def _detect_operation_type(self, user_input: str) -> str:
        """Detect the type of file operation."""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['read', 'open', 'load', 'show']):
            return 'read'
        elif any(word in user_input_lower for word in ['analyze', 'examine', 'review']):
            return 'analyze'
        elif any(word in user_input_lower for word in ['list', 'show files', 'directory']):
            return 'list'
        elif any(word in user_input_lower for word in ['search', 'find', 'grep']):
            return 'search'
        else:
            return 'general'
            
    def _read_file(self, user_input: str) -> Dict[str, Any]:
        """Read and display file content."""
        file_path = self._extract_file_path(user_input)
        if not file_path:
            return {
                'content': "I couldn't find a file path in your request. Please specify a file to read.",
                'reasoning': ['No file path detected'],
                'metadata': {'skill': 'file', 'type': 'read'}
            }
            
        if not os.path.exists(file_path):
            return {
                'content': f"File not found: {file_path}",
                'reasoning': ['File does not exist'],
                'metadata': {'skill': 'file', 'type': 'read', 'file_path': file_path}
            }
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Limit content length for display
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
                
            return {
                'content': f"File content ({file_path}):\n\n{content}",
                'reasoning': ['Read file successfully'],
                'metadata': {'skill': 'file', 'type': 'read', 'file_path': file_path, 'size': len(content)}
            }
            
        except Exception as e:
            return {
                'content': f"Error reading file {file_path}: {str(e)}",
                'reasoning': [f"Read error: {str(e)}"],
                'metadata': {'skill': 'file', 'type': 'read', 'file_path': file_path, 'error': str(e)}
            }
            
    def _analyze_file(self, user_input: str) -> Dict[str, Any]:
        """Analyze file content."""
        file_path = self._extract_file_path(user_input)
        if not file_path:
            return {
                'content': "I couldn't find a file path in your request. Please specify a file to analyze.",
                'reasoning': ['No file path detected'],
                'metadata': {'skill': 'file', 'type': 'analyze'}
            }
            
        if not os.path.exists(file_path):
            return {
                'content': f"File not found: {file_path}",
                'reasoning': ['File does not exist'],
                'metadata': {'skill': 'file', 'type': 'analyze', 'file_path': file_path}
            }
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Analyze file
            analysis = self._perform_file_analysis(file_path, content)
            
            return {
                'content': f"File analysis for {file_path}:\n{analysis}",
                'reasoning': ['Analyzed file content'],
                'metadata': {'skill': 'file', 'type': 'analyze', 'file_path': file_path}
            }
            
        except Exception as e:
            return {
                'content': f"Error analyzing file {file_path}: {str(e)}",
                'reasoning': [f"Analysis error: {str(e)}"],
                'metadata': {'skill': 'file', 'type': 'analyze', 'file_path': file_path, 'error': str(e)}
            }
            
    def _list_files(self, user_input: str) -> Dict[str, Any]:
        """List files in a directory."""
        # Extract directory path
        dir_path = self._extract_directory_path(user_input)
        if not dir_path:
            dir_path = '.'  # Current directory
            
        if not os.path.exists(dir_path):
            return {
                'content': f"Directory not found: {dir_path}",
                'reasoning': ['Directory does not exist'],
                'metadata': {'skill': 'file', 'type': 'list', 'dir_path': dir_path}
            }
            
        try:
            files = os.listdir(dir_path)
            files.sort()
            
            # Format file list
            file_list = []
            for file in files:
                file_path = os.path.join(dir_path, file)
                if os.path.isdir(file_path):
                    file_list.append(f"📁 {file}/")
                else:
                    size = os.path.getsize(file_path)
                    file_list.append(f"📄 {file} ({size} bytes)")
                    
            content = f"Files in {dir_path}:\n" + "\n".join(file_list)
            
            return {
                'content': content,
                'reasoning': ['Listed directory contents'],
                'metadata': {'skill': 'file', 'type': 'list', 'dir_path': dir_path, 'file_count': len(files)}
            }
            
        except Exception as e:
            return {
                'content': f"Error listing directory {dir_path}: {str(e)}",
                'reasoning': [f"List error: {str(e)}"],
                'metadata': {'skill': 'file', 'type': 'list', 'dir_path': dir_path, 'error': str(e)}
            }
            
    def _search_files(self, user_input: str) -> Dict[str, Any]:
        """Search for files or content."""
        # Extract search terms
        search_terms = self._extract_search_terms(user_input)
        if not search_terms:
            return {
                'content': "I couldn't find search terms in your request. Please specify what to search for.",
                'reasoning': ['No search terms detected'],
                'metadata': {'skill': 'file', 'type': 'search'}
            }
            
        # This is a simplified search - in practice, you'd implement more sophisticated search
        return {
            'content': f"Search functionality for '{search_terms}' would be implemented here. This is a placeholder response.",
            'reasoning': ['Search request received'],
            'metadata': {'skill': 'file', 'type': 'search', 'terms': search_terms}
        }
        
    def _general_file_response(self, user_input: str) -> Dict[str, Any]:
        """General file response."""
        return {
            'content': "I can help you with file operations like reading, analyzing, and listing files. What specific file task would you like help with?",
            'reasoning': ['General file response'],
            'metadata': {'skill': 'file', 'type': 'general'}
        }
        
    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """Extract file path from user input."""
        # Look for file paths
        patterns = [
            r'["\']([^"\']*\.(txt|md|json|csv|py|js|html|css))["\']',  # Quoted paths
            r'([a-zA-Z0-9_/\\-]+\.(txt|md|json|csv|py|js|html|css))',  # Unquoted paths
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1)
                
        return None
        
    def _extract_directory_path(self, user_input: str) -> Optional[str]:
        """Extract directory path from user input."""
        # Look for directory paths
        patterns = [
            r'["\']([^"\']*[/\\][^"\']*)["\']',  # Quoted paths
            r'([a-zA-Z0-9_/\\-]+[/\\])',  # Unquoted paths
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1)
                
        return None
        
    def _extract_search_terms(self, user_input: str) -> Optional[str]:
        """Extract search terms from user input."""
        # Remove common words
        cleaned = re.sub(r'\b(search|find|look for|grep)\b', '', user_input, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        if cleaned:
            return cleaned
        return None
        
    def _perform_file_analysis(self, file_path: str, content: str) -> str:
        """Perform analysis on file content."""
        analysis = []
        
        # Basic file info
        file_size = len(content)
        line_count = len(content.splitlines())
        word_count = len(content.split())
        
        analysis.append(f"File size: {file_size} characters")
        analysis.append(f"Lines: {line_count}")
        analysis.append(f"Words: {word_count}")
        
        # File type analysis
        if file_path.endswith('.py'):
            analysis.append("File type: Python script")
            # Count functions and classes
            function_count = content.count('def ')
            class_count = content.count('class ')
            analysis.append(f"Functions: {function_count}")
            analysis.append(f"Classes: {class_count}")
        elif file_path.endswith('.js'):
            analysis.append("File type: JavaScript")
            function_count = content.count('function ')
            analysis.append(f"Functions: {function_count}")
        elif file_path.endswith('.html'):
            analysis.append("File type: HTML")
            tag_count = len(re.findall(r'<[^>]+>', content))
            analysis.append(f"HTML tags: {tag_count}")
        elif file_path.endswith('.md'):
            analysis.append("File type: Markdown")
            header_count = content.count('#')
            analysis.append(f"Headers: {header_count}")
        elif file_path.endswith('.json'):
            analysis.append("File type: JSON")
            try:
                import json
                json.loads(content)
                analysis.append("JSON is valid")
            except:
                analysis.append("JSON is invalid")
        elif file_path.endswith('.csv'):
            analysis.append("File type: CSV")
            row_count = content.count('\n')
            analysis.append(f"Rows: {row_count}")
        else:
            analysis.append("File type: Unknown")
            
        return "\n".join(analysis)
