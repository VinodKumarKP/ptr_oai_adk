import os
import re
import json
import hashlib
import urllib.parse

def macro_env(*args, **kwargs):
    """
    Macro to get environment variable.
    Usage: {{ ENV VAR_NAME [default_value] }}
    """
    if not args:
        return "Error: ENV macro requires a variable name."
    
    var_name = args[0]
    default_val = args[1] if len(args) > 1 else ""
    
    return os.environ.get(var_name, default_val)

def macro_calc(*args, **kwargs):
    """
    Macro to evaluate a mathematical expression.
    Usage: {{ CALC expression }}
    """
    if not args:
        return "Error: CALC macro requires an expression."
    
    expression = " ".join(args)
    
    if not re.match(r'^[\d\.\+\-\*\/\(\)\s%]+$', expression):
        return "Error: Invalid characters in CALC expression. Only numbers and basic math operators allowed."

    try:
        # pylint: disable=eval-used
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

def macro_url_encode(*args, **kwargs):
    """
    Macro to URL encode a string.
    Usage: {{ URL_ENCODE string }}
    """
    if not args:
        return ""
    
    text = " ".join(args)
    return urllib.parse.quote(text)

def macro_hash(*args, **kwargs):
    """
    Macro to generate a hash of a string.
    Usage: {{ HASH string [algorithm] }}
    Default algorithm: md5
    Supported algorithms: md5, sha1, sha256, sha512
    """
    if not args:
        return "Error: HASH macro requires a string."
    
    text = args[0]
    algorithm = args[1].lower() if len(args) > 1 else 'md5'
    
    if algorithm not in hashlib.algorithms_available:
            return f"Error: Unsupported hash algorithm '{algorithm}'."
            
    try:
        h = hashlib.new(algorithm)
        h.update(text.encode('utf-8'))
        return h.hexdigest()
    except Exception as e:
        return f"Error generating hash: {e}"

def macro_http_get(*args, **kwargs):
    """
    Macro to fetch data from a URL using HTTP GET.
    Usage: {{ HTTP_GET url [headers_json] }}
    
    url: The URL to fetch
    headers_json: Optional JSON string with headers (e.g., '{"Authorization": "Bearer token"}')
    
    Examples:
    {{ HTTP_GET https://api.example.com/data }}
    {{ HTTP_GET https://api.example.com/data '{"Authorization": "Bearer xyz"}' }}
    """
    if not args:
        return "Error: HTTP_GET macro requires a URL."
    
    url = args[0]
    headers = {}
    
    if len(args) > 1:
        try:
            headers = json.loads(args[1])
        except json.JSONDecodeError:
            return "Error: headers_json must be valid JSON."
    
    try:
        import urllib.request
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
            return content
            
    except ImportError:
        return "Error: urllib is not available."
    except Exception as e:
        return f"Error fetching URL {url}: {e}"

def macro_sql_query(*args, **kwargs):
    """
    Macro to query a SQLite database.
    Usage: {{ SQL_QUERY db_path query }}
    
    db_path: Path to SQLite database file (relative to project_root)
    query: SQL query to execute
    
    Returns results as JSON array.
    
    Example:
    {{ SQL_QUERY test.db "SELECT * FROM users LIMIT 10" }}
    """
    if len(args) < 2:
        return "Error: SQL_QUERY macro requires db_path and query."
    
    db_path = args[0]
    query = args[1]
    processor = kwargs.get('processor')
    
    # Resolve path
    if not os.path.isabs(db_path) and processor:
        db_path = os.path.join(processor.project_root, db_path)
    
    if not os.path.exists(db_path):
        return f"Error: Database file not found at {db_path}"
    
    try:
        import sqlite3
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = []
        for row in rows:
            results.append(dict(row))
        
        conn.close()
        
        return json.dumps(results, indent=2)
        
    except ImportError:
        return "Error: sqlite3 module not available."
    except Exception as e:
        return f"Error executing SQL query: {e}"

def macro_truncate(*args, **kwargs):
    """
    Macro to truncate text to a maximum length.
    Usage: {{ TRUNCATE text max_length [suffix] }}
    Default suffix: "..."
    """
    if len(args) < 2:
        return "Error: TRUNCATE macro requires text and max_length."
    
    text = args[0]
    try:
        max_length = int(args[1])
    except ValueError:
        return "Error: max_length must be an integer."
    
    suffix = args[2] if len(args) > 2 else "..."
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
