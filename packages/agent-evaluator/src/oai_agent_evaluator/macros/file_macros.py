import os
import json
import base64

def _read_pdf(file_path: str) -> str:
    """Reads text from a PDF file using pypdf."""
    try:
        import pypdf
    except ImportError:
        return "Error: pypdf package is required to read PDF files. Please install it using 'pip install pypdf'."

    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF file {file_path}: {e}"

def _read_docx(file_path: str) -> str:
    """Reads text and tables from a DOCX file using python-docx."""
    try:
        import docx
        from docx.document import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return "Error: python-docx package is required to read DOCX files. Please install it using 'pip install python-docx'."

    try:
        doc = docx.Document(file_path)
        content = []
        
        def iter_block_items(parent):
            if isinstance(parent, Document):
                parent_elm = parent.element.body
            else:
                parent_elm = parent.element
                
            for child in parent_elm.iterchildren():
                if child.tag.endswith('p'):
                    yield Paragraph(child, parent)
                elif child.tag.endswith('tbl'):
                    yield Table(child, parent)

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                content.append(block.text)
            elif isinstance(block, Table):
                table_text = []
                for row in block.rows:
                    row_cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                    table_text.append("| " + " | ".join(row_cells) + " |")
                content.append("\n".join(table_text))
        
        return "\n\n".join(content)
    except Exception as e:
        return f"Error reading DOCX file {file_path}: {e}"

def _read_ppt(file_path: str) -> str:
    """Reads text from a PPT/PPTX file using python-pptx."""
    try:
        from pptx import Presentation
    except ImportError:
        return "Error: python-pptx package is required to read PPT/PPTX files. Please install it using 'pip install python-pptx'."

    try:
        prs = Presentation(file_path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    except Exception as e:
        return f"Error reading PPT file {file_path}: {e}"

def _read_excel(file_path: str) -> str:
    """Reads text from an Excel file using pandas and converts to CSV format."""
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas and openpyxl packages are required to read Excel files. Please install them using 'pip install pandas openpyxl'."

    try:
        # Read all sheets
        xls = pd.ExcelFile(file_path)
        output = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            csv_data = df.to_csv(index=False)
            output.append(f"Sheet: {sheet_name}\n{csv_data}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Error reading Excel file {file_path}: {e}"

def _read_image_ocr(file_path: str) -> str:
    """Reads text from an image file using pytesseract (OCR)."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "Error: pillow and pytesseract packages are required to read image files. Please install them using 'pip install pillow pytesseract'."

    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error reading image file {file_path}: {e}"

def macro_open(*args, **kwargs):
    """Macro to read file content. Usage: {{ OPEN file_path [file_type] }}"""
    if not args:
        return "Error: OPEN macro requires a file path."
    
    file_path = args[0]
    processor = kwargs.get('processor')
    
    # Resolve path relative to project_root if not absolute
    if not os.path.isabs(file_path) and processor:
        file_path = os.path.join(processor.project_root, file_path)
        
    if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

    # Determine file type
    file_type = None
    if len(args) > 1:
        file_type = args[1].lower()
    else:
        _, ext = os.path.splitext(file_path)
        if ext:
            file_type = ext[1:].lower()

    try:
        if file_type == 'pdf':
            return _read_pdf(file_path)
        elif file_type == 'docx':
            return _read_docx(file_path)
        elif file_type in ['ppt', 'pptx']:
            return _read_ppt(file_path)
        elif file_type in ['xls', 'xlsx']:
            return _read_excel(file_path)
        elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']:
            return _read_image_ocr(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def macro_image(*args, **kwargs):
    """
    Macro to read an image file and return its base64 encoded string.
    Usage: {{ IMAGE file_path }}
    """
    if not args:
        return "Error: IMAGE macro requires a file path."
    
    file_path = args[0]
    processor = kwargs.get('processor')
    
    # Resolve path relative to project_root if not absolute
    if not os.path.isabs(file_path) and processor:
        file_path = os.path.join(processor.project_root, file_path)
        
    if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        return f"Error reading image file {file_path}: {e}"

def macro_path(*args, **kwargs):
    """
    Macro to resolve a path relative to the project root.
    Usage: {{ PATH relative_path [project_root] }}
    If project_root is not provided, uses the default project root.
    """
    if not args:
        return "Error: PATH macro requires a relative path."
    
    path = args[0]
    processor = kwargs.get('processor')
    
    # If absolute path, return as is
    if os.path.isabs(path):
        return path
        
    root = args[1] if len(args) > 1 else (processor.project_root if processor else None)
    
    if not root:
        return "Error: Project root not set and not provided."
        
    return os.path.join(root, path)

def macro_list_files(*args, **kwargs):
    """
    Macro to list files in a directory.
    Usage: {{ LIST_FILES directory_path [pattern] }}
    Returns a JSON list of filenames.
    """
    if not args:
        return "Error: LIST_FILES macro requires a directory path."
    
    dir_path = args[0]
    pattern = args[1] if len(args) > 1 else None
    processor = kwargs.get('processor')
    
    # Resolve path
    if not os.path.isabs(dir_path) and processor:
        dir_path = os.path.join(processor.project_root, dir_path)
        
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return f"Error: Directory not found at {dir_path}"
        
    try:
        files = os.listdir(dir_path)
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        
        # Sort for deterministic output
        files.sort()
        return json.dumps(files)
    except Exception as e:
        return f"Error listing files: {e}"

def macro_concat(*args, **kwargs):
    """
    Macro to concatenate content from multiple files.
    Usage: {{ CONCAT file1 file2 ... }}
    """
    if not args:
        return "Error: CONCAT macro requires at least one file path."
    
    contents = []
    processor = kwargs.get('processor')
    
    for file_path in args:
        # Resolve path relative to project_root if not absolute
        if not os.path.isabs(file_path) and processor:
            file_path = os.path.join(processor.project_root, file_path)
        
        if not os.path.exists(file_path):
            contents.append(f"Error: File not found at {file_path}")
            continue
        
        try:
            # Use the existing OPEN macro logic
            # We can call macro_open directly but need to pass processor
            result = macro_open(file_path, processor=processor)
            contents.append(result)
        except Exception as e:
            contents.append(f"Error reading file {file_path}: {e}")
    
    return "\n\n---\n\n".join(contents)

def macro_sample(*args, **kwargs):
    """
    Macro to extract sample lines from a file.
    Usage: {{ SAMPLE file_path n_lines [random] }}
    If 'random' is specified as third argument, sample random lines.
    Otherwise, returns first n_lines.
    """
    if len(args) < 2:
        return "Error: SAMPLE macro requires file_path and n_lines."
    
    file_path = args[0]
    try:
        n_lines = int(args[1])
    except ValueError:
        return "Error: n_lines must be an integer."
    
    is_random = len(args) > 2 and args[2].lower() == 'random'
    processor = kwargs.get('processor')
    
    # Resolve path
    if not os.path.isabs(file_path) and processor:
        file_path = os.path.join(processor.project_root, file_path)
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        import random
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return ""
        
        if is_random:
            # Random sample
            sample_size = min(n_lines, len(lines))
            sampled_lines = random.sample(lines, sample_size)
        else:
            # First n lines
            sampled_lines = lines[:n_lines]
        
        return "".join(sampled_lines)
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def macro_json_extract(*args, **kwargs):
    """
    Macro to extract data from a JSON file using JSONPath.
    Usage: {{ JSON_EXTRACT file_path json_path }}
    """
    if len(args) < 2:
        return "Error: JSON_EXTRACT macro requires file_path and json_path."
    
    file_path = args[0]
    json_path = args[1]
    processor = kwargs.get('processor')
    
    # Resolve path
    if not os.path.isabs(file_path) and processor:
        file_path = os.path.join(processor.project_root, file_path)
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Simple JSONPath implementation for common patterns
        result = _jsonpath_extract(data, json_path)
        
        # Convert result to string
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file {file_path}: {e}"
    except Exception as e:
        return f"Error extracting JSON: {e}"

def _jsonpath_extract(data, path):
    """Simple JSONPath implementation for common patterns."""
    import re
    if not path.startswith('$'):
        raise ValueError("JSONPath must start with $")
    
    # Remove leading $
    path = path[1:]
    if not path:
        return data
    
    # Remove leading .
    if path.startswith('.'):
        path = path[1:]
    
    current = data
    parts = re.split(r'\.|\[', path)
    
    for part in parts:
        if not part:
            continue
        
        # Handle array index like "0]"
        if part.endswith(']'):
            part = part[:-1]
            
            # Handle wildcards
            if part == '*':
                if isinstance(current, list):
                    return current
                elif isinstance(current, dict):
                    return list(current.values())
                return current
            
            # Handle numeric index
            try:
                idx = int(part)
                if isinstance(current, (list, tuple)):
                    current = current[idx]
                else:
                    return None
            except (ValueError, IndexError, KeyError):
                return None
        else:
            # Handle object key
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
    
    return current

def macro_template(*args, **kwargs):
    """
    Macro to populate a template file with variables.
    Usage: {{ TEMPLATE template_file variables_json }}
    """
    if len(args) < 2:
        return "Error: TEMPLATE macro requires template_file and variables_json."
    
    template_file = args[0]
    variables_arg = args[1]
    processor = kwargs.get('processor')
    
    # Resolve path
    if not os.path.isabs(template_file) and processor:
        template_file = os.path.join(processor.project_root, template_file)
    
    if not os.path.exists(template_file):
        return f"Error: Template file not found at {template_file}"
    
    try:
        # Read template
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Parse variables
        if isinstance(variables_arg, str):
            variables = json.loads(variables_arg)
        else:
            variables = variables_arg
        
        # Simple template substitution using {key} syntax
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in variables: {e}"
    except Exception as e:
        return f"Error processing template: {e}"
