import os
import sys
import unittest
import json
import base64
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

# Add current directory to path for local import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the new package structure
from oai_agent_evaluator.macros import MacroProcessor
# Import internal functions for direct testing
from oai_agent_evaluator.macros.file_macros import _read_pdf, _read_ppt, _read_excel, _read_image_ocr, _read_docx


class TestMacroProcessor(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.processor = MacroProcessor(project_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # ========== EXISTING MACRO TESTS ==========

    def test_macro_date(self):
        # Test default date (today)
        result = self.processor.process("Today is {{ DATE }}")
        expected = f"Today is {datetime.now().strftime('%Y-%m-%d')}"
        self.assertEqual(result, expected)

        # Test custom format
        result = self.processor.process("Today is {{ DATE %d-%m-%Y }}")
        expected = f"Today is {datetime.now().strftime('%d-%m-%Y')}"
        self.assertEqual(result, expected)

        # Test offset
        result = self.processor.process("Tomorrow is {{ DATE %Y-%m-%d 1 }}")
        expected = f"Tomorrow is {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}"
        self.assertEqual(result, expected)
        
        # Test negative offset
        result = self.processor.process("Yesterday: {{ DATE %Y-%m-%d -1 }}")
        expected = f"Yesterday: {(datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d')}"
        self.assertEqual(result, expected)
        
        # Test invalid offset
        result = self.processor.process("Error: {{ DATE %Y-%m-%d invalid }}")
        self.assertIn("Error: Invalid offset", result)

    def test_macro_now(self):
        # Test default format
        result = self.processor.process("Now is {{ NOW }}")
        self.assertNotIn("{{ NOW }}", result)
        
        # Test custom format
        result = self.processor.process("Now is {{ NOW %H:%M }}")
        self.assertRegex(result, r"Now is \d{2}:\d{2}")

    def test_macro_uuid(self):
        result = self.processor.process("ID: {{ UUID }}")
        self.assertRegex(result, r"ID: [0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
        
        # Test multiple UUIDs are different
        result1 = self.processor.process("{{ UUID }}")
        result2 = self.processor.process("{{ UUID }}")
        self.assertNotEqual(result1, result2)

    def test_macro_random_int(self):
        result = self.processor.process("Value: {{ RANDOM_INT 10 20 }}")
        value = int(result.split(": ")[1])
        self.assertTrue(10 <= value <= 20)
        
        # Test edge cases
        result = self.processor.process("{{ RANDOM_INT 5 5 }}")
        self.assertEqual(result, "5")
        
        # Error cases
        result = self.processor.process("{{ RANDOM_INT 10 }}")
        self.assertIn("Error: RANDOM_INT macro requires min and max", result)
        
        result = self.processor.process("{{ RANDOM_INT a b }}")
        self.assertIn("Error: RANDOM_INT arguments must be integers", result)

    def test_macro_random_choice(self):
        result = self.processor.process("Pick: {{ RANDOM_CHOICE A B C }}")
        choice = result.split(": ")[1]
        self.assertIn(choice, ["A", "B", "C"])
        
        # Single item
        result = self.processor.process("{{ RANDOM_CHOICE single }}")
        self.assertEqual(result, "single")
        
        # Error case
        result = self.processor.process("{{ RANDOM_CHOICE }}")
        self.assertIn("Error: RANDOM_CHOICE macro requires at least one item", result)

    def test_macro_env(self):
        os.environ["TEST_VAR"] = "test_value"
        result = self.processor.process("Env: {{ ENV TEST_VAR }}")
        self.assertEqual(result, "Env: test_value")
        
        # Test default
        result = self.processor.process("Env: {{ ENV MISSING_VAR default }}")
        self.assertEqual(result, "Env: default")
        
        # Test no default
        del os.environ["TEST_VAR"]
        result = self.processor.process("{{ ENV TEST_VAR }}")
        self.assertEqual(result, "")
        
        # Error case
        result = self.processor.process("{{ ENV }}")
        self.assertIn("Error: ENV macro requires a variable name", result)

    def test_macro_calc(self):
        # Basic operations
        result = self.processor.process("Result: {{ CALC 5 * 10 + 2 }}")
        self.assertEqual(result, "Result: 52")
        
        # Division - handles both int and float
        result = self.processor.process("{{ CALC 100 / 4 }}")
        self.assertIn(result, ["25", "25.0"])
        
        # Parentheses
        result = self.processor.process("{{ CALC (10 + 5) * 2 }}")
        self.assertIn(result, ["30", "30.0"])
        
        # Decimal
        result = self.processor.process("{{ CALC 10.5 * 2 }}")
        self.assertEqual(result, "21.0")
        
        # Error cases
        result = self.processor.process("{{ CALC }}")
        self.assertIn("Error: CALC macro requires an expression", result)
        
        result = self.processor.process("{{ CALC 5 + a }}")
        self.assertIn("Error: Invalid characters", result)
        
        result = self.processor.process("{{ CALC 5 / 0 }}")
        self.assertIn("Error evaluating expression", result)

    def test_macro_base64(self):
        result = self.processor.process("Encoded: {{ BASE64 hello world }}")
        expected = base64.b64encode(b"hello world").decode("utf-8")
        self.assertEqual(result, f"Encoded: {expected}")
        
        # Empty args
        result = self.processor.process("{{ BASE64 }}")
        self.assertEqual(result, "")

    def test_macro_json_escape(self):
        result = self.processor.process('Escaped: {{ JSON_ESCAPE "a \\"quote\\"" }}')
        self.assertEqual(result, 'Escaped: a \\"quote\\"')
        
        # Empty args
        result = self.processor.process("{{ JSON_ESCAPE }}")
        self.assertEqual(result, "")

    def test_macro_url_encode(self):
        result = self.processor.process("Encoded: {{ URL_ENCODE hello world&more }}")
        self.assertEqual(result, "Encoded: hello%20world%26more")
        
        # Empty args
        result = self.processor.process("{{ URL_ENCODE }}")
        self.assertEqual(result, "")

    def test_macro_hash(self):
        result = self.processor.process("Hash: {{ HASH test }}")
        self.assertEqual(result, "Hash: 098f6bcd4621d373cade4e832627b4f6")
        
        result = self.processor.process("Hash: {{ HASH test sha256 }}")
        self.assertEqual(result, "Hash: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
        
        # SHA1
        result = self.processor.process("{{ HASH test sha1 }}")
        self.assertEqual(result, "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3")
        
        # Error cases
        result = self.processor.process("{{ HASH }}")
        self.assertIn("Error: HASH macro requires a string", result)
        
        result = self.processor.process("{{ HASH test invalid_algo }}")
        self.assertIn("Error: Unsupported hash algorithm", result)

    def test_macro_if(self):
        # Truthy cases
        result = self.processor.process("Result: {{ IF true Yes No }}")
        self.assertEqual(result, "Result: Yes")
        
        result = self.processor.process("{{ IF yes Yes No }}")
        self.assertEqual(result, "Yes")
        
        result = self.processor.process("{{ IF 1 Yes No }}")
        self.assertEqual(result, "Yes")
        
        result = self.processor.process("{{ IF on Yes No }}")
        self.assertEqual(result, "Yes")
        
        # Falsy cases
        result = self.processor.process("Result: {{ IF false Yes No }}")
        self.assertEqual(result, "Result: No")
        
        result = self.processor.process("{{ IF no Yes No }}")
        self.assertEqual(result, "No")
        
        result = self.processor.process("{{ IF 0 Yes No }}")
        self.assertEqual(result, "No")
        
        result = self.processor.process("{{ IF '' Yes No }}")
        self.assertEqual(result, "No")
        
        # Error case
        result = self.processor.process("{{ IF true }}")
        self.assertIn("Error: IF macro requires condition", result)

    def test_macro_set_get(self):
        result = self.processor.process("Set: {{ SET my_var 123 }}, Get: {{ GET my_var }}")
        self.assertEqual(result, "Set: 123, Get: 123")
        
        # Test default value
        result = self.processor.process("{{ GET missing_var default_value }}")
        self.assertEqual(result, "default_value")
        
        # Error cases
        result = self.processor.process("{{ SET }}")
        self.assertIn("Error: SET macro requires var_name", result)
        
        result = self.processor.process("{{ GET }}")
        self.assertIn("Error: GET macro requires a variable name", result)

    def test_macro_path(self):
        result = self.processor.process("Path: {{ PATH data/file.txt }}")
        expected = os.path.join(self.test_dir, "data/file.txt")
        self.assertEqual(result, f"Path: {expected}")
        
        abs_path = "/tmp/absolute/file.txt"
        result = self.processor.process(f"Path: {{{{ PATH {abs_path} }}}}")
        self.assertEqual(result, f"Path: {abs_path}")
        
        # Custom root
        # result = self.processor.process("{{ PATH config.json /custom/root }}")
        # self.assertEqual(result, "/custom/root/config.json")
        
        # Error cases
        result = self.processor.process("{{ PATH }}")
        self.assertIn("Error: PATH macro requires a relative path", result)
        
        # Test with no root
        proc_no_root = MacroProcessor(project_root="")
        result = proc_no_root.process("{{ PATH file.txt }}")
        self.assertIn("Error: Project root not set", result)

    def test_macro_open(self):
        file_path = os.path.join(self.test_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("Hello File Content")
            
        result = self.processor.process("Content: {{ OPEN test.txt }}")
        self.assertEqual(result, "Content: Hello File Content")
        
        # Test absolute path
        result = self.processor.process(f"Content: {{{{ OPEN {file_path} }}}}")
        self.assertEqual(result, "Content: Hello File Content")
        
        # Error cases
        result = self.processor.process("{{ OPEN }}")
        self.assertIn("Error: OPEN macro requires a file path", result)
        
        result = self.processor.process("{{ OPEN missing.txt }}")
        self.assertIn("Error: File not found", result)

    def test_macro_list_files(self):
        os.makedirs(os.path.join(self.test_dir, "data"))
        open(os.path.join(self.test_dir, "data", "a.txt"), "w").close()
        open(os.path.join(self.test_dir, "data", "b.json"), "w").close()
        open(os.path.join(self.test_dir, "data", "c.log"), "w").close()
        
        result = self.processor.process("Files: {{ LIST_FILES data }}")
        files = json.loads(result.split(": ")[1])
        self.assertEqual(sorted(files), ["a.txt", "b.json", "c.log"])
        
        # Test pattern matching
        result = self.processor.process("{{ LIST_FILES data *.txt }}")
        files = json.loads(result)
        self.assertEqual(files, ["a.txt"])
        
        # Error cases
        result = self.processor.process("{{ LIST_FILES }}")
        self.assertIn("Error: LIST_FILES macro requires a directory path", result)
        
        result = self.processor.process("{{ LIST_FILES missing_dir }}")
        self.assertIn("Error: Directory not found", result)

    def test_macro_loop(self):
        result = self.processor.process('{{ LOOP 3 "Item" ", " }}')
        self.assertEqual(result, "Item, Item, Item")
        
        # With LOOP_INDEX
        template = '{{ LOOP 3 "Index [[ GET LOOP_INDEX ]]" "-" }}'
        result = self.processor.process(template)
        self.assertEqual(result, "Index 0-Index 1-Index 2")
        
        # With LOOP_COUNT
        template = '{{ LOOP 3 "Count [[ GET LOOP_COUNT ]]" ", " }}'
        result = self.processor.process(template)
        self.assertEqual(result, "Count 1, Count 2, Count 3")
        
        # Error cases
        result = self.processor.process("{{ LOOP }}")
        self.assertIn("Error: LOOP macro requires count and template", result)
        
        result = self.processor.process('{{ LOOP invalid "template" }}')
        self.assertIn("Error: LOOP count must be an integer", result)

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('faker') is not None,
        "faker package not installed"
    )
    @patch('faker.Faker')
    def test_macro_faker(self, mock_faker):
        mock_instance = MagicMock()
        mock_instance.name.return_value = "John Doe"
        mock_instance.email.return_value = "john@example.com"
        
        mock_faker.return_value = mock_instance
        
        result = self.processor.process("Name: {{ FAKER name }}")
        self.assertEqual(result, "Name: John Doe")
        
        result = self.processor.process("{{ FAKER email }}")
        self.assertEqual(result, "john@example.com")
        
        # Error cases
        result = self.processor.process("{{ FAKER }}")
        self.assertIn("Error: FAKER macro requires a provider method name", result)
        
        # Test exception handling
        mock_instance.error_method.side_effect = Exception("Some error")
        result = self.processor.process("{{ FAKER error_method }}")
        self.assertIn("Error executing Faker method", result)

    @unittest.skipUnless(
        __import__('importlib').util.find_spec('faker') is not None,
        "faker package not installed"
    )
    @patch('faker.Faker')
    def test_macro_faker_locale(self, mock_faker):
        mock_instance = MagicMock()
        mock_instance.name.return_value = "Jean Dupont"
        mock_faker.return_value = mock_instance
        
        result = self.processor.process("Name: {{ FAKER_LOCALE fr_FR name }}")
        mock_faker.assert_called_with('fr_FR')
        self.assertEqual(result, "Name: Jean Dupont")
        
        # Error cases
        result = self.processor.process("{{ FAKER_LOCALE fr_FR }}")
        self.assertIn("Error: FAKER_LOCALE macro requires a locale", result)
        
        result = self.processor.process("{{ FAKER_LOCALE }}")
        self.assertIn("Error: FAKER_LOCALE macro requires a locale", result)

    def test_macro_image(self):
        file_path = os.path.join(self.test_dir, "test.png")
        with open(file_path, "wb") as f:
            f.write(b"fake image content")
            
        result = self.processor.process("Image: {{ IMAGE test.png }}")
        expected = base64.b64encode(b"fake image content").decode('utf-8')
        self.assertEqual(result, f"Image: {expected}")
        
        # Error cases
        result = self.processor.process("{{ IMAGE }}")
        self.assertIn("Error: IMAGE macro requires a file path", result)
        
        result = self.processor.process("{{ IMAGE missing.png }}")
        self.assertIn("Error: File not found", result)

    # ========== NEW MACRO TESTS ==========

    def test_macro_concat(self):
        # Create test files
        with open(os.path.join(self.test_dir, "file1.txt"), "w") as f:
            f.write("Content 1")
        with open(os.path.join(self.test_dir, "file2.txt"), "w") as f:
            f.write("Content 2")
        with open(os.path.join(self.test_dir, "file3.txt"), "w") as f:
            f.write("Content 3")
        
        # Test basic concatenation
        result = self.processor.process("{{ CONCAT file1.txt file2.txt }}")
        self.assertIn("Content 1", result)
        self.assertIn("Content 2", result)
        self.assertIn("---", result)
        
        # Test three files
        result = self.processor.process("{{ CONCAT file1.txt file2.txt file3.txt }}")
        self.assertIn("Content 1", result)
        self.assertIn("Content 2", result)
        self.assertIn("Content 3", result)
        
        # Test with missing file
        result = self.processor.process("{{ CONCAT file1.txt missing.txt }}")
        self.assertIn("Content 1", result)
        self.assertIn("Error: File not found", result)
        
        # Error case - no files
        result = self.processor.process("{{ CONCAT }}")
        self.assertIn("Error: CONCAT macro requires at least one file path", result)

    def test_macro_sample(self):
        # Create test file with multiple lines
        log_file = os.path.join(self.test_dir, "logs.txt")
        with open(log_file, "w") as f:
            for i in range(100):
                f.write(f"Log entry {i}\n")
        
        # Test first n lines
        result = self.processor.process("{{ SAMPLE logs.txt 5 }}")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 5)
        self.assertIn("Log entry 0", result)
        self.assertIn("Log entry 4", result)
        
        # Test random sampling
        result = self.processor.process("{{ SAMPLE logs.txt 10 random }}")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 10)
        
        # Test edge case - request more lines than available
        with open(os.path.join(self.test_dir, "small.txt"), "w") as f:
            f.write("Line 1\nLine 2\n")
        result = self.processor.process("{{ SAMPLE small.txt 10 }}")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 2)
        
        # Error cases
        result = self.processor.process("{{ SAMPLE }}")
        self.assertIn("Error: SAMPLE macro requires file_path and n_lines", result)
        
        result = self.processor.process("{{ SAMPLE logs.txt invalid }}")
        self.assertIn("Error: n_lines must be an integer", result)
        
        result = self.processor.process("{{ SAMPLE missing.txt 5 }}")
        self.assertIn("Error: File not found", result)

    def test_macro_truncate(self):
        # Basic truncation
        result = self.processor.process("{{ TRUNCATE 'This is a long text' 10 }}")
        self.assertEqual(len(result), 10)
        self.assertTrue(result.endswith("..."))
        
        # Custom suffix
        result = self.processor.process("{{ TRUNCATE 'Long text here' 8 '[...]' }}")
        self.assertTrue(result.endswith("[...]"))
        self.assertEqual(len(result), 8)
        
        # Text shorter than max_length
        result = self.processor.process("{{ TRUNCATE 'Short' 100 }}")
        self.assertEqual(result, "Short")
        
        # Empty suffix
        result = self.processor.process("{{ TRUNCATE 'Text here' 4 '' }}")
        self.assertEqual(result, "Text")
        
        # Error cases
        result = self.processor.process("{{ TRUNCATE }}")
        self.assertIn("Error: TRUNCATE macro requires text and max_length", result)
        
        result = self.processor.process("{{ TRUNCATE text invalid }}")
        self.assertIn("Error: max_length must be an integer", result)

    def test_macro_repeat(self):
        # Basic repeat
        result = self.processor.process("{{ REPEAT test 3 ', ' }}")
        self.assertEqual(result, "test, test, test")
        
        # With newline
        result = self.processor.process("{{ REPEAT line 2 '\n' }}")
        self.assertEqual(result, "line\nline")
        
        # No separator
        result = self.processor.process("{{ REPEAT word 4 }}")
        self.assertEqual(result, "wordwordwordword")
        
        # Single repeat
        result = self.processor.process("{{ REPEAT text 1 }}")
        self.assertEqual(result, "text")
        
        # Error cases
        result = self.processor.process("{{ REPEAT }}")
        self.assertIn("Error: REPEAT macro requires text and count", result)
        
        result = self.processor.process("{{ REPEAT text invalid }}")
        self.assertIn("Error: count must be an integer", result)

    def test_macro_json_extract(self):
        # Create JSON file
        json_data = {
            "users": [
                {"name": "Alice", "email": "alice@example.com", "age": 30},
                {"name": "Bob", "email": "bob@example.com", "age": 25}
            ],
            "config": {
                "settings": {
                    "theme": "dark",
                    "language": "en"
                }
            },
            "count": 42
        }
        json_file = os.path.join(self.test_dir, "data.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f)
        
        # Test simple extraction
        result = self.processor.process("{{ JSON_EXTRACT data.json $.users[0].name }}")
        self.assertEqual(result, "Alice")
        
        # Test nested extraction
        result = self.processor.process("{{ JSON_EXTRACT data.json $.config.settings.theme }}")
        self.assertEqual(result, "dark")
        
        # Test number extraction
        result = self.processor.process("{{ JSON_EXTRACT data.json $.count }}")
        self.assertEqual(result, "42")
        
        # Test array wildcard
        result = self.processor.process("{{ JSON_EXTRACT data.json $.users[*] }}")
        data = json.loads(result)
        self.assertEqual(len(data), 2)
        
        # Error cases
        result = self.processor.process("{{ JSON_EXTRACT }}")
        self.assertIn("Error: JSON_EXTRACT macro requires file_path and json_path", result)
        
        result = self.processor.process("{{ JSON_EXTRACT missing.json $.key }}")
        self.assertIn("Error: File not found", result)
        
        # Invalid JSON
        with open(os.path.join(self.test_dir, "invalid.json"), "w") as f:
            f.write("not valid json")
        result = self.processor.process("{{ JSON_EXTRACT invalid.json $.key }}")
        self.assertIn("Error: Invalid JSON", result)

    def test_macro_template(self):
        # Create template file
        template_file = os.path.join(self.test_dir, "template.txt")
        with open(template_file, "w") as f:
            f.write("Hello {name}, your appointment is on {date} at {time}.")
        
        # Test basic template
        variables = json.dumps({"name": "John", "date": "2024-01-15", "time": "2pm"})
        result = self.processor.process(f"{{{{ TEMPLATE template.txt '{variables}' }}}}")
        self.assertIn("Hello John", result)
        self.assertIn("2024-01-15", result)
        self.assertIn("2pm", result)
        
        # Test partial substitution
        variables = json.dumps({"name": "Alice"})
        result = self.processor.process(f"{{{{ TEMPLATE template.txt '{variables}' }}}}")
        self.assertIn("Hello Alice", result)
        self.assertIn("{date}", result)  # Unsubstituted
        
        # Error cases
        result = self.processor.process("{{ TEMPLATE }}")
        self.assertIn("Error: TEMPLATE macro requires template_file and variables_json", result)
        
        result = self.processor.process("{{ TEMPLATE missing.txt '{}' }}")
        self.assertIn("Error: Template file not found", result)
        
        result = self.processor.process("{{ TEMPLATE template.txt 'invalid json' }}")
        self.assertIn("Error: Invalid JSON", result)

    # ========== FILE READER TESTS WITH MOCKS ==========

    def test_read_pdf(self):
        with patch.dict('sys.modules', {'pypdf': MagicMock()}):
            import pypdf
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF Content"
            mock_reader.pages = [mock_page]
            pypdf.PdfReader.return_value = mock_reader
            
            pdf_path = os.path.join(self.test_dir, "test.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF")
                
            result = self.processor.process("{{ OPEN test.pdf }}")
            self.assertIn("PDF Content", result)

    def test_read_pdf_missing_dependency(self):
        with patch.dict('sys.modules', {'pypdf': None}):
            pdf_path = os.path.join(self.test_dir, "test.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF")
            result = self.processor.process("{{ OPEN test.pdf }}")
            self.assertIn("Error: pypdf package is required", result)

    def test_read_docx_missing_dep(self):
        with patch.dict('sys.modules', {'docx': None}):
            docx_path = os.path.join(self.test_dir, "test.docx")
            with open(docx_path, "wb") as f:
                f.write(b"PK")
            result = self.processor.process("{{ OPEN test.docx }}")
            self.assertIn("Error: python-docx package is required", result)

    def test_read_ppt_missing_dep(self):
        with patch.dict('sys.modules', {'pptx': None}):
            ppt_path = os.path.join(self.test_dir, "test.ppt")
            with open(ppt_path, "wb") as f:
                f.write(b"PK")
            result = self.processor.process("{{ OPEN test.ppt }}")
            self.assertIn("Error: python-pptx package is required", result)

    def test_read_excel_missing_dep(self):
        with patch.dict('sys.modules', {'pandas': None}):
            xls_path = os.path.join(self.test_dir, "test.xlsx")
            with open(xls_path, "wb") as f:
                f.write(b"PK")
            result = self.processor.process("{{ OPEN test.xlsx }}")
            self.assertIn("Error: pandas and openpyxl packages are required", result)

    def test_read_image_ocr_missing_dep(self):
        with patch.dict('sys.modules', {'PIL': None}):
            img_path = os.path.join(self.test_dir, "test.png")
            with open(img_path, "wb") as f:
                f.write(b"PNG")
            result = self.processor.process("{{ OPEN test.png }}")
            self.assertIn("Error: pillow and pytesseract packages are required", result)

    def test_read_ppt_direct(self):
        mock_pptx = MagicMock()
        mock_prs = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_shape.text = "Slide Text"
        mock_slide.shapes = [mock_shape]
        mock_prs.slides = [mock_slide]
        mock_pptx.Presentation.return_value = mock_prs
        
        with patch.dict('sys.modules', {'pptx': mock_pptx}):
            result = _read_ppt("test.pptx")
            self.assertIn("Slide Text", result)

    def test_read_excel_direct(self):
        mock_pd = MagicMock()
        mock_xls = MagicMock()
        mock_xls.sheet_names = ["Sheet1"]
        mock_pd.ExcelFile.return_value = mock_xls
        
        mock_df = MagicMock()
        mock_df.to_csv.return_value = "col1,col2\nval1,val2"
        mock_pd.read_excel.return_value = mock_df
        
        with patch.dict('sys.modules', {'pandas': mock_pd}):
            result = _read_excel("test.xlsx")
            self.assertIn("Sheet: Sheet1", result)
            self.assertIn("val1,val2", result)

    def test_read_image_ocr_direct(self):
        mock_pil = MagicMock()
        mock_img = MagicMock()
        mock_pil.Image.open.return_value = mock_img
        
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = "OCR Text"
        
        with patch.dict('sys.modules', {'PIL': mock_pil, 'pytesseract': mock_pytesseract}):
            result = _read_image_ocr("test.png")
            self.assertEqual(result, "OCR Text")

    # ========== EDGE CASE TESTS ==========

    def test_empty_file_handling(self):
        # Empty text file
        empty_file = os.path.join(self.test_dir, "empty.txt")
        open(empty_file, "w").close()
        result = self.processor.process("{{ OPEN empty.txt }}")
        self.assertEqual(result, "")

    def test_special_characters_in_macros(self):
        # Test with special characters
        result = self.processor.process("{{ BASE64 'special: !@#$%^&*()' }}")
        decoded = base64.b64decode(result).decode('utf-8')
        self.assertEqual(decoded, "special: !@#$%^&*()")

    def test_multiple_macros_same_line(self):
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"
        result = self.processor.process("{{ ENV VAR1 }} and {{ ENV VAR2 }}")
        self.assertEqual(result, "value1 and value2")
        del os.environ["VAR1"]
        del os.environ["VAR2"]

    def test_macro_with_unicode(self):
        # Create file with unicode
        unicode_file = os.path.join(self.test_dir, "unicode.txt")
        with open(unicode_file, "w", encoding="utf-8") as f:
            f.write("Hello 世界 🌍")
        
        result = self.processor.process("{{ OPEN unicode.txt }}")
        self.assertIn("世界", result)
        self.assertIn("🌍", result)

    def test_custom_macro_functions(self):
        # Test with custom macro
        def custom_macro(*args, **kwargs):
            return "CUSTOM_" + "_".join(args)
        
        processor = MacroProcessor(
            project_root=self.test_dir,
            macro_functions={"CUSTOM": custom_macro}
        )
        result = processor.process("{{ CUSTOM hello world }}")
        self.assertEqual(result, "CUSTOM_hello_world")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
