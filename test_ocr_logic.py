
import unittest
from ocr_engine import extract_aadhaar_details, extract_pan_details, Verhoeff, process_document

class TestOCREngine(unittest.TestCase):
    
    def test_aadhaar_extraction_complex_name(self):
        text = """
        GOVERNMENT OF INDIA
        K.L. Rahul
        DOB: 15/08/1995
        Male
        1234 5678 9012
        """
        data = extract_aadhaar_details(text)
        print(f"Aadhaar Data: {data}")
        # Logic should now keep dots
        self.assertEqual(data['name'], "K.L. Rahul")
        
    def test_aadhaar_name_validation_logic(self):
        # Test valid names
        valid_names = ["Rahul Kumar", "A. Singh", "Jean-Luc Picard", "O'Neil"]
        import re
        for name in valid_names:
             self.assertTrue(re.match(r"^[a-zA-Z\s\.\-\']+$", name) and len(name) >= 3, f"Failed valid: {name}")

        # Test invalid names
        invalid_names = ["Rahul123", "User!", "@Admin", "AB", ""]
        for name in invalid_names:
             self.assertFalse(re.match(r"^[a-zA-Z\s\.\-\']+$", name) and len(name) >= 3, f"Failed invalid: {name}")

    def test_pan_extraction_with_dots(self):
        text = """
        INCOME TAX DEPARTMENT
        GOVT OF INDIA
        V. KOHLI
        05/11/1988
        ABCDE1234P
        """
        data = extract_pan_details(text)
        print(f"PAN Data: {data}")
        self.assertEqual(data['name'], "V. KOHLI")
        self.assertEqual(data['pan_number'], "ABCDE1234P")

if __name__ == '__main__':
    unittest.main()
