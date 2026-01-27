
from ocr_engine import extract_aadhaar_details, extract_pan_details
import logging

# Setup basic logging to see what's happening inside
logging.basicConfig(level=logging.INFO)

def test_aadhaar_noise():
    print("--- ADAHAAR NOISE TEST ---")
    texts = [
        """
        GOVERNMENT OF INDIA
        Rahul_Kumar
        DOB: 12/05/1990
        Gender: MALE
        1234 5678 9012
        """,
        """
        GOVERNMENT OF INDIA
        Name: S.K. Singh|
        DOB: 12/05/1990
        MALE
        1234 5678 9012
        """,
        """
        GOVT OF INDIA
        P. Sharma
        Address: 123 Lane
        DOB: 12/05/1990
        1234 5678 9012
        """
    ]
    
    for i, t in enumerate(texts):
        print(f"Case {i+1}:")
        try:
            res = extract_aadhaar_details(t)
            print(f"Result: {res['name']} (Valid? {res.get('name') != 'INVALID'})")
        except Exception as e:
            print(f"Error: {e}")

def test_pan_noise():
    print("\n--- PAN NOISE TEST ---")
    texts = [
        """
        INCOME TAX DEPARTMENT
        GOVT OF INDIA
        V. KOHLI.
        Father's Name
        PREM KOHLI
        05/11/1988
        ABCDE1234P
        """,
        """
        INCOME TAX DEPARTMENT
        JANE DOE_
        12/12/1990
        ABCDE1234P
        """
    ]
    for i, t in enumerate(texts):
        print(f"Case {i+1}:")
        try:
            res = extract_pan_details(t)
            print(f"Result: {res['name']} (Valid? {res.get('name') != 'INVALID'})")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_aadhaar_noise()
    test_pan_noise()
