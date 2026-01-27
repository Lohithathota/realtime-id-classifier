#!/usr/bin/env python3
"""Test relationship keyword filtering"""

from ocr_engine import extract_aadhaar_details, extract_pan_details

print("=== Test 1: Aadhaar with S/O (Son of) ===")
aadhaar_text_1 = """
GOVERNMENT OF INDIA
Rajesh Kumar
S/O Ramesh Kumar
DOB: 15/08/1990
Male
1234 5678 9012
"""
result = extract_aadhaar_details(aadhaar_text_1)
print(f"Extracted name: '{result['name']}'")
print(f"Expected: 'Rajesh Kumar' (NOT 'Ramesh Kumar')\n")

print("=== Test 2: Aadhaar with W/O (Wife of) ===")
aadhaar_text_2 = """
GOVERNMENT OF INDIA
Priya Sharma
W/O Vijay Sharma
DOB: 20/03/1992
Female
9876 5432 1098
"""
result = extract_aadhaar_details(aadhaar_text_2)
print(f"Extracted name: '{result['name']}'")
print(f"Expected: 'Priya Sharma' (NOT 'Vijay Sharma')\n")

print("=== Test 3: PAN with Father's Name ===")
pan_text = """
INCOME TAX DEPARTMENT
GOVT OF INDIA
AMIT PATEL
Father's Name
SURESH PATEL
12/05/1985
ABCDE1234P
"""
result = extract_pan_details(pan_text)
print(f"Extracted name: '{result['name']}'")
print(f"Expected: 'AMIT PATEL' (NOT 'SURESH PATEL')\n")

print("=== Test 4: PAN with S/O ===")
pan_text_2 = """
INCOME TAX DEPARTMENT
RAVI KUMAR
S/O MOHAN KUMAR
15/08/1988
ABCDE1234P
"""
result = extract_pan_details(pan_text_2)
print(f"Extracted name: '{result['name']}'")
print(f"Expected: 'RAVI KUMAR' (NOT 'MOHAN KUMAR')")
