import re
import pytesseract
from PIL import Image
import io
from collections import Counter
import os

# Configure Tesseract path for Windows
if os.name == 'nt':
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(os.environ.get('USERPROFILE', ''), r'AppData\Local\Tesseract-OCR\tesseract.exe')
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# ===============================
# VERHOEFF (AADHAAR)
# ===============================
class Verhoeff:
    d = [
        [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5], [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7], [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3], [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0]
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4], [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7], [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]
    ]

    @classmethod
    def validate(cls, number):
        if not number.isdigit() or len(number) != 12: return False
        c = 0
        for i, n in enumerate(map(int, reversed(number))):
            c = cls.d[c][cls.p[i % 8][n]]
        return c == 0

# ===============================
# UNIVERSAL NAME EXTRACTION
# ===============================
def extract_name_universal(data, document_type=None):
    """
    Extracts name using layout-aware line grouping and proximity to DOB/Gender.
    - Aadhaar: Title Case lines above DOB/Gender.
    - PAN: ALL CAPS lines above Father's Name/DOB.
    """
    n = len(data["text"])
    if n == 0: return None

    # 1. Group words into lines
    lines_meta = []
    word_indices = sorted(range(n), key=lambda i: (data["top"][i], data["left"][i]))
    
    current_line = []
    last_top = -1
    threshold = 12 
    
    for i in word_indices:
        word = data["text"][i].strip()
        if not word: continue
        top = data["top"][i]
        if last_top == -1 or abs(top - last_top) <= threshold:
            current_line.append(i)
        else:
            lines_meta.append(current_line)
            current_line = [i]
        last_top = top
    if current_line: lines_meta.append(current_line)

    lines_text = []
    for line_indices in lines_meta:
        line_indices = sorted(line_indices, key=lambda i: data["left"][i])
        text = " ".join([data["text"][i] for i in line_indices])
        lines_text.append({"text": text.strip(), "top": data["top"][line_indices[0]]})

    bad_keywords = {'GOVERNMENT', 'INDIA', 'UIDAI', 'AADHAAR', 'PAN', 'CARD', 'FATHER', 
                    'INCOME', 'TAX', 'DEPARTMENT', 'ACCOUNT', 'NUMBER', 'IDENTIFICATION',
                    'ADDRESS', 'ENROLMENT', 'MALE', 'FEMALE', 'DATE', 'GENDER', 'BIRTH'}

    # 2. Find Anchor (DOB or Gender)
    anchor_idx = -1
    for idx, line in enumerate(lines_text):
        normalized = line["text"].upper()
        if re.search(r"\d{2}/\d{2}/\d{4}", normalized) or "MALE" in normalized or "FEMALE" in normalized:
            anchor_idx = idx
            break

    # 3. Search backwards from anchor
    search_limit = max(0, anchor_idx - 4) if anchor_idx != -1 else 0
    candidate_lines = []
    
    start_idx = anchor_idx - 1 if anchor_idx != -1 else len(lines_text) - 1
    for i in range(start_idx, search_limit - 1, -1):
        line = lines_text[i]["text"]
        normalized = line.upper()
        if any(k in normalized for k in bad_keywords): continue
        if not any(c.isalpha() for c in line): continue
        if len(line) < 3: continue
        candidate_lines.append(line)

    if not candidate_lines: return None

    # 4. Filter based on Doc Type specifics
    if document_type == "PAN Card":
        # Usually ALL CAPS
        for line in candidate_lines:
            if line.isupper() and len(line.split()) >= 1:
                return line
    elif document_type == "Aadhaar Card":
        # Usually Title Case
        for line in candidate_lines:
            words = line.split()
            if all(w[0].isupper() for w in words if w[0].isalpha()):
                return line

    return candidate_lines[0] # Return closest valid line above anchor

# ===============================
# EXTRACTION FUNCTIONS
# ===============================
def extract_gender_simple(text):
    """
    Extracts gender using regex word boundaries to avoid substring issues 
    (e.g., matching 'MALE' inside 'FEMALE').
    """
    text = text.upper()
    
    # Check for FEMALE first or use boundaries to prevent 'MALE' substring match
    if re.search(r"\bFEMALE\b", text): 
        return "FEMALE"
    if re.search(r"\bMALE\b", text): 
        return "MALE"
    if re.search(r"\bTRANSGENDER\b", text): 
        return "TRANSGENDER"
    
    # Fallback for single characters with boundaries
    if re.search(r"\bF\b", text): 
        return "FEMALE"
    if re.search(r"\bM\b", text): 
        return "MALE"
        
    return None

def run_ocr(image_bytes):
    """
    Run OCR once to get both text and position data, significantly improving performance.
    Uses BILINEAR resizing for a better balance between speed and accuracy.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    max_dimension = max(width, height)
    
    # Resize logic (optimized filter)
    if max_dimension > 2000:
        scale = 1600 / max_dimension
        image = image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    elif max_dimension < 800:
        scale = 1200 / max_dimension
        image = image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    
    # SINGLE CALL to Tesseract (this is where the speed gain is)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")
    
    # Reconstruct text from the data dictionary
    # Filter out empty strings and join with spaces
    words = [w.strip() for w in data["text"] if w.strip()]
    text = " ".join(words).upper()
    
    return text, data

def extract_aadhaar(text):
    """
    Extract Aadhaar number - STRICT RULE: Any 12-digit number in XXXX XXXX XXXX format
    """
    # Common OCR misreads - fix before extraction
    text = text.replace('O', '0').replace('o', '0')  # O -> 0
    text = text.replace('I', '1').replace('l', '1')  # I/l -> 1
    text = text.replace('S', '5').replace('s', '5')  # S -> 5
    text = text.replace('Z', '2').replace('z', '2')  # Z -> 2
    text = text.replace('B', '8').replace('b', '8')  # B -> 8
    
    # Find ALL 12-digit patterns (with or without spaces)
    patterns = re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
    
    for pattern in patterns:
        num = re.sub(r"\D", "", pattern)  # Remove spaces
        if len(num) == 12:
            # Prefer Verhoeff-valid numbers
            if Verhoeff.validate(num):
                return num
    
    # If no Verhoeff-valid number, return first 12-digit number found
    for pattern in patterns:
        num = re.sub(r"\D", "", pattern)
        if len(num) == 12:
            return num
    
    return None

def extract_pan(text):
    m = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    return m.group(0) if m else None

def extract_dob(text):
    """Extract DOB or Year of Birth from text"""
    # Try DD/MM/YYYY format
    m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
    if m:
        return m.group(0)
    
    # Try DD-MM-YYYY format
    m = re.search(r"\b\d{2}-\d{2}-\d{4}\b", text)
    if m:
        return m.group(0).replace('-', '/')
    
    # Try DD.MM.YYYY format
    m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text)
    if m:
        return m.group(0).replace('.', '/')
    
    # Try Year of Birth (YYYY) - look for 4-digit year between 1900-2024
    m = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text)
    if m:
        return m.group(0)
    
    return None

# ===============================
# MAIN PIPELINE
# ===============================
def identify_and_extract(image_file):
    image_bytes = image_file.read()
    text, data = run_ocr(image_bytes)

    # First, identify document type
    aadhaar = extract_aadhaar(text)
    pan = extract_pan(text)
    
    # Keyword-based identification fallback (more robust)
    # Check for Aadhaar keywords with variations
    aadhaar_keywords = [
        "AADHAAR", "AADHAR", "UIDAI", "UNIQUE", "IDENTIFICATION",
        "GOVERNMENT", "INDIA", "GOI", "GOVT", "ENROLMENT",
        "DOB", "MALE", "FEMALE"  # Common Aadhaar fields
    ]
    
    # Check for PAN keywords
    pan_keywords = [
        "INCOME", "TAX", "DEPARTMENT", "PERMANENT", "ACCOUNT",
        "PAN", "FATHER", "SIGNATURE"
    ]
    
    # Count keyword matches
    aadhaar_match_count = sum(1 for keyword in aadhaar_keywords if keyword in text)
    pan_match_count = sum(1 for keyword in pan_keywords if keyword in text)
    
    # Aadhaar if 2+ keywords match
    is_aadhaar_text = aadhaar_match_count >= 2
    is_pan_text = pan_match_count >= 2
    
    if aadhaar:
        document_type = "Aadhaar Card"
    elif pan:
        document_type = "PAN Card"
    elif is_aadhaar_text:
        document_type = "Aadhaar Card"
    elif is_pan_text:
        document_type = "PAN Card"
    else:
        document_type = "Other Document"
    
    # Now extract name with document type context
    name = extract_name_universal(data, document_type)
    gender = extract_gender_simple(text) if document_type != "PAN Card" else None
    dob = extract_dob(text)
    
    extracted = {
        "name": name,
        "dob_or_yob": dob
    }
    
    # Only include gender if it's not a PAN card
    if document_type != "PAN Card":
        extracted["gender"] = gender
    
    if aadhaar:
        extracted["aadhaar_number"] = aadhaar
    elif pan:
        extracted["pan_number"] = pan
    elif is_aadhaar_text:
        extracted["aadhaar_number"] = None 
    elif is_pan_text:
        extracted["pan_number"] = None
    
    # Calculate Comprehensive Confidence Score
    confidence_score = 0.0
    
    # Base score for document type identification (40%)
    if aadhaar or pan:
        confidence_score += 0.40
    elif is_aadhaar_text or is_pan_text:
        confidence_score += 0.20
    
    # Name extraction quality (30%)
    if name:
        confidence_score += 0.30
    
    # DOB extraction (20%)
    if dob:
        confidence_score += 0.20
    
    # Gender extraction (10%) - Only for non-PAN documents
    if document_type != "PAN Card":
        if gender:
            confidence_score += 0.10
    else:
        # For PAN, redistribute the 10% to other critical fields (e.g., Number or Name)
        if pan:
            confidence_score += 0.10
    
    # Cap at 100%
    confidence_score = min(confidence_score, 1.0)
    extracted["confidence"] = f"{int(confidence_score * 100)}%"

    return {
        "document_type": document_type,
        "validation_status": "VALID" if (aadhaar or pan) else "PARTIAL",
        "extracted_fields": extracted
    }
