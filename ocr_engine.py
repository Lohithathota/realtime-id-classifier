import re
import pytesseract
from PIL import Image
import io
from collections import Counter

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
    Extracts name using confidence-filtered OCR data.
    For Aadhaar: Only considers text from RIGHT side (photo is on left).
    """
    n = len(data["text"])
    
    # Calculate image width for positional filtering
    image_width = max(data["left"][i] + data["width"][i] for i in range(n) if data["text"][i].strip())
    right_threshold = image_width * 0.4  # Right 60% of image
    
    clean_words_with_meta = []
    
    # Filter by confidence and validity
    bad_words = {
        'GOVERNMENT', 'INDIA', 'UIDAI', 'AADHAAR', 'PAN', 'CARD', 'DOB', 'YEAR',
        'FATHER', 'ADDRESS', 'STATE', 'DISTRICT', 'PINCODE', 'PO', 'MALE', 
        'FEMALE', 'INCOME', 'TAX', 'DEPARTMENT', 'PERMANENT', 'ACCOUNT',
        'NUMBER', 'NAME', 'DATE', 'BIRTH', 'ISSUE', 'VALID', 'DIGILOCKER',
        'AUTHORITY', 'UNIQUE', 'IDENTIFICATION'  # Common left-side text
    }

    for i in range(n):
        word = data["text"][i].strip().upper()
        conf = int(data["conf"][i])
        left_pos = data["left"][i]
        
        # For Aadhaar, only consider RIGHT side text
        if document_type == "Aadhaar Card" and left_pos < right_threshold:
            continue
        
        # Strict filter: Must be alpha, >2 chars, high confidence > 60
        if (conf > 60 and 
            len(word) > 2 and 
            word.isalpha() and 
            word not in bad_words):
            
            clean_words_with_meta.append(word)

    if len(clean_words_with_meta) < 2:
        return None
    
    # Score words
    word_scores = {}
    for word in clean_words_with_meta:
        score = 0
        if len(word) >= 4: score += 3
        if word[0] in 'ABCDJKMPSRT': score += 2
        word_scores[word] = score
        
    # Sort by score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_words = [w[0] for w in sorted_words[:4]]
    
    # Try 2-3 word combinations
    for length in range(2, min(4, len(candidate_words)+1)):
        name = ' '.join(candidate_words[:length])
        # Reasonable length check
        if 10 <= len(name) <= 25:
            return name
            
    return None

# ===============================
# EXTRACTION FUNCTIONS
# ===============================
def extract_gender_simple(text):
    text = text.upper()
    if "MALE" in text: return "MALE"
    if "FEMALE" in text: return "FEMALE"
    if "TRANSGENDER" in text: return "TRANSGENDER"
    if "M " in text or " M" in text: return "MALE"
    if "F " in text or " F" in text: return "FEMALE"
    return None

def run_ocr(image_bytes):
    """
    Run OCR with intelligent image preprocessing for optimal text extraction.
    Handles large images by resizing while maintaining aspect ratio.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Get original dimensions
    width, height = image.size
    
    # Optimal OCR size: 1200-1600px on longest side
    max_dimension = max(width, height)
    
    if max_dimension > 2000:
        # Large image - resize to 1600px max
        scale = 1600 / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    elif max_dimension < 800:
        # Small image - upscale to 1200px for better OCR
        scale = 1200 / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Run OCR on optimized image
    text = pytesseract.image_to_string(image, config="--psm 6").upper()
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
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
    gender = extract_gender_simple(text)
    dob = extract_dob(text)
    
    extracted = {
        "name": name,
        "gender": gender,
        "dob_or_yob": dob
    }
    
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
    if aadhaar:
        confidence_score += 0.40
    elif pan:
        confidence_score += 0.40
    elif is_aadhaar_text or is_pan_text:
        confidence_score += 0.20  # Partial identification
    
    # Name extraction quality (30%)
    if name:
        name_length = len(name.replace(' ', ''))
        if 10 <= name_length <= 25:
            confidence_score += 0.30
        elif 6 <= name_length < 10 or 25 < name_length <= 30:
            confidence_score += 0.15  # Partial credit for borderline names
    
    # Gender extraction (10%)
    if gender:
        confidence_score += 0.10
    
    # DOB extraction (20%)
    if dob:
        confidence_score += 0.20
    
    # Cap at 100%
    confidence_score = min(confidence_score, 1.0)
    extracted["confidence"] = f"{int(confidence_score * 100)}%"

    return {
        "document_type": document_type,
        "validation_status": "VALID" if (aadhaar or pan) else "PARTIAL",
        "extracted_fields": extracted
    }
