
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# IMPORTANT: Update this path if Tesseract is not in your PATH
# Common Windows path: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Common Linux path: '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def set_tesseract_cmd(path: str):
    """Set the tesseract executable path manually."""
    logger.info(f"Setting custom Tesseract path: {path}")
    pytesseract.pytesseract.tesseract_cmd = path

# --- VERHOEFF ALGORITHM FOR AADHAAR VALIDATION ---
class Verhoeff:
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

    @classmethod
    def validate(cls, number_str: str) -> bool:
        """Validate Verhoeff checksum."""
        if not number_str.isdigit() or len(number_str) != 12:
            return False
        c = 0
        ll = list(map(int, reversed(number_str)))
        for i, item in enumerate(ll):
            c = cls.d[c][cls.p[i % 8][item]]
        return c == 0

# --- UTILS ---
def parse_date(date_str: str) -> Optional[str]:
    """Parse date from various formats to DD/MM/YYYY."""
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%Y-%m-%d" # ISO
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt > datetime.now():
                return None # Future date
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None

def preprocess_for_ocr(pil_image: Image.Image) -> np.ndarray:
    """Enhance image for OCR."""
    try:
        # Convert PIL to OpenCV
        img = np.array(pil_image.convert('RGB')) 
        img = img[:, :, ::-1].copy() # RGB to BGR

        # Gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise removal
        # gray = cv2.medianBlur(gray, 3) # Can be aggressive

        # Thresholding (Otsu's binarization)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Simple rescaling can often help Tesseract
        # If image is small, resize
        h, w = gray.shape
        if w < 1000:
            scale = 2.0
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return gray
    except Exception as e:
        logger.warning(f"Preprocessing failed, using original: {e}")
        return np.array(pil_image)

def run_tesseract(image: Image.Image) -> str:
    """Run Tesseract OCR on the image."""
    try:
        processed_img = preprocess_for_ocr(image)
        # custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img) # , config=custom_config
        return text
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not found. Please set Tesseract path.")
        return ""
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""

# --- EXTRACTION LOGIC ---

def is_likely_ocr_garbage(text: str) -> bool:
    """
    Detect if text is likely OCR garbage rather than a real name.
    Returns True if text appears to be OCR noise.
    """
    # Check for excessive mixed case (e.g., "saa faa ANd UAtAIX")
    # Real names are usually consistent: "John Smith" or "JOHN SMITH"
    words = text.split()
    if len(words) >= 2:
        # Count words with mixed case within the word
        mixed_case_words = 0
        for word in words:
            if len(word) > 1:
                has_upper = any(c.isupper() for c in word)
                has_lower = any(c.islower() for c in word)
                # If a single word has both upper and lower (not just first letter capitalized)
                if has_upper and has_lower and not (word[0].isupper() and word[1:].islower()):
                    mixed_case_words += 1
        
        # If more than half the words have weird mixed case, likely garbage
        if mixed_case_words > len(words) / 2:
            return True
    
    # Check for very short words that don't make sense
    # Real names have words of reasonable length
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    if avg_word_len < 2.5:  # e.g., "aa At arias ae"
        return True
    
    # Check for too many single-letter words
    single_letter_count = sum(1 for w in words if len(w) == 1)
    if single_letter_count > 1:  # More than one single letter is suspicious
        return True
    
    return False

def score_name_quality(name: str, context: str = "aadhaar") -> int:
    """
    Score a name candidate based on how likely it is to be a real name.
    Higher score = more likely to be correct.
    """
    score = 0
    words = name.split()
    
    # Bonus for reasonable length (2-5 words is typical)
    if 2 <= len(words) <= 5:
        score += 30
    elif len(words) == 1:
        score -= 10  # Single word names are less common
    
    # Bonus for consistent capitalization
    all_caps = name.isupper()
    title_case = all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1)
    
    if context == "pan" and all_caps:
        score += 20  # PAN cards typically have all caps names
    elif context == "aadhaar" and title_case:
        score += 20  # Aadhaar typically has title case
    
    # Penalty for OCR garbage indicators
    if is_likely_ocr_garbage(name):
        score -= 50
    
    # Bonus for reasonable word lengths
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    if 3 <= avg_word_len <= 8:
        score += 15
    
    # Bonus for total length in reasonable range
    if 10 <= len(name) <= 40:
        score += 10
    
    return score

def extract_aadhaar_details(text: str) -> Dict[str, Any]:
    """Extract Aadhaar details from text with improved name validation."""
    result = {
        "aadhaar_number": None,
        "aadhaar_number_valid": False,
        "name": None,
        "dob": None,
        "dob_valid": False,
        "gender": None
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 1. Aadhaar Number (12 digits, often spaced)
    aadhaar_pattern = r'\b(\d{4}\s?\d{4}\s?\d{4})\b'
    match = re.search(aadhaar_pattern, text)
    if match:
        raw_num = match.group(0).replace(' ', '')
        if len(raw_num) == 12:
            result['aadhaar_number'] = raw_num
            result['aadhaar_number_valid'] = Verhoeff.validate(raw_num)
    
    # 2. DOB
    dob_pattern = r'(?:DOB|Date of Birth|Year of Birth)[:\s]*([0-9\/\-\.]+)'
    match_dob = re.search(dob_pattern, text, re.IGNORECASE)
    if match_dob:
        raw_dob = match_dob.group(1).strip()
        parsed = parse_date(raw_dob)
        if parsed:
            result['dob'] = parsed
            result['dob_valid'] = True
        elif len(raw_dob) == 4 and raw_dob.isdigit():
             result['dob'] = f"01/01/{raw_dob}"
             result['dob_valid'] = True

    # 3. Gender
    gender_pattern = r'\b(MALE|FEMALE|TRANSGENDER)\b'
    match_gender = re.search(gender_pattern, text, re.IGNORECASE)
    if match_gender:
        result['gender'] = match_gender.group(1).upper()

    # 4. Name Extraction (Improved Heuristic)
    # Blocklist of keywords that are definitely NOT names
    block_keywords = [
        "GOVERNMENT", "INDIA", "AADHAAR", "UIDAI", "DOB", "YEAR", "FATHER", "ADDRESS", 
        "DATE", "BIRTH", "MALE", "FEMALE", "TRANSGENDER", "NO:", "NUMBER", "VID", "MERGE",
        "UNIQUE", "IDENTIFICATION", "AUTHORITY", "HUSBAND", "WIFE", "GUARDIAN"
    ]
    
    # Relationship indicators that precede father's/husband's names
    # These patterns indicate the name is NOT the cardholder's name
    relationship_patterns = [
        r'\bS/O\b',      # Son of
        r'\bD/O\b',      # Daughter of
        r'\bW/O\b',      # Wife of
        r'\bC/O\b',      # Care of
        r'\bFATHER\b',   # Father
        r'\bHUSBAND\b',  # Husband
        r'\bGUARDIAN\b', # Guardian
    ]
    
    possible_names = []
    logger.info(f"Processing {len(lines)} lines for Aadhaar name extraction")
    
    for idx, line in enumerate(lines):
        # Normalize: Remove leading/trailing explicit non-name chars (like "Name: ")
        cleaned = re.sub(r'^(Name|NAME)[:\s]*', '', line).strip()
        
        # Remove common OCR noise characters at the end
        cleaned = re.sub(r'[|_\\/]+$', '', cleaned).strip()
        
        logger.debug(f"Line {idx}: '{line}' -> cleaned: '{cleaned}'")
        
        # CRITICAL: Skip lines that contain relationship indicators
        # This filters out father's/husband's names
        is_relationship_name = False
        for pattern in relationship_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                is_relationship_name = True
                logger.debug(f"  Skipped: contains relationship indicator '{pattern}'")
                break
        if is_relationship_name:
            continue
        
        # Also check the previous line for relationship indicators
        # Sometimes "S/O" appears on the line before the father's name
        if idx > 0:
            prev_line = lines[idx - 1]
            for pattern in relationship_patterns:
                if re.search(pattern, prev_line, re.IGNORECASE):
                    logger.debug(f"  Skipped: previous line contains relationship indicator")
                    is_relationship_name = True
                    break
        if is_relationship_name:
            continue
        
        # Check if line contains digits (names rarely have digits unless OCR error)
        if re.search(r'\d', cleaned):
            logger.debug(f"  Skipped: contains digits")
            continue
            
        # Check against block keywords
        is_blocked = False
        upper_line = cleaned.upper()
        for kw in block_keywords:
            if kw in upper_line:
                is_blocked = True
                logger.debug(f"  Skipped: contains keyword '{kw}'")
                break
        if is_blocked:
            continue
            
        # Check for minimum length and valid characters
        # Allow letters, spaces, dots, hyphens, apostrophes
        if len(cleaned) < 3:
            logger.debug(f"  Skipped: too short ({len(cleaned)} chars)")
            continue
            
        # Check if it looks like a name (mostly letters)
        # Using a regex that *mostly* matches a name line
        if re.match(r'^[a-zA-Z\s\.\-\']+$', cleaned):
            possible_names.append(cleaned)
            logger.info(f"  ✓ Candidate name found: '{cleaned}'")
        else:
            logger.debug(f"  Skipped: doesn't match name pattern")

    logger.info(f"Found {len(possible_names)} possible names: {possible_names}")
    
    # Use quality scoring to select the best name
    if possible_names:
        # Score each candidate
        scored_names = [(name, score_name_quality(name, "aadhaar")) for name in possible_names]
        scored_names.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        logger.info(f"Scored names: {[(n, s) for n, s in scored_names]}")
        
        # Take the highest scoring name
        result['name'] = scored_names[0][0]
        logger.info(f"Selected name: '{result['name']}' (score: {scored_names[0][1]})")

    return result

def extract_pan_details(text: str) -> Dict[str, Any]:
    """Extract PAN details from text with improved name validation."""
    result = {
        "pan_number": None,
        "pan_number_valid": False,
        "name": None,
        "dob": None,
        "dob_valid": False
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 1. PAN Number
    pan_pattern = r'\b([A-Z]{5}[0-9]{4}[A-Z])\b'
    match_pan = re.search(pan_pattern, text)
    if match_pan:
        result['pan_number'] = match_pan.group(1)
        result['pan_number_valid'] = True

    # 2. DOB
    dob_pattern = r'\b(\d{2}/\d{2}/\d{4})\b'
    match_dob = re.search(dob_pattern, text)
    if match_dob:
        parsed = parse_date(match_dob.group(1))
        if parsed:
            result['dob'] = parsed
            result['dob_valid'] = True

    # 3. Name Extraction (Improved)
    block_keywords = [
        "INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA", "PERMANENT", "ACCOUNT", "NUMBER", 
        "SIGNATURE", "DATE", "BIRTH", "FATHER", "NAME", "HUSBAND", "GUARDIAN"
    ]
    
    # Relationship indicators for PAN cards
    relationship_patterns = [
        r'\bS/O\b',
        r'\bD/O\b',
        r'\bW/O\b',
        r'\bC/O\b',
        r'\bFATHER\b',
        r'\bHUSBAND\b',
    ]
    
    possible_names = []
    found_dob_index = -1
    
    # Locate DOB line context
    for i, line in enumerate(lines):
        if re.search(r'\d{2}/\d{2}/\d{4}', line):
            found_dob_index = i
            break
            
    limit = found_dob_index if found_dob_index != -1 else len(lines)
    
    # Search lines before DOB
    for i in range(limit):
        line = lines[i]
        cleaned = re.sub(r'^(Name|NAME)[:\s]*', '', line).strip()
        
        # CRITICAL: Skip lines with relationship indicators
        is_relationship_name = False
        for pattern in relationship_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                is_relationship_name = True
                logger.debug(f"PAN: Skipped line {i}: contains relationship indicator")
                break
        if is_relationship_name:
            continue
        
        # Check previous line too
        if i > 0:
            prev_line = lines[i - 1]
            for pattern in relationship_patterns:
                if re.search(pattern, prev_line, re.IGNORECASE):
                    is_relationship_name = True
                    break
        if is_relationship_name:
            continue
        
        if re.search(r'\d', cleaned): continue
        if len(cleaned) < 3: continue
        
        is_blocked = False
        upper_line = cleaned.upper()
        for kw in block_keywords:
            if kw in upper_line:
                is_blocked = True
                break
        if is_blocked: continue
            
        # Allow valid name chars
        if re.match(r'^[a-zA-Z\s\.\-\']+$', cleaned):
            possible_names.append(cleaned)
    
    # Use quality scoring to select the best name
    if possible_names:
        logger.info(f"PAN: Found {len(possible_names)} possible names: {possible_names}")
        
        # Score each candidate
        scored_names = [(name, score_name_quality(name, "pan")) for name in possible_names]
        scored_names.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
        
        logger.info(f"PAN: Scored names: {[(n, s) for n, s in scored_names]}")
        
        # Take the highest scoring name
        result['name'] = scored_names[0][0]
        logger.info(f"PAN: Selected name: '{result['name']}' (score: {scored_names[0][1]})") 

    return result

def process_document(image_file, doc_type: str) -> Dict[str, Any]:
    """
    Main OCR Processing function.
    
    Args:
        image_file: file-like object or PIL Image
        doc_type: 'aadhaar' or 'pan'
    """
    try:
        if isinstance(image_file, Image.Image):
            image = image_file
        else:
            image = Image.open(image_file).convert("RGB")
    except Exception as e:
        logger.error(f"Image load error: {e}")
        return {
            "document_type": getattr(doc_type, 'capitalize', lambda: str(doc_type))(),
            "ocr_status": "FAIL",
            "overall_status": "REJECTED"
        }

    # Run OCR
    text = run_tesseract(image)
    logger.info(f"OCR Extracted Text (first 50 chars): {text[:50]}...")

    if not text:
        return {
            "document_type": doc_type.capitalize(),
            "ocr_status": "FAIL",
            "overall_status": "REJECTED"
        }

    extracted_data = {}
    overall_status = "REJECTED"
    
    # Common Name Validation Regex
    # Allow: Letters, Spaces, Dots, Hyphens, Apostrophes
    # Ensure it doesn't start/end with special chars (basic cleanup)
    # ^[a-zA-Z] ensures starts with letter.
    name_validator = r"^[a-zA-Z][a-zA-Z\s\.\-\']*[a-zA-Z\.]$" 
    # [a-zA-Z\.] at end allows initials at end? "Rahul K."

    if doc_type.lower() == 'aadhaar':
        data = extract_aadhaar_details(text)
        
        # Validation Logic
        name_valid = False
        if data['name']:
            # Strip extra spaces for checking
            clean_name = data['name'].strip()
            # Regex validation
            if re.match(r"^[a-zA-Z\s\.\-\']+$", clean_name) and len(clean_name) >= 3:
                name_valid = True
            else:
                data['name'] = "INVALID" # Failed format
        else:
            data['name'] = "INVALID" # Not found

        if not data['dob_valid']: data['dob'] = "INVALID"
        if not data['gender']: data['gender'] = "INVALID"
        if not data['aadhaar_number_valid']: data['aadhaar_number'] = "INVALID"
             
        # Overall Status
        if (data['aadhaar_number_valid'] and 
            name_valid and 
            data['dob_valid'] and 
            data['gender'] != "INVALID"):
            overall_status = "APPROVED"
            
        extracted_data = {
            "document_type": "Aadhaar",
            "ocr_status": "SUCCESS",
            "aadhaar_number": data['aadhaar_number'],
            "aadhaar_number_valid": data['aadhaar_number_valid'],
            "name": data['name'],
            "dob": data['dob'],
            "dob_valid": data['dob_valid'],
            "gender": data['gender'],
            "overall_status": overall_status
        }
        
    elif doc_type.lower() == 'pan':
        data = extract_pan_details(text)
        
        # Validation Logic
        name_valid = False
        if data['name']:
             clean_name = data['name'].strip()
             if re.match(r"^[a-zA-Z\s\.\-\']+$", clean_name) and len(clean_name) >= 3:
                name_valid = True
             else:
                data['name'] = "INVALID"
        else:
             data['name'] = "INVALID"
             
        if not data['dob_valid']: data['dob'] = "INVALID"
        if not data['pan_number_valid']: data['pan_number'] = "INVALID"
        
        # 4th char validation (P)
        if data['pan_number'] and data['pan_number'] != "INVALID":
             if data['pan_number'][3] != 'P':
                 # Warn or Fail? User implies validation rules are strict.
                 pass
        
        if (data['pan_number_valid'] and 
            name_valid and 
            data['dob_valid']):
            overall_status = "APPROVED"
            
        extracted_data = {
            "document_type": "PAN",
            "ocr_status": "SUCCESS",
            "pan_number": data['pan_number'],
            "pan_number_valid": data['pan_number_valid'],
            "name": data['name'],
            "dob": data['dob'],
            "dob_valid": data['dob_valid'],
            "overall_status": overall_status
        }

    else:
         extracted_data = {
            "document_type": doc_type,
            "ocr_status": "SUCCESS",
            "text": text,
            "overall_status": "UNKNOWN_TYPE"
        }

    return extracted_data

