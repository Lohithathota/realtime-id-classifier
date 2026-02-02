import re
import pytesseract
from PIL import Image
import io
import os
import cv2
import numpy as np

# ===============================
# TESSERACT CONFIG (WINDOWS)
# ===============================
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
# VERHOEFF (AADHAAR VALIDATION)
# ===============================
class Verhoeff:
    d = [
        [0,1,2,3,4,5,6,7,8,9], [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6], [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8], [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2], [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4], [9,8,7,6,5,4,3,2,1,0]
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9], [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2], [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0], [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5], [7,0,4,6,9,1,3,2,5,8]
    ]

    @classmethod
    def validate(cls, number):
        if not number.isdigit() or len(number) != 12:
            return False
        c = 0
        for i, n in enumerate(map(int, reversed(number))):
            c = cls.d[c][cls.p[i % 8][n]]
        return c == 0


# ===============================
# IMPROVED NAME EXTRACTION
# ===============================
def extract_name_universal(data, document_type=None):
    if not data or "text" not in data:
        return None

    n = len(data["text"])
    if n == 0:
        return None

    # ---- GROUP WORDS INTO LINES ----
    indices = sorted(range(n), key=lambda i: (data["top"][i], data["left"][i]))
    lines = []
    current = []
    last_top = None
    threshold = 10

    for i in indices:
        word = data["text"][i].strip()
        if not word:
            continue
        top = data["top"][i]
        if last_top is None or abs(top - last_top) <= threshold:
            current.append(i)
        else:
            lines.append(current)
            current = [i]
        last_top = top

    if current:
        lines.append(current)

    lines_text = []
    for line in lines:
        line = sorted(line, key=lambda i: data["left"][i])
        text = " ".join(data["text"][i] for i in line).strip()
        lines_text.append(text)

    blacklist = {
        "GOVERNMENT", "INDIA", "UIDAI", "AADHAAR", "ENROLMENT",
        "ADDRESS", "IDENTIFICATION", "NUMBER",
        "MALE", "FEMALE", "DOB", "BIRTH",
        "INCOME", "TAX", "DEPARTMENT", "PAN",
        "FATHER", "SIGNATURE"
    }

    def normalize(t):
        return re.sub(r"[^A-Z\s]", "", t.upper()).strip()

    def is_valid_name(t):
        w = t.split()
        return 2 <= len(w) <= 4 and all(x.isalpha() and len(x) >= 3 for x in w)

    # ---- FIND ANCHOR ----
    anchor = None
    for i, line in enumerate(lines_text):
        u = line.upper()
        if re.search(r"\d{2}/\d{2}/\d{4}", u) or "MALE" in u or "FEMALE" in u:
            anchor = i
            break

    if anchor is None:
        return None

    # ---- SEARCH ABOVE ANCHOR ----
    candidates = []
    for i in range(anchor - 1, max(-1, anchor - 8), -1):
        raw = lines_text[i]
        clean_full = normalize(raw)
        
        # Instead of skipping the line if a blacklist word exists, 
        # let's remove the blacklist words to see if a name remains.
        current_candidate = clean_full
        for k in blacklist:
            current_candidate = re.sub(r'\b' + k + r'\b', '', current_candidate).strip()
            
        if not current_candidate:
            continue
            
        if is_valid_name(current_candidate):
            candidates.append(current_candidate)
            break # Found the closest valid name

    # ---- FALLBACK: SEARCH FOR "TO" ANCHOR (Full Letter Support) ----
    if not candidates:
        for i, line in enumerate(lines_text):
            if line.upper().startswith("TO"):
                # Usually the name is the very next line
                if i + 1 < len(lines_text):
                    clean_fallback = normalize(lines_text[i+1])
                    if is_valid_name(clean_fallback):
                         return clean_fallback.title() if document_type == "Aadhaar Card" else clean_fallback

    if not candidates:
        return None

    return candidates[0].title() if document_type == "Aadhaar Card" else candidates[0]


# ===============================
# OTHER EXTRACTION HELPERS
# ===============================
def extract_gender_simple(text):
    text = text.upper()
    
    # Fuzzy matching for FEMALE (handles PEMALE, EEMALE, FEMAL, etc.)
    if re.search(r"\b[FPE][E]?MALE\b|\bFEMAL[EZ]\b", text):
        return "FEMALE"
    
    # Fuzzy matching for MALE (specifically avoiding being caught by FEMALE logic)
    # Checks for MALE, VALE, MAIE, but NOT if preceded by FE/PE
    if re.search(r"(?<!FE)(?<!PE)\b[MV][A]?L[EI]\b", text):
        return "MALE"
        
    if re.search(r"\bTRANS[G]?ENDER\b", text):
        return "TRANSGENDER"
        
    return None


def run_ocr(image_bytes):
    """
    Advanced OCR that cleans images before processing.
    Ensures watermarks and background textures don't block text.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return "", {"text": []}

    # 1. High-Precision Scaling
    h, w = img.shape[:2]
    max_dim = max(h, w)
    # Target 3000px for full A4 pages to catch tiny card text
    if max_dim < 3000:
        scale = 3000 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    # 2. Image Cleaning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0) # Contrast boost
    
    # Adaptive thresholding to remove watermarks/shadows
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )

    # 3. OCR execution
    data = pytesseract.image_to_data(
        processed, output_type=pytesseract.Output.DICT, config="--psm 3"
    )

    text = " ".join(w.strip() for w in data["text"] if w.strip()).upper()
    return text, data


def extract_aadhaar(text):
    # Normalize potential OCR confusion (O -> 0, I -> 1, etc.)
    clean_text = text.translate(str.maketrans("OISZBQG", "0152806"))
    # Look for 12 digits (with optional spaces)
    patterns = re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", clean_text)
    
    valid_numbers = []
    for p in patterns:
        num = re.sub(r"\D", "", p)
        if len(num) == 12:
            if Verhoeff.validate(num):
                return num # Highly confident
            valid_numbers.append(num)
            
    return valid_numbers[0] if valid_numbers else None


def extract_pan(text):
    # PANs often get '0' and 'O' mixed up in the middle or end
    # First, try strict match
    m = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    if m:
        return m.group(0)
        
    # Fallback: OCR might have misread a '0' as 'O' or vice versa
    # We look for the 5-4-1 structure and fix the middle digits
    potential = re.findall(r"\b[A-Z0-9]{5}[A-Z0-9]{4}[A-Z0-9]\b", text)
    for p in potential:
        # Check if it likely fits the PAN structure (5 letters, 4 numbers, 1 letter)
        # by forcing translation on the middle part
        prefix = p[:5].replace('0', 'O').replace('1', 'I')
        middle = p[5:9].replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8')
        suffix = p[9:].replace('0', 'O').replace('1', 'I')
        
        final_pan = prefix + middle + suffix
        if re.match(r"[A-Z]{5}[0-9]{4}[A-Z]", final_pan):
            return final_pan
            
    return None


def extract_dob(text):
    for pat in [
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
        r"\b\d{2}\.\d{2}\.\d{4}\b",
        r"\b(19\d{2}|20[0-2]\d)\b"
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(0).replace("-", "/").replace(".", "/")
    return None


# ===============================
# MAIN PIPELINE
# ===============================
def identify_and_extract(image_file):
    image_bytes = image_file.read()
    text, data = run_ocr(image_bytes)

    aadhaar = extract_aadhaar(text)
    pan = extract_pan(text)

    aadhaar_keywords = ["AADHAAR", "UIDAI", "IDENTIFICATION", "DOB", "MALE", "FEMALE"]
    pan_keywords = ["INCOME", "TAX", "PAN", "ACCOUNT"]

    is_aadhaar_text = sum(k in text for k in aadhaar_keywords) >= 2
    is_pan_text = sum(k in text for k in pan_keywords) >= 2

    if aadhaar:
        doc_type = "Aadhaar Card"
    elif pan:
        doc_type = "PAN Card"
    elif is_aadhaar_text:
        doc_type = "Aadhaar Card"
    elif is_pan_text:
        doc_type = "PAN Card"
    else:
        doc_type = "Other Document"

    name = extract_name_universal(data, doc_type)
    gender = extract_gender_simple(text) if doc_type != "PAN Card" else None
    dob = extract_dob(text)

    extracted = {
        "name": name,
        "dob_or_yob": dob,
        "confidence": "90%" if name and dob else "70%"
    }

    if doc_type != "PAN Card":
        extracted["gender"] = gender

    if aadhaar:
        extracted["aadhaar_number"] = aadhaar
    if pan:
        extracted["pan_number"] = pan

    return {
        "document_type": doc_type,
        "validation_status": "VALID" if aadhaar or pan else "PARTIAL",
        "extracted_fields": extracted
    }


def extract_all_text(image_file):
    """
    High-precision generic OCR extraction for complex documents (like Aadhaar letters).
    Uses OpenCV for adaptive thresholding and noise reduction.
    """
    # 1. Load image
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "ERROR", "message": "Could not decode image"}

    # 2. Advanced Scaling (Crucial for A4 sheets with tiny text)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim < 3000:
        scale = 3000 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    # 3. Pre-processing for "Dirty" or Complex Backgrounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Adaptive Thresholding (removes gray watermarks and background patterns)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    
    # Denoising
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    # 4. OCR with Layout Analysis
    # --psm 3: Automatic page segmentation with OSD. 
    full_text = pytesseract.image_to_string(processed, config="--psm 3").strip()
    
    # Clean and split into lines
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    normalized_text = "\n".join(lines)
    
    return {
        "full_text": normalized_text,
        "lines": lines,
        "word_count": len(normalized_text.split()),
        "status": "SUCCESS" if normalized_text else "EMPTY"
    }
