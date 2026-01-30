import re
import pytesseract
from PIL import Image
import io
import os

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
    for i in range(anchor - 1, max(-1, anchor - 6), -1):
        raw = lines_text[i]
        clean = normalize(raw)
        if not clean:
            continue
        if any(k in clean for k in blacklist):
            continue
        if not is_valid_name(clean):
            continue
        candidates.append(clean)

    if not candidates:
        return None

    return candidates[0].title() if document_type == "Aadhaar Card" else candidates[0]


# ===============================
# OTHER EXTRACTION HELPERS
# ===============================
def extract_gender_simple(text):
    text = text.upper()
    if re.search(r"\bFEMALE\b", text):
        return "FEMALE"
    if re.search(r"\bMALE\b", text):
        return "MALE"
    if re.search(r"\bTRANSGENDER\b", text):
        return "TRANSGENDER"
    return None


def run_ocr(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    m = max(w, h)

    if m > 2000:
        s = 1600 / m
        image = image.resize((int(w * s), int(h * s)), Image.BILINEAR)
    elif m < 800:
        s = 1200 / m
        image = image.resize((int(w * s), int(h * s)), Image.BILINEAR)

    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config="--psm 6"
    )

    text = " ".join(w.strip() for w in data["text"] if w.strip()).upper()
    return text, data


def extract_aadhaar(text):
    text = text.translate(str.maketrans("OISZB", "01528"))
    patterns = re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
    for p in patterns:
        num = re.sub(r"\D", "", p)
        if len(num) == 12 and Verhoeff.validate(num):
            return num
    for p in patterns:
        num = re.sub(r"\D", "", p)
        if len(num) == 12:
            return num
    return None


def extract_pan(text):
    m = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    return m.group(0) if m else None


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
