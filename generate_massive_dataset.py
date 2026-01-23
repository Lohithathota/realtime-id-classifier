import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from multiprocessing import Pool, cpu_count
import re

# ================= CONFIG =================
DATASET_DIR = Path("dataset/train")
CLASSES = ["aadhaar", "pan", "payment_receipt", "other"]
IMAGES_PER_CLASS = 1000

# Ensure directories exist
for cls in CLASSES:
    (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)

# ================= GLOBAL UNIQUE STORAGE =================
USED_AADHAAR = set()
USED_PAN = set()

# ================= UTILITIES =================
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def random_date():
    return f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(1960,2005)}"

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

# ================= VALIDATION =================
def generate_unique_aadhaar():
    while True:
        num = "".join(random.choices(string.digits, k=12))
        if num not in USED_AADHAAR:
            USED_AADHAAR.add(num)
            return f"{num[:4]} {num[4:8]} {num[8:]}"

def generate_unique_pan():
    while True:
        pan = (
            "".join(random.choices(string.ascii_uppercase, k=5)) +
            "".join(random.choices(string.digits, k=4)) +
            random.choice(string.ascii_uppercase)
        )
        if pan not in USED_PAN:
            USED_PAN.add(pan)
            return pan

def validate_aadhaar(a):
    clean = a.replace(" ", "")
    return clean.isdigit() and len(clean) == 12

def validate_pan(p):
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", p))

# ================= AADHAAR =================
def create_aadhaar(i):
    img = Image.new("RGB", (400, 250), (245, 245, 245))
    d = ImageDraw.Draw(img)

    # Header
    d.text((110, 10), "Government of India", font=get_font(16), fill="black")

    # Photo
    d.rectangle((20, 70, 100, 160), fill="gray")

    # Details
    d.text((120, 70), "Name: " + random.choice(["RAHUL", "ANITA", "SUNIL", "PRIYA"]), font=get_font(14), fill="black")
    d.text((120, 100), "DOB: " + random_date(), font=get_font(14), fill="black")

    # Aadhaar number
    aadhaar = generate_unique_aadhaar()
    assert validate_aadhaar(aadhaar)
    d.text((80, 200), aadhaar, font=get_font(22), fill="black")

    img.save(DATASET_DIR / "aadhaar" / f"aadhaar_{i}.jpg")

# ================= PAN =================
def create_pan(i):
    img = Image.new("RGB", (400, 250), (220, 235, 245))
    d = ImageDraw.Draw(img)

    d.text((120, 10), "INCOME TAX DEPARTMENT", font=get_font(14), fill="black")
    d.text((150, 30), "GOVT. OF INDIA", font=get_font(12), fill="black")

    d.rectangle((20, 60, 90, 150), fill="gray")

    d.text((120, 70), "Name", font=get_font(10), fill="black")
    d.text((120, 85), random.choice(["RAJESH", "SNEHA", "AMIT"]), font=get_font(14), fill="black")

    d.text((120, 110), "DOB", font=get_font(10), fill="black")
    d.text((120, 125), random_date(), font=get_font(14), fill="black")

    pan = generate_unique_pan()
    assert validate_pan(pan)
    d.text((120, 190), pan, font=get_font(22), fill="black")

    img.save(DATASET_DIR / "pan" / f"pan_{i}.jpg")

# ================= PAYMENT RECEIPT =================
def create_payment_receipt(i):
    img = Image.new("RGB", (350, 600), "white")
    d = ImageDraw.Draw(img)

    d.text((100, 50), "Payment Successful", font=get_font(22), fill="green")
    d.text((100, 100), f"₹{random.randint(10,9999)}", font=get_font(26), fill="black")
    d.text((80, 150), "Txn ID: TXN" + "".join(random.choices(string.digits, k=10)), font=get_font(14), fill="black")

    img.save(DATASET_DIR / "payment_receipt" / f"receipt_{i}.jpg")

# ================= OTHER =================
def create_other(i):
    img = Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
    d = ImageDraw.Draw(img)
    d.text((40, 100), random.choice(["CAT", "TREE", "RANDOM"]), font=get_font(24), fill=random_color())
    img.save(DATASET_DIR / "other" / f"other_{i}.jpg")

# ================= MULTIPROCESS =================
def worker(task):
    i, cls = task
    if cls == "aadhaar":
        create_aadhaar(i)
    elif cls == "pan":
        create_pan(i)
    elif cls == "payment_receipt":
        create_payment_receipt(i)
    else:
        create_other(i)

if __name__ == "__main__":
    tasks = []
    for i in range(IMAGES_PER_CLASS):
        for c in CLASSES:
            tasks.append((i, c))

    with Pool(cpu_count()) as p:
        p.map(worker, tasks)

    print("✅ Dataset generated with VALID Aadhaar & PAN numbers")
