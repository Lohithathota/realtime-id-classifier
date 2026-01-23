import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from multiprocessing import Pool, cpu_count

# output directories
DATASET_DIR = Path("dataset/train")
CLASSES = ["aadhaar", "pan", "payment_receipt", "other"]
IMAGES_PER_CLASS = 1000

# Ensure directories exist
for class_name in CLASSES:
    (DATASET_DIR / class_name).mkdir(parents=True, exist_ok=True)

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def random_text(length=10, digits=False):
    chars = string.digits if digits else string.ascii_uppercase + " "
    return ''.join(random.choices(chars, k=length))

def random_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1960, 2025)
    return f"{day:02d}/{month:02d}/{year}"

def get_random_font(size=20):
    # Try to load a standard font, fallback to default
    try:
        # Windows standard fonts
        fonts = ["arial.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf"]
        font_name = random.choice(fonts)
        return ImageFont.truetype(font_name, size)
    except:
        return ImageFont.load_default()

def create_aadhaar(index):
    # 1. Background
    bg_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
    img = Image.new('RGB', (400, 250), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 2. Header (India Flag colors or solid)
    if random.choice([True, False]):
        draw.rectangle([(0, 0), (400, 40)], fill=(255, 153, 51)) # Saffron
        draw.rectangle([(0, 40), (400, 50)], fill=(255, 255, 255)) # White
        draw.rectangle([(0, 50), (400, 60)], fill=(19, 136, 8)) # Green
    else:
        draw.rectangle([(0,0), (400, 50)], fill=random_color())

    # 3. Photo Placeholder
    photo_x = random.randint(10, 30)
    photo_y = random.randint(70, 90)
    draw.rectangle([(photo_x, photo_y), (photo_x+80, photo_y+100)], fill=(200, 200, 200), outline="black")
    
    # 4. Text Content (Name, DOB, Gender)
    font_main = get_random_font(18)
    font_small = get_random_font(14)
    font_number = get_random_font(24)
    
    text_x = photo_x + 100
    draw.text((text_x, 80), random_text(15).title(), font=font_main, fill="black")
    draw.text((text_x, 110), f"DOB: {random_date()}", font=font_small, fill="black")
    draw.text((text_x, 135), random.choice(["MALE", "FEMALE"]), font=font_small, fill="black")
    
    # 5. Aadhaar Number (Unique feature: 4 4 4 digits)
    aadhaar_num = f"{random_text(4, True)} {random_text(4, True)} {random_text(4, True)}"
    draw.text((80, 210), aadhaar_num, font=font_number, fill="black")
    
    # 6. Noise/Texture overlay
    if random.random() > 0.5:
        # Add a random line or shape
        draw.line([(random.randint(0,400), random.randint(0,250)), (random.randint(0,400), random.randint(0,250))], fill=random_color(), width=1)

    # Save
    img.save(DATASET_DIR / "aadhaar" / f"syn_aadhaar_{index}.jpg")

def create_pan(index):
    # PAN Cards usually light blue patterns
    bg_color = (random.randint(200, 240), random.randint(230, 255), random.randint(240, 255))
    img = Image.new('RGB', (400, 250), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.text((150, 10), "INCOME TAX DEPARTMENT", font=get_random_font(15), fill="black")
    draw.text((170, 30), "GOVT. OF INDIA", font=get_random_font(12), fill="black")
    
    # Photo & QR Placeholders
    draw.rectangle([(10, 60), (90, 160)], fill="gray", outline="black") # Photo
    draw.rectangle([(310, 60), (390, 140)], fill="white", outline="black") # QR
    
    # Details
    draw.text((110, 70), "Name", font=get_random_font(10), fill="black")
    draw.text((110, 85), random_text(20), font=get_random_font(16), fill="black")
    
    draw.text((110, 110), "Father's Name", font=get_random_font(10), fill="black")
    draw.text((110, 125), random_text(20), font=get_random_font(16), fill="black")
    
    draw.text((110, 150), "Date of Birth", font=get_random_font(10), fill="black")
    draw.text((110, 165), random_date(), font=get_random_font(16), fill="black")
    
    # PAN Number (Unique feature: 10 chars)
    pan_num = random_text(10)
    font_pan = get_random_font(22)
    # Center the PAN number a bit
    draw.text((120, 200), pan_num, font=font_pan, fill="black")
    
    img.save(DATASET_DIR / "pan" / f"syn_pan_{index}.jpg")

def create_payment_receipt(index):
    # White background usually
    img = Image.new('RGB', (350, 600), "white")
    draw = ImageDraw.Draw(img)
    
    # Green success circle often seen
    draw.ellipse([(150, 50), (200, 100)], fill=(30, 200, 30))
    
    # Amount
    amount = f"₹{random.randint(1, 10000)}.{random.randint(0,99):02d}"
    font_large = get_random_font(30)
    draw.text((120, 120), amount, font=font_large, fill="black")
    
    # "Paid to"
    draw.text((100, 170), "Paid to " + random_text(10).title(), font=get_random_font(18), fill="black")
    
    # Transaction Details box
    start_y = 250
    draw.line([(20, start_y), (330, start_y)], fill="gray")
    
    details = [
        ("Txn ID", "T" + random_text(12, True)),
        ("Ref No", random_text(10, True)),
        ("Date", random_date() + f" {random.randint(0,23):02d}:{random.randint(0,59):02d}"),
        ("Status", "Successful")
    ]
    
    for i, (label, val) in enumerate(details):
        y = start_y + 20 + (i * 40)
        draw.text((30, y), label, font=get_random_font(14), fill="gray")
        draw.text((200, y), val, font=get_random_font(14), fill="black")
        
    # App Logos (simulated with text)
    app_name = random.choice(["PhonePe", "Google Pay", "Paytm", "BHIM"])
    draw.text((130, 550), f"Powered by {app_name}", font=get_random_font(12), fill="gray")
    
    img.save(DATASET_DIR / "payment_receipt" / f"syn_receipt_{index}.jpg")

def create_other(index):
    # Completely random
    img_type = random.choice(["noise", "shapes", "gradient"])
    
    if img_type == "noise":
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        
    elif img_type == "shapes":
        bg = random_color()
        img = Image.new('RGB', (224, 224), bg)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 10)):
            x1 = random.randint(0, 200)
            y1 = random.randint(0, 200)
            draw.rectangle([x1, y1, x1+random.randint(10,50), y1+random.randint(10,50)], fill=random_color())
            draw.ellipse([x1, y1, x1+random.randint(10,50), y1+random.randint(10,50)], fill=random_color())
            
    else: # Gradient-ish
        img = Image.new('RGB', (224, 224), random_color())
        draw = ImageDraw.Draw(img)
        for i in range(0, 224, 20):
            draw.line([(i, 0), (224-i, 224)], fill=random_color(), width=5)
            
    # Add some random text unrelated to docs
    draw = ImageDraw.Draw(img)
    if random.choice([True, False]):
        draw.text((50, 100), random.choice(["CAT", "DOG", "TREE", "CAR", "SKY", "VACATION"]), font=get_random_font(25), fill=random_color())

    img.save(DATASET_DIR / "other" / f"syn_other_{index}.jpg")

def generate(i):
    # This function is called by the pool
    # We use (index % 4) to distribute work, or just randomize
    # Actually, the user wants 1000 of EACH.
    # So we will run this 4000 times, but the main loop handles offsets.
    pass 

def worker(args):
    idx, kind = args
    if kind == "aadhaar":
        create_aadhaar(idx)
    elif kind == "pan":
        create_pan(idx)
    elif kind == "payment_receipt":
        create_payment_receipt(idx)
    elif kind == "other":
        create_other(idx)

if __name__ == "__main__":
    tasks = []
    print(f"Generating {IMAGES_PER_CLASS} images per class...")
    
    for i in range(IMAGES_PER_CLASS):
        tasks.append((i, "aadhaar"))
        tasks.append((i, "pan"))
        tasks.append((i, "payment_receipt"))
        tasks.append((i, "other"))
        
    # Use multiprocessing to speed it up
    with Pool(processes=cpu_count()) as pool:
        pool.map(worker, tasks)
        
    print("Done! Generated 4000 images.")
