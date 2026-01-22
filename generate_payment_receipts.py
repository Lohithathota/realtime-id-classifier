"""
Generate synthetic payment receipt images (PhonePe, Paytm, Google Pay style).
"""

from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
import textwrap


def generate_phonepe_receipt(size=(224, 224)):
    """Generate a PhonePe-style payment receipt"""
    img = Image.new('RGB', size, color=(0, 100, 200))  # PhonePe blue
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([0, 0, size[0], 40], fill=(0, 80, 180))
    draw.text((10, 5), "PhonePe", fill=(255, 255, 255))
    
    # Content
    y = 50
    line_height = 25
    
    # Transaction details
    details = [
        f"To: {random.choice(['John', 'Sarah', 'Mike', 'Emma'])}",
        f"Amount: Rs. {random.randint(100, 50000)}.00",
        f"Date: {random.randint(1, 28)}/01/2026",
        f"Time: {random.randint(8, 22)}:{random.randint(0, 59):02d} PM",
        "Status: SUCCESS",
        f"Ref ID: {random.randint(100000, 999999)}"
    ]
    
    for detail in details:
        draw.text((15, y), detail, fill=(255, 255, 255))
        y += line_height
    
    # Footer
    draw.rectangle([0, size[1]-20, size[0], size[1]], fill=(0, 80, 180))
    
    return img


def generate_paytm_receipt(size=(224, 224)):
    """Generate a Paytm-style payment receipt"""
    img = Image.new('RGB', size, color=(0, 150, 136))  # Paytm teal
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([0, 0, size[0], 40], fill=(0, 120, 110))
    draw.text((10, 5), "Paytm", fill=(255, 255, 255))
    
    # Content
    y = 50
    line_height = 25
    
    details = [
        f"Merchant: {random.choice(['Store', 'Shop', 'Store', 'Mall'])}",
        f"Amount: Rs. {random.randint(100, 50000)}.00",
        f"Date: {random.randint(1, 28)}/01/2026",
        f"Order: PAY{random.randint(100000, 999999)}",
        "Payment: Completed",
        f"TxnID: {random.randint(1000000000, 9999999999)}"
    ]
    
    for detail in details:
        draw.text((15, y), detail, fill=(255, 255, 255))
        y += line_height
    
    # Footer
    draw.rectangle([0, size[1]-20, size[0], size[1]], fill=(0, 120, 110))
    
    return img


def generate_googlepay_receipt(size=(224, 224)):
    """Generate a Google Pay-style payment receipt"""
    img = Image.new('RGB', size, color=(255, 255, 255))  # Google Pay white
    draw = ImageDraw.Draw(img)
    
    # Header
    draw.rectangle([0, 0, size[0], 40], fill=(245, 245, 245))
    draw.text((10, 5), "Google Pay", fill=(51, 103, 214))
    
    # Content
    y = 50
    line_height = 25
    
    details = [
        f"Sent to: {random.choice(['9876543210', '9123456789', '8765432109'])}",
        f"Amount: Rs. {random.randint(100, 50000)}.00",
        f"Date: {random.randint(1, 28)} Jan 2026",
        f"Time: {random.randint(8, 22)}:{random.randint(0, 59):02d} PM",
        "Status: Delivered",
        f"Ref: GP{random.randint(100000, 999999)}"
    ]
    
    for detail in details:
        draw.text((15, y), detail, fill=(51, 103, 214))
        y += line_height
    
    # Footer
    draw.rectangle([0, size[1]-20, size[0], size[1]], fill=(245, 245, 245))
    
    return img


def main():
    """Generate payment receipt images"""
    
    train_receipt = Path("dataset/train/payment_receipt")
    val_receipt = Path("dataset/val/payment_receipt")
    
    print("=== Generating Payment Receipt Images ===\n")
    
    # Receipt generators
    generators = [
        ("phonepe", generate_phonepe_receipt),
        ("paytm", generate_paytm_receipt),
        ("googlepay", generate_googlepay_receipt)
    ]
    
    # Generate training receipts
    print("Generating training receipts...")
    for i in range(3):  # 3 types of receipts
        for j in range(5):  # 5 of each type = 15 training images
            receipt_type, generator = generators[i % len(generators)]
            img = generator()
            img_path = train_receipt / f"{receipt_type}_{i:02d}_{j:02d}.jpg"
            img.save(img_path)
            print(f"  Generated: {img_path.name}")
    
    train_count = len(list(train_receipt.glob('*.jpg')))
    print(f"\n✅ Created {train_count} training payment receipts")
    
    # Generate validation receipts
    print("\nGenerating validation receipts...")
    for i in range(3):
        for j in range(1):
            receipt_type, generator = generators[i % len(generators)]
            img = generator()
            img_path = val_receipt / f"{receipt_type}_val_{i:02d}_{j:02d}.jpg"
            img.save(img_path)
            print(f"  Generated: {img_path.name}")
    
    val_count = len(list(val_receipt.glob('*.jpg')))
    print(f"\n✅ Created {val_count} validation payment receipts")
    
    total = train_count + val_count
    print(f"\n✅ Total payment receipt images: {total}")
    print("\nNext: Run augment dataset and retrain model")


if __name__ == "__main__":
    main()
