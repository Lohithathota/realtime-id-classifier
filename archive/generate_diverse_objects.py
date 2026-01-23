"""
Generate more diverse 'other' class images to improve model generalization.
Creates images that look like: toys, dolls, animals, objects, etc.
"""

from PIL import Image, ImageDraw
import random
from pathlib import Path
import math


def generate_toy_bear(size=(224, 224)):
    """Generate a teddy bear-like toy"""
    img = Image.new('RGB', size, color=(200, 180, 150))
    draw = ImageDraw.Draw(img)
    
    # Head
    draw.ellipse([60, 40, 140, 120], fill=(139, 69, 19), outline=(80, 40, 10), width=3)
    
    # Ears
    draw.ellipse([70, 20, 100, 50], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    draw.ellipse([110, 20, 140, 50], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    
    # Eyes
    draw.ellipse([80, 60, 95, 75], fill=(0, 0, 0))
    draw.ellipse([110, 60, 125, 75], fill=(0, 0, 0))
    
    # Snout
    draw.ellipse([85, 85, 115, 105], fill=(180, 140, 100))
    
    # Body
    draw.ellipse([50, 120, 150, 200], fill=(139, 69, 19), outline=(80, 40, 10), width=3)
    
    # Arms
    draw.ellipse([30, 130, 60, 170], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    draw.ellipse([160, 130, 190, 170], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    
    # Legs
    draw.ellipse([65, 200, 95, 220], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    draw.ellipse([115, 200, 145, 220], fill=(139, 69, 19), outline=(80, 40, 10), width=2)
    
    return img


def generate_flower(size=(224, 224)):
    """Generate a flower image"""
    img = Image.new('RGB', size, color=(144, 238, 144))  # Light green background
    draw = ImageDraw.Draw(img)
    
    cx, cy = size[0] // 2, size[1] // 2
    
    # Stem
    draw.line([cx, cy+50, cx, cy+180], fill=(34, 139, 34), width=4)
    
    # Leaves
    draw.ellipse([cx-30, cy+80, cx-10, cy+120], fill=(50, 150, 50))
    draw.ellipse([cx+10, cy+100, cx+30, cy+140], fill=(50, 150, 50))
    
    # Petals (flower)
    petal_radius = 30
    petal_distance = 40
    num_petals = 5
    
    for i in range(num_petals):
        angle = (2 * math.pi * i) / num_petals
        px = cx + petal_distance * math.cos(angle)
        py = cy - petal_distance * math.sin(angle)
        draw.ellipse([px-petal_radius, py-petal_radius, px+petal_radius, py+petal_radius], 
                    fill=(255, 100, 200))  # Pink petals
    
    # Center
    draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill=(255, 200, 0))  # Yellow center
    
    return img


def generate_car(size=(224, 224)):
    """Generate a car-like vehicle"""
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Body
    draw.rectangle([40, 100, 180, 140], fill=(255, 0, 0), outline=(200, 0, 0), width=2)
    
    # Roof
    draw.polygon([(60, 100), (120, 60), (160, 100)], fill=(200, 0, 0), outline=(150, 0, 0), width=2)
    
    # Windows
    draw.rectangle([70, 80, 110, 100], fill=(173, 216, 230))  # Light blue
    draw.rectangle([120, 80, 160, 100], fill=(173, 216, 230))
    
    # Door handle
    draw.rectangle([110, 110, 115, 130], fill=(100, 100, 100))
    
    # Wheels
    draw.ellipse([50, 135, 70, 155], fill=(0, 0, 0))
    draw.ellipse([150, 135, 170, 155], fill=(0, 0, 0))
    
    # Bumper
    draw.rectangle([30, 140, 40, 145], fill=(0, 0, 0))
    draw.rectangle([180, 140, 190, 145], fill=(0, 0, 0))
    
    return img


def generate_soccer_ball(size=(224, 224)):
    """Generate a soccer ball"""
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    cx, cy = size[0] // 2, size[1] // 2
    radius = 60
    
    # Main circle (ball)
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Pentagons and hexagons pattern
    for i in range(3):
        angle = (2 * math.pi * i) / 3
        px = cx + 40 * math.cos(angle)
        py = cy + 40 * math.sin(angle)
        draw.regular_polygon((px, py, 20), 5, fill=(0, 0, 0))  # Black pentagon
    
    for i in range(3):
        angle = (2 * math.pi * i + math.pi/3) / 3
        px = cx + 30 * math.cos(angle)
        py = cy + 30 * math.sin(angle)
        draw.ellipse([px-10, py-10, px+10, py+10], fill=(0, 0, 0))  # Dots
    
    return img


def generate_book(size=(224, 224)):
    """Generate a book"""
    img = Image.new('RGB', size, color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # Book cover
    draw.rectangle([50, 40, 170, 200], fill=(200, 100, 50), outline=(100, 50, 0), width=3)
    
    # Book spine (3D effect)
    draw.polygon([(50, 40), (60, 30), (180, 30), (170, 40)], fill=(150, 75, 25))
    draw.polygon([(170, 40), (180, 30), (180, 190), (170, 200)], fill=(150, 75, 25))
    
    # Title area
    draw.rectangle([60, 80, 160, 110], fill=(255, 255, 200))
    draw.text((70, 85), "BOOK", fill=(0, 0, 0))
    
    # Page lines
    for y in [130, 145, 160, 175]:
        draw.line([(60, y), (160, y)], fill=(100, 100, 100), width=1)
    
    return img


def generate_clock(size=(224, 224)):
    """Generate a clock"""
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    cx, cy = size[0] // 2, size[1] // 2
    radius = 50
    
    # Clock face
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=(255, 255, 200), outline=(0, 0, 0), width=3)
    
    # Numbers (simple marks)
    for i in range(12):
        angle = (2 * math.pi * i) / 12 - math.pi/2
        x1 = cx + (radius - 10) * math.cos(angle)
        y1 = cy + (radius - 10) * math.sin(angle)
        x2 = cx + radius * math.cos(angle)
        y2 = cy + radius * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)
    
    # Hour hand
    h_angle = math.pi/3
    draw.line([(cx, cy), (cx + 25 * math.cos(h_angle), cy + 25 * math.sin(h_angle))], 
             fill=(0, 0, 0), width=4)
    
    # Minute hand
    m_angle = 2 * math.pi / 3
    draw.line([(cx, cy), (cx + 35 * math.cos(m_angle), cy + 35 * math.sin(m_angle))], 
             fill=(0, 0, 0), width=3)
    
    # Center dot
    draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill=(0, 0, 0))
    
    return img


def main():
    """Generate diverse other class images"""
    
    train_other = Path("dataset/train/other")
    val_other = Path("dataset/val/other")
    
    print("=== Generating Diverse 'Other' Class Images ===\n")
    
    generators = [
        ("bear", generate_toy_bear),
        ("flower", generate_flower),
        ("car", generate_car),
        ("ball", generate_soccer_ball),
        ("book", generate_book),
        ("clock", generate_clock),
    ]
    
    # Generate training images
    print("Generating training images...")
    count = 0
    for i in range(3):  # 3 rounds
        for obj_type, generator in generators:
            img = generator()
            img_path = train_other / f"{obj_type}_{i:02d}.jpg"
            img.save(img_path)
            count += 1
            print(f"  Generated: {img_path.name}")
    
    train_count = len(list(train_other.glob('*.jpg')))
    print(f"\n✅ Total training 'other' images: {train_count}")
    
    # Generate validation images
    print("\nGenerating validation images...")
    val_count = 0
    for obj_type, generator in generators:
        img = generator()
        img_path = val_other / f"{obj_type}_val.jpg"
        img.save(img_path)
        val_count += 1
        print(f"  Generated: {img_path.name}")
    
    val_total = len(list(val_other.glob('*.jpg')))
    print(f"\n✅ Total validation 'other' images: {val_total}")
    print(f"\n✅ Total new 'other' images: {train_count + val_total}")
    print("\nNext: Run augmentation and retrain the model")


if __name__ == "__main__":
    main()
