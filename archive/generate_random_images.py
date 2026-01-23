"""
Generate random images for the 'other' class (non-documents).
Creates random noise images that represent any random/non-document images.
"""

from PIL import Image, ImageDraw
import random
from pathlib import Path
import numpy as np


def generate_random_image(size=(224, 224), image_type='noise'):
    """
    Generate a random image.
    
    Args:
        size: Image size (width, height)
        image_type: Type of random image to generate
        
    Returns:
        PIL Image
    """
    if image_type == 'noise':
        # Pure random noise
        data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(data)
    
    elif image_type == 'geometric':
        # Random geometric shapes
        img = Image.new('RGB', size, color=(random.randint(100, 200), 
                                           random.randint(100, 200), 
                                           random.randint(100, 200)))
        draw = ImageDraw.Draw(img)
        
        # Draw random rectangles
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
            x2, y2 = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
        
        return img
    
    elif image_type == 'gradient':
        # Random gradient background
        data = np.zeros((*size, 3), dtype=np.uint8)
        start_color = [random.randint(0, 255) for _ in range(3)]
        end_color = [random.randint(0, 255) for _ in range(3)]
        
        for i in range(size[0]):
            ratio = i / size[0]
            color = [int(start_color[j] + (end_color[j] - start_color[j]) * ratio) for j in range(3)]
            data[i, :] = color
        
        return Image.fromarray(data.astype(np.uint8))
    
    elif image_type == 'circles':
        # Random circles
        img = Image.new('RGB', size, color=(random.randint(100, 200), 
                                           random.randint(100, 200), 
                                           random.randint(100, 200)))
        draw = ImageDraw.Draw(img)
        
        for _ in range(random.randint(5, 12)):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            r = random.randint(10, 50)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)
        
        return img


def main():
    """Generate random images for training"""
    
    train_other = Path("dataset/train/other")
    val_other = Path("dataset/val/other")
    
    print("=== Generating Random Images for 'Other' Class ===\n")
    
    # Generate training images
    print("Generating training images...")
    image_types = ['noise', 'geometric', 'gradient', 'circles']
    
    for i in range(4):
        for j in range(5):
            img_type = image_types[i % len(image_types)]
            img = generate_random_image(image_type=img_type)
            img_path = train_other / f"random_{i:02d}_{j:02d}.jpg"
            img.save(img_path)
            print(f"  Generated: {img_path.name}")
    
    print(f"\n✅ Created {len(list(train_other.glob('*.jpg')))} training images")
    
    # Generate validation images
    print("\nGenerating validation images...")
    for i in range(4):
        for j in range(1):
            img_type = image_types[i % len(image_types)]
            img = generate_random_image(image_type=img_type)
            img_path = val_other / f"random_val_{i:02d}_{j:02d}.jpg"
            img.save(img_path)
            print(f"  Generated: {img_path.name}")
    
    print(f"\n✅ Created {len(list(val_other.glob('*.jpg')))} validation images")
    
    total = len(list(train_other.glob('*.jpg'))) + len(list(val_other.glob('*.jpg')))
    print(f"\n✅ Total 'other' images generated: {total}")
    print("\nNext: Run augment_dataset.py to create augmented versions")


if __name__ == "__main__":
    main()
