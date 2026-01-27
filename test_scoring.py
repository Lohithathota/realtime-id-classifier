#!/usr/bin/env python3
"""Quick test to verify the scoring logic works correctly"""

from ocr_engine import score_name_quality, is_likely_ocr_garbage

# Test with the actual PAN candidates from the log
pan_candidates = [
    'saa faa ANd UAtAIX',
    'on tare Cal GA HTS', 
    'ama',
    'VANAM MEGHANA',
    'RAJESH KARNA VANAM',
    'aa At arias ae'
]

print("=== PAN Name Scoring Test ===\n")
scored = [(name, score_name_quality(name, "pan"), is_likely_ocr_garbage(name)) 
          for name in pan_candidates]
scored.sort(key=lambda x: x[1], reverse=True)

for name, score, is_garbage in scored:
    print(f"Score: {score:3d} | Garbage: {is_garbage} | '{name}'")

print(f"\n✓ Winner: '{scored[0][0]}' (score: {scored[0][1]})")
print(f"Expected: 'VANAM MEGHANA' or 'RAJESH KARNA VANAM'")
