# Experiment 7
# Image Compression using Huffman Coding and Run Length Coding

import cv2
import numpy as np
import heapq
from collections import defaultdict, Counter

# -----------------------------------
# Load grayscale image
# -----------------------------------
image = cv2.imread("images1.jpg", 0)

if image is None:
    print("Error: Image not found.")
    exit()

pixels = image.flatten()

# ===================================
# HUFFMAN CODING
# ===================================

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build frequency table
freq = Counter(pixels)

# Create priority queue
heap = [Node(sym, fr) for sym, fr in freq.items()]
heapq.heapify(heap)

# Build Huffman Tree
while len(heap) > 1:
    left = heapq.heappop(heap)
    right = heapq.heappop(heap)

    merged = Node(None, left.freq + right.freq)
    merged.left = left
    merged.right = right
    heapq.heappush(heap, merged)

root = heap[0]

# Generate Huffman Codes
codes = {}

def generate_codes(node, code=""):
    if node is None:
        return
    if node.symbol is not None:
        codes[node.symbol] = code
        return
    generate_codes(node.left, code + "0")
    generate_codes(node.right, code + "1")

generate_codes(root)

# Encode image
huffman_bits = ''.join(codes[p] for p in pixels)
huffman_size = len(huffman_bits)

# ===================================
# RUN LENGTH CODING
# ===================================

rle = []
count = 1

for i in range(1, len(pixels)):
    if pixels[i] == pixels[i - 1]:
        count += 1
    else:
        rle.append((pixels[i - 1], count))
        count = 1

rle.append((pixels[-1], count))

# Assume each pair uses 16 bits (8 for pixel, 8 for count)
rle_size = len(rle) * 16

# ===================================
# Original Size
# ===================================
original_size = len(pixels) * 8

# Compression Ratios
huffman_ratio = original_size / huffman_size
rle_ratio = original_size / rle_size

# ===================================
# Results
# ===================================
print("Original Size:", original_size, "bits")
print("Huffman Compressed Size:", huffman_size, "bits")
print("RLE Compressed Size:", rle_size, "bits")

print("\nCompression Ratio:")
print("Huffman Coding =", round(huffman_ratio, 2))
print("Run Length Coding =", round(rle_ratio, 2))

print("\nConclusion:")
print("Huffman Coding works better for images with uneven pixel frequency.")
print("Run Length Coding works better for images with repeated pixels.")
print("Natural images usually favor Huffman, binary/simple images favor RLE.")