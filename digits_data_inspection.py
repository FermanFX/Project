import numpy as np
import matplotlib.pyplot as plt
from starter_pack.src.data_loader import DataLoader

'''
This script performs an in-depth inspection of the digits dataset, including:
- Basic dataset information (shape, data types, value ranges)
- Class distribution and split distribution
- Average digit images and variance visualization
- Interactive inspection of specific digits
'''

# Load digits data
loader = DataLoader()
digits = loader.load_digits()
X = digits['X']
y = digits['y']
train_idx = digits['train_idx']
val_idx = digits['val_idx']
test_idx = digits['test_idx']

print("=" * 60)
print("DIGITS DATASET INSPECTION")
print("=" * 60)

# Basic information
print(f"\nDATASET INFO:")
print(f"  Total samples: {X.shape[0]}")
print(f"  Features per sample: {X.shape[1]} (8x8 = 64 pixels)")
print(f"  Number of classes: {len(np.unique(y))} (digits 0-9)")
print(f"  Data type: {X.dtype}")
print(f"  Value range: [{X.min():.0f}, {X.max():.0f}]")

# Split information
print(f"\nDATA SPLITS:")
print(f"  Training: {len(train_idx)} samples")
print(f"  Validation: {len(val_idx)} samples")
print(f"  Test: {len(test_idx)} samples")

# Class distribution
print(f"\nCLASS DISTRIBUTION:")
unique, counts = np.unique(y, return_counts=True)
for digit, count in zip(unique, counts):
    percentage = count / len(y) * 100
    print(f"  Digit {digit}: {count} samples ({percentage:.1f}%)")

# Split distribution per class
print(f"\nSPLIT DISTRIBUTION PER CLASS:")
print(f"{'Digit':<6} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print("-" * 40)
for digit in range(10):
    train_count = np.sum(y[train_idx] == digit)
    val_count = np.sum(y[val_idx] == digit)
    test_count = np.sum(y[test_idx] == digit)
    total = train_count + val_count + test_count
    print(f"{digit:<6} {train_count:>8} {val_count:>8} {test_count:>8} {total:>8}")

print("\n" + "=" * 60)
print("VISUALIZING DIGIT IMAGES")
print("=" * 60)

# Function to display digits
def show_digits_grid(X, y, num_samples=25, title="Sample Digits"):
    """Display a grid of digit images."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.ravel()

    indices = np.random.choice(len(X), num_samples, replace=False)

    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(8, 8), cmap='gray')
        axes[i].set_title(f'Digit: {y[idx]}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# Show average digit images
print("\n" + "=" * 60)
print("AVERAGE DIGIT IMAGES")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for digit in range(10):
    digit_mask = y == digit
    digit_avg = X[digit_mask].mean(axis=0)
    axes[digit].imshow(digit_avg.reshape(8, 8), cmap='gray')
    axes[digit].set_title(f'Digit {digit}', fontsize=12)
    axes[digit].axis('off')

plt.suptitle('Average Image of Each Digit', fontsize=14)
plt.tight_layout()
plt.show()

# Show variance per digit
print("\n" + "=" * 60)
print("DIGIT VARIANCE (How much variation exists per digit)")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for digit in range(10):
    digit_mask = y == digit
    digit_std = X[digit_mask].std(axis=0)
    axes[digit].imshow(digit_std.reshape(8, 8), cmap='hot')
    axes[digit].set_title(f'Digit {digit} (Std Dev)', fontsize=12)
    axes[digit].axis('off')

plt.suptitle('Pixel Standard Deviation per Digit (higher = more variation)', fontsize=14)
plt.tight_layout()
plt.show()

# Confusion between similar digits
print("\n" + "=" * 60)
print("FINDING CONFUSABLE DIGIT PAIRS")
print("=" * 60)

# Compute pairwise distances between average digits
digit_means = []
for digit in range(10):
    digit_mask = y == digit
    digit_means.append(X[digit_mask].mean(axis=0))

digit_means = np.array(digit_means)

# Compute distances between average digits
distances = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        distances[i, j] = np.linalg.norm(digit_means[i] - digit_means[j])

# Find most similar digit pairs
print("\nMOST SIMILAR DIGIT PAIRS:")
similar_pairs = []
for i in range(10):
    for j in range(i+1, 10):
        similar_pairs.append((i, j, distances[i, j]))

similar_pairs.sort(key=lambda x: x[2])
for i, j, dist in similar_pairs[:5]:
    print(f"  Digit {i} and {j}: distance = {dist:.2f}")

# Interactive inspection
print("\n" + "=" * 60)
print("INTERACTIVE INSPECTION")
print("=" * 60)

while True:
    try:
        digit_input = input("\nEnter a digit to inspect (0-9) or 'q' to quit: ")
        if digit_input.lower() == 'q':
            break

        digit = int(digit_input)
        if digit not in range(10):
            print("Please enter a digit between 0 and 9")
            continue

        # Get all samples of this digit
        digit_indices = np.where(y == digit)[0]

        print(f"\nDIGIT {digit} STATISTICS:")
        print(f"  Total samples: {len(digit_indices)}")
        print(f"  In training: {np.sum(y[train_idx] == digit)}")
        print(f"  In validation: {np.sum(y[val_idx] == digit)}")
        print(f"  In test: {np.sum(y[test_idx] == digit)}")

        # Show 12 examples
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        axes = axes.ravel()

        sample_indices = np.random.choice(digit_indices, min(12, len(digit_indices)), replace=False)
        for i, idx in enumerate(sample_indices):
            axes[i].imshow(X[idx].reshape(8, 8), cmap='gray')
            axes[i].set_title(f'Digit {digit}', fontsize=10)
            axes[i].axis('off')

        for i in range(len(sample_indices), 12):
            axes[i].axis('off')

        plt.suptitle(f'Examples of Digit {digit}', fontsize=14)
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("Invalid input. Please enter a digit (0-9) or 'q' to quit.")
    except KeyboardInterrupt:
        break

print("\nInspection complete!")