import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter

img = cv2.imread('/content/Misc_72.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
M, N = img.shape

print(img)

plt.figure(figsize=(16, 6))
plt.plot(1, 4, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

def compute_point_spread_indicator(I0, Im, In):
    # Avoid log(0) issues
    if I0 <= 0 or Im <= 0 or In <= 0:
        return 0
    numerator = np.log(I0) - np.log(Im)
    denominator = np.log(I0) - np.log(In)
    if denominator == 0:
        return 0
    return numerator / denominator

def point_spread_filter(img):
    padded = np.pad(img, 1, mode='reflect')
    filtered = np.zeros_like(img)

    for i in range(M):
        for j in range(N):
            c = padded[i+1, j+1]
            direct = [padded[i+1, j], padded[i+1, j+2], padded[i, j+1], padded[i+2, j+1]]
            diagonal = [padded[i, j], padded[i, j+2], padded[i+2, j], padded[i+2, j+2]]
            Im = np.mean(direct)
            In = np.mean(diagonal)
            p = compute_point_spread_indicator(c, Im, In)
            if 0.35 <= p <= 0.65:
                filtered[i, j] = c
            else:
                filtered[i, j] = np.median(padded[i:i+3, j:j+3])
    return filtered

filtered_img = point_spread_filter(img)
print(filtered_img)
plt.plot(1, 4, 2)
plt.title("Filtered")
plt.imshow(filtered_img, cmap='gray')

def point_spread_local_contrast(img):
    padded = np.pad(img, 1, mode='reflect')
    R = np.zeros_like(img)

    for i in range(M):
        for j in range(N):
            I0 = padded[i+1, j+1]
            neighbors = [padded[i+1, j], padded[i+1, j+2], padded[i, j+1], padded[i+2, j+1],
                         padded[i, j], padded[i, j+2], padded[i+2, j], padded[i+2, j+2]]
            direct = [padded[i+1, j], padded[i+1, j+2], padded[i, j+1], padded[i+2, j+1]]
            L = min([I0**2 / (n + 1e-6) for n in neighbors])
            r = (min(direct) / (max(direct) + 1e-6))
            R[i, j] = r * L
    return R

PSLCM = point_spread_local_contrast(filtered_img)

plt.plot(1, 4, 3)
plt.title("Enhanced")
plt.imshow(PSLCM, cmap='gray')

def high_boost_enhancement(PSLCM):
    MF = cv2.GaussianBlur(PSLCM,(9,9),1)  # Approx 9x9
    B = np.maximum(PSLCM - MF, 1)
    enhanced = PSLCM * B
    return enhanced

enhanced_img = high_boost_enhancement(PSLCM)

def normalize_image(enhanced_img):
    norm = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

enhanced_image=normalize_image(enhanced_img)

print(np.sort(enhanced_image.flatten())[::-1])
print(enhanced_image)
plt.plot(1, 4, 3)
plt.title("Enhanced")
plt.imshow(enhanced_image, cmap='hot')

def threshold_image(enhanced_image, method: str = 'static', thresh_val: int = 200):
    if method == 'static':
        thresh = cv2.inRange(enhanced_image, thresh_val, 255)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(
            enhanced_image, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:
        raise ValueError("Unknown threshold method. Use 'static' or 'adaptive'.")
    return thresh

threshed_img=threshold_image(enhanced_image)

plt.plot(1, 4, 3)
plt.title("Enhanced")
plt.imshow(threshed_img, cmap='hot')

def adaptive_segmentation(enhanced):
    V = np.sort(enhanced.flatten())[::-1]
    S = np.unique(V)
    len_S = len(S)
    for i in range(min(10, len_S - 2), 1, -1):
        s0, s1, s2 = S[i], S[i + 1], S[i + 2]
        r = (s0 - s1) / (s1 - s2 + 1e-6)
        if r > 5:
            T = s0
            break
        elif r < 0.2:
            T = s1
            break
    else:
        T = V[10] if len(V) > 10 else V[-1]
    mask = (enhanced >= T).astype(np.uint8) * 255
    return mask

binary_mask = adaptive_segmentation(enhanced_image)

plt.plot(1, 4, 4)
plt.title("Target Mask")
plt.imshow(binary_mask, cmap='gray')
plt.tight_layout()
plt.show()

if img.dtype != np.uint8:
    original_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    original_img = img.copy()

# Convert original to BGR if it's grayscale
if len(original_img.shape) == 2:
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
else:
    original_bgr = original_img.copy()

# Create colored mask
colored_mask = np.zeros_like(original_bgr, dtype=np.uint8)
colored_mask[threshed_img > 0] = (0, 255, 0)

# Blend with original image
overlay = cv2.addWeighted(original_bgr, 1-0.5, colored_mask, 0.5, 0)

result= overlay.astype(np.uint8)

plt.plot(1, 4, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Transparent Overlay')
plt.axis('off')

