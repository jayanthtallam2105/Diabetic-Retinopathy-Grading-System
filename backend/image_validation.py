"""
Image quality validation for retinal fundus images.
Performs checks for blur, brightness, contrast, and circular retina detection.
"""
import io
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image


def validate_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    """
    Validate retinal fundus image quality before inference.
    
    Returns:
        {
            "accepted": bool,
            "blur_score": float,
            "brightness_score": float (0-100),
            "retina_detected": bool,
            "image_quality_score": float (0-100),
            "message": str,
            "resolution": {"width": int, "height": int}
        }
    """
    try:
        # Load image
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img_pil)
        height, width = img_array.shape[:2]
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 1. Resolution check
        min_resolution = 224  # Minimum acceptable resolution
        if width < min_resolution or height < min_resolution:
            return {
                "accepted": False,
                "blur_score": 0.0,
                "brightness_score": 0.0,
                "retina_detected": False,
                "image_quality_score": 0.0,
                "message": f"Image resolution too low ({width}x{height}). Minimum required: {min_resolution}x{min_resolution}.",
                "resolution": {"width": width, "height": height},
            }
        
        # 2. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 100.0  # Lower = more blur
        
        # 3. Brightness check (mean pixel intensity)
        brightness_mean = np.mean(gray)
        brightness_score = (brightness_mean / 255.0) * 100.0
        brightness_min = 30.0  # Too dark
        brightness_max = 220.0  # Too bright
        
        # 4. Contrast check (standard deviation)
        contrast_std = np.std(gray)
        contrast_score = min((contrast_std / 64.0) * 100.0, 100.0)  # Normalize to 0-100
        contrast_min = 15.0  # Minimum acceptable contrast
        
        # 5. Circular retina detection using HoughCircles
        retina_detected = False
        circles = None
        
        # Preprocess for circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # HoughCircles parameters tuned for retinal fundus images
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(width, height) * 0.3),
            param1=50,
            param2=30,
            minRadius=int(min(width, height) * 0.15),
            maxRadius=int(min(width, height) * 0.45),
        )
        
        if circles is not None and len(circles[0]) > 0:
            retina_detected = True
        
        # Calculate overall quality score (0-100)
        quality_scores = []
        
        # Blur component (0-40 points)
        if blur_score >= blur_threshold:
            quality_scores.append(40.0)
        else:
            quality_scores.append((blur_score / blur_threshold) * 40.0)
        
        # Brightness component (0-20 points)
        if brightness_min <= brightness_mean <= brightness_max:
            quality_scores.append(20.0)
        elif brightness_mean < brightness_min:
            quality_scores.append((brightness_mean / brightness_min) * 20.0)
        else:
            quality_scores.append(((255.0 - brightness_mean) / (255.0 - brightness_max)) * 20.0)
        
        # Contrast component (0-20 points)
        if contrast_std >= contrast_min:
            quality_scores.append(20.0)
        else:
            quality_scores.append((contrast_std / contrast_min) * 20.0)
        
        # Retina detection component (0-20 points)
        if retina_detected:
            quality_scores.append(20.0)
        else:
            quality_scores.append(0.0)
        
        image_quality_score = sum(quality_scores)
        
        # Acceptance criteria
        accepted = (
            blur_score >= blur_threshold * 0.7  # Allow some tolerance
            and brightness_min * 0.8 <= brightness_mean <= brightness_max * 1.1
            and contrast_std >= contrast_min * 0.8
            and image_quality_score >= 50.0  # Minimum overall quality
        )
        
        # Generate message
        if accepted:
            if image_quality_score >= 80:
                quality_label = "Good"
            elif image_quality_score >= 60:
                quality_label = "Fair"
            else:
                quality_label = "Poor"
            message = f"Image quality: {quality_label} (Score: {image_quality_score:.1f}/100)"
        else:
            reasons = []
            if blur_score < blur_threshold * 0.7:
                reasons.append("too blurry")
            if brightness_mean < brightness_min * 0.8:
                reasons.append("too dark")
            elif brightness_mean > brightness_max * 1.1:
                reasons.append("too bright")
            if contrast_std < contrast_min * 0.8:
                reasons.append("low contrast")
            if not retina_detected:
                reasons.append("retinal structure not detected")
            if image_quality_score < 50.0:
                reasons.append("overall quality too low")
            
            message = f"Image rejected: {', '.join(reasons)}. Quality score: {image_quality_score:.1f}/100"
        
        return {
            "accepted": accepted,
            "blur_score": float(blur_score),
            "brightness_score": float(brightness_score),
            "retina_detected": retina_detected,
            "image_quality_score": float(image_quality_score),
            "message": message,
            "resolution": {"width": width, "height": height},
            "contrast_score": float(contrast_score),
        }
        
    except Exception as e:
        return {
            "accepted": False,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "retina_detected": False,
            "image_quality_score": 0.0,
            "message": f"Image validation failed: {str(e)}",
            "resolution": {"width": 0, "height": 0},
        }
