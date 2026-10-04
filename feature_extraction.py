import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def compute_distance(p1, p2):
    """Calculates the 2D Euclidean distance between two 3D landmark points."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_features(frame):
    """
    Processes an image frame and extracts 7 facial features matching stress_data.csv:
    [left_eye, right_eye, mouth_open, mouth_width, left_eyebrow, right_eyebrow, face_ratio]
    
    Returns:
        numpy.ndarray: Shape (1, 7) array with calculated feature values,
                       or None if no face is detected.
    """
    if frame is None:
        return None

    # Convert BGR frame from OpenCV to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Return None if no face is detected in the frame
    if not results.multi_face_landmarks:
        return None

    # Get landmarks for the first detected face
    landmarks = results.multi_face_landmarks[0].landmark

    # --- Feature Calculations ---
    # 1. Left Eye Opening (Landmarks 386 & 374)
    left_eye = compute_distance(landmarks[386], landmarks[374])

    # 2. Right Eye Opening (Landmarks 159 & 145)
    right_eye = compute_distance(landmarks[159], landmarks[145])

    # 3. Mouth Opening (Landmarks 13 & 14)
    mouth_open = compute_distance(landmarks[13], landmarks[14])

    # 4. Mouth Width (Landmarks 61 & 291)
    mouth_width = compute_distance(landmarks[61], landmarks[291])

    # 5. Left Eyebrow Distance to Eye (Landmarks 70 & 159)
    left_eyebrow = compute_distance(landmarks[70], landmarks[159])

    # 6. Right Eyebrow Distance to Eye (Landmarks 300 & 386)
    right_eyebrow = compute_distance(landmarks[300], landmarks[386])

    # 7. Face Ratio (Height: Landmarks 10 to 152 / Width: Landmarks 234 to 454)
    face_height = compute_distance(landmarks[10], landmarks[152])
    face_width = compute_distance(landmarks[234], landmarks[454])
    face_ratio = face_height / face_width if face_width != 0 else 1.0

    # Bundle into a (1, 7) numpy array matching model expectations
    features = np.array([[
        left_eye,
        right_eye,
        mouth_open,
        mouth_width,
        left_eyebrow,
        right_eyebrow,
        face_ratio
    ]], dtype=np.float64)

    return features