import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import pygame
import os
import time # Import time for potential future debugging/timing

# -----------------------------
# SUPPRESS TFLITE LOGS
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------
# INIT PYGAME SOUNDS & MULTIPLE CHANNELS (MODIFIED)
# -----------------------------
pygame.mixer.init()
# Use 2 channels for hands (0-1) and one for animals (2)
MAX_HANDS = 1 
HAND_CHANNELS = [pygame.mixer.Channel(i) for i in range(MAX_HANDS)]
ANIMAL_CHANNEL = pygame.mixer.Channel(MAX_HANDS) 

# Habitat colors
HABITATS = ["blue", "green", "red", "beige"]

SOUNDS = {
    # "blue": pygame.mixer.Sound("Sounds_VF/BASIC_v3/BASIC-BASIC-sparse-live-8bars.wav"),
    # "blue": pygame.mixer.Sound("Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Sparse Live Bottom.wav"),
    "blue": pygame.mixer.Sound("Sounds_VF/SPEECH_US_v1_x8_REPEATS/Sparse-x8-100percent-USA_v1.aif"),

    # "green": pygame.mixer.Sound("Sounds_VF/BASIC_v3/BASIC-BASIC-dense-live-8bars-v3.wav"),
    # "green": pygame.mixer.Sound("Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Dense Live Bottom.wav"),
    "green": pygame.mixer.Sound("Sounds_VF/SPEECH_US_v1_x8_REPEATS/Dense-x8-100percent-USA_v1.aif"),

    # "red": pygame.mixer.Sound("Sounds_VF/BASIC_v3/BASIC-BASIC-rippled-sand-8bars-v3.wav"),
    # "red": pygame.mixer.Sound("Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Rippled Sand.wav"),
    "red": pygame.mixer.Sound("Sounds_VF/SPEECH_US_v1_x8_REPEATS/Rippled-x8-100percent-USA_v1.aif"),

    # "beige": pygame.mixer.Sound("Sounds_VF/BASIC_v3/BASIC-BASIC-flat-sand-8bars-v3.wav"),
    # "beige": pygame.mixer.Sound("Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Flat Sand.wav"),
    "beige": pygame.mixer.Sound("Sounds_VF/SPEECH_US_v1_x8_REPEATS/Flat-x8-100percent-USA_v1.aif"),

    "blue_caption": pygame.mixer.Sound("sounds/Habitat Descriptions - Sparse Live Bottom.wav"),
    "green_caption": pygame.mixer.Sound("sounds/Habitat Descriptions - Dense Live Bottom.wav"),
    "red_caption": pygame.mixer.Sound("sounds/Habitat Descriptions - Rippled Sand.wav"),
    "beige_caption": pygame.mixer.Sound("sounds/Habitat Descriptions - Flat Sand.wav"),

    "blue_caption_narrator": pygame.mixer.Sound("sounds/Habitat_Legend-NarratorONLY-Sparse_Live_Bottom.wav"),
    "green_caption_narrator": pygame.mixer.Sound("sounds/Habitat_Legend-NarratorONLY-Dense_Live_Bottom.wav"),
    "red_caption_narrator": pygame.mixer.Sound("sounds/Habitat_Legend-NarratorONLY-Rippled_Sand.wav"),
    "beige_caption_narrator": pygame.mixer.Sound("sounds/Habitat_Legend-NarratorONLY-Flat_Sand.wav"),

    "correct": pygame.mixer.Sound("sounds/right_2.mp3"),
    "wrong": pygame.mixer.Sound("sounds/wrong.mp3"),
}

# Animal → allowed habitat
ANIMAL_HABITATS = {
    9: ["blue"], #coral  
    4: ["blue", "beige"], #crab
    5: ["green"], #turtle
    6: ["red", "beige"], #ray
    7: ["blue"], #anemone
    8: ["red", "beige"] #shell
}

# -----------------------------
# LOAD MAP & MASKS
# -----------------------------
MAP_IMAGE_ID10 = cv2.imread("craved_map/craved_map1.jpeg")
MAP_IMAGE_ID9 = cv2.imread("craved_map2/Frame 7_id9.png") 
MAP_IMAGE_ID8 = cv2.imread("craved_map3/Frame 8_id8.png")
MAP_IMAGE_ID4 = cv2.imread("map_6x4/map_6x4.jpeg")

# Use a default map image for initial dimensions
if MAP_IMAGE_ID10 is None:
    print("Error: Could not load default map image 'craved_map/craved_map1.jpeg'")
    exit() 

# Set initial MAP_H and MAP_W based on a loaded map
if MAP_IMAGE_ID4 is not None:
    MAP_H, MAP_W, _ = MAP_IMAGE_ID4.shape
else: # Fallback if MAP_IMAGE_ID4 fails
    MAP_H, MAP_W, _ = MAP_IMAGE_ID10.shape


MASKS_ID10 = {
    'blue': cv2.imread("craved_map/masks/blue_mask.png", cv2.IMREAD_GRAYSCALE),
    'green': cv2.imread("craved_map/masks/green_mask.png", cv2.IMREAD_GRAYSCALE),
    'red': cv2.imread("craved_map/masks/red_mask.png", cv2.IMREAD_GRAYSCALE),
    'beige': cv2.imread("craved_map/masks/beige_mask.png", cv2.IMREAD_GRAYSCALE),
    'blue_caption': cv2.imread("craved_map/masks/blue_caption.png", cv2.IMREAD_GRAYSCALE),
    'green_caption': cv2.imread("craved_map/masks/green_caption.png", cv2.IMREAD_GRAYSCALE),
    'red_caption': cv2.imread("craved_map/masks/red_caption.png", cv2.IMREAD_GRAYSCALE),
    'beige_caption': cv2.imread("craved_map/masks/beige_caption.png", cv2.IMREAD_GRAYSCALE),
}

MASKS_ID9 = {
    'blue': cv2.imread("craved_map2/masks/blue_mask.png", cv2.IMREAD_GRAYSCALE),
    'green': cv2.imread("craved_map2/masks/green_mask.png", cv2.IMREAD_GRAYSCALE),
    'red': cv2.imread("craved_map2/masks/red_mask.png", cv2.IMREAD_GRAYSCALE),
    'beige': cv2.imread("craved_map2/masks/beige_mask.png", cv2.IMREAD_GRAYSCALE),
    'blue_caption': cv2.imread("craved_map2/masks/blue_caption.png", cv2.IMREAD_GRAYSCALE),
    'green_caption': cv2.imread("craved_map2/masks/green_caption.png", cv2.IMREAD_GRAYSCALE),
    'red_caption': cv2.imread("craved_map2/masks/red_caption.png", cv2.IMREAD_GRAYSCALE),
    'beige_caption': cv2.imread("craved_map2/masks/beige_caption.png", cv2.IMREAD_GRAYSCALE),
}

MASKS_ID8 = {
    'blue': cv2.imread("craved_map3/masks/blue_mask.png", cv2.IMREAD_GRAYSCALE),
    'green': cv2.imread("craved_map3/masks/green_mask.png", cv2.IMREAD_GRAYSCALE),
    'red': cv2.imread("craved_map3/masks/red_mask.png", cv2.IMREAD_GRAYSCALE),
    'beige': cv2.imread("craved_map3/masks/beige_mask.png", cv2.IMREAD_GRAYSCALE),
    'blue_caption': cv2.imread("craved_map3/masks/blue_caption.png", cv2.IMREAD_GRAYSCALE),
    'green_caption': cv2.imread("craved_map3/masks/green_caption.png", cv2.IMREAD_GRAYSCALE),
    'red_caption': cv2.imread("craved_map3/masks/red_caption.png", cv2.IMREAD_GRAYSCALE),
    'beige_caption': cv2.imread("craved_map3/masks/beige_caption.png", cv2.IMREAD_GRAYSCALE),
}

MASKS_ID4 = {
    'blue': cv2.imread("map_6x4/masks/blue_mask.png", cv2.IMREAD_GRAYSCALE),
    'green': cv2.imread("map_6x4/masks/green_mask.png", cv2.IMREAD_GRAYSCALE),
    'red': cv2.imread("map_6x4/masks/red_mask.png", cv2.IMREAD_GRAYSCALE),
    'beige': cv2.imread("map_6x4/masks/beige_mask.png", cv2.IMREAD_GRAYSCALE),
    # 'blue_caption': cv2.imread("craved_map3/masks/blue_caption.png", cv2.IMREAD_GRAYSCALE),
    # 'green_caption': cv2.imread("craved_map3/masks/green_caption.png", cv2.IMREAD_GRAYSCALE),
    # 'red_caption': cv2.imread("craved_map3/masks/red_caption.png", cv2.IMREAD_GRAYSCALE),
    # 'beige_caption': cv2.imread("craved_map3/masks/beige_caption.png", cv2.IMREAD_GRAYSCALE),
}

MAP_CONFIG = {
    11: {'image': MAP_IMAGE_ID10, 'masks': MASKS_ID10},
    9: {'image': MAP_IMAGE_ID9, 'masks': MASKS_ID9},
    3: {'image': MAP_IMAGE_ID8, 'masks': MASKS_ID8},
    4: {'image': MAP_IMAGE_ID4, 'masks': MASKS_ID4},
}

ACTIVE_MAP_ID = 11
MAP_IMAGE = MAP_IMAGE_ID10
MASKS = MASKS_ID10

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def check_region(mx, my):
    """Return the mask key at the given map coordinates."""
    # Ensure MASKS is not None before iterating
    if MASKS is None:
        return None
        
    for key, mask in MASKS.items():
        if mask is not None and 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
            if mask[my, mx] > 0:
                return key
    return None

def reset_homography():
    """Resets the homography and history for a manual refresh."""
    global averaged_h_matrix, last_marker_id, h_matrix_history
    print("Manual Homography Reset Triggered.")
    averaged_h_matrix = None
    # Note: last_marker_id is intentionally NOT reset here to allow re-registration 
    # on the current map marker.
    h_matrix_history = []

# --- NEW: Homography Calculation Functions ---
# Removed MAP_CONFIG from arguments as global MAP_CONFIG is used
def calculate_homography(marker_id, marker_corners_f, TL_OFFSET_X, TL_OFFSET_Y, BR_OFFSET_X, BR_OFFSET_Y, BL_OFFSET_X, BL_OFFSET_Y):
    """Calculates a single homography matrix for the given marker, applying offsets."""
    
    # Use globals to update map settings and reference current map size
    global MAP_IMAGE, MASKS, ACTIVE_MAP_ID, MAP_H, MAP_W, last_marker_id, MAP_CONFIG
    
    # 1. Update map/masks if the marker ID changed (or if it's the first detection)
    if last_marker_id is None or last_marker_id != marker_id:
        if marker_id in MAP_CONFIG:
            MAP_IMAGE = MAP_CONFIG[marker_id]['image']
            MASKS = MAP_CONFIG[marker_id]['masks']
            ACTIVE_MAP_ID = marker_id
            # IMPORTANT: Update global dimensions here
            MAP_H, MAP_W, _ = MAP_IMAGE.shape
            print(f"Map switched to ID {marker_id}. New dimensions: {MAP_W}x{MAP_H}")
            
    # 2. Define map_marker_pts with the manual offsets, using the updated global MAP_W/MAP_H
    # Corner Order: Top-Left (0), Top-Right (1), Bottom-Right (2), Bottom-Left (3)
    
    if marker_id == 3:
        map_marker_pts = np.array([
            # Top-Left
            [MAP_W-490 + TL_OFFSET_X, 0 + TL_OFFSET_Y], 
            # Top-Right (fixed)
            [MAP_W, 0],
            # Bottom-Right
            [MAP_W + BR_OFFSET_X, 492 + BR_OFFSET_Y],
            # Bottom-Left
            [MAP_W-490 + BL_OFFSET_X, 492 + BL_OFFSET_Y]
        ], dtype=np.float32)
    elif marker_id == 9:
        map_marker_pts = np.array([
            # Top-Left
            [MAP_W-490 + TL_OFFSET_X, 0 + TL_OFFSET_Y], 
            # Top-Right (fixed)
            [MAP_W, 0],
            # Bottom-Right
            [MAP_W + BR_OFFSET_X, 492 + BR_OFFSET_Y],
            # Bottom-Left
            [MAP_W-490 + BL_OFFSET_X, 492 + BL_OFFSET_Y] 
        ], dtype=np.float32)
    elif marker_id == 4:
        map_marker_pts = np.array([
            # Top-Left
            [MAP_W - 344 + TL_OFFSET_X, 30 + TL_OFFSET_Y],
            # Top-Right (fixed)
            [MAP_W - 42, 30],
            # Bottom-Right
            [MAP_W - 42 + BR_OFFSET_X, 338 + BR_OFFSET_Y],
            # Bottom-Left
            [MAP_W - 344 + BL_OFFSET_X, 338 + BL_OFFSET_Y] 
        ], dtype=np.float32)
    elif marker_id == 11:
        map_marker_pts = np.array([
            # Top-Left
            [MAP_W - 277 + TL_OFFSET_X, 0 + TL_OFFSET_Y],
            # Top-Right (fixed)
            [MAP_W, 0],
            # Bottom-Right
            [MAP_W + BR_OFFSET_X, 277 + BR_OFFSET_Y],
            # Bottom-Left
            [MAP_W - 277 + BL_OFFSET_X, 277 + BL_OFFSET_Y]
        ], dtype=np.float32)
    else:
        return None
        
    H, _ = cv2.findHomography(marker_corners_f, map_marker_pts)
    last_marker_id = marker_id
    return H


def update_averaged_homography(H):
    """Adds a new homography to history and calculates the average."""
    global h_matrix_history, averaged_h_matrix
    if H is not None:
        h_matrix_history.append(H)
        if len(h_matrix_history) > MAX_H_HISTORY:
            h_matrix_history.pop(0)
        averaged_h_matrix = np.mean(h_matrix_history, axis=0)
# -----------------------------
# CAMERA + ARUCO + MEDIAPIPE
# -----------------------------
cap = cv2.VideoCapture(1)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
PARAMS = aruco.DetectorParameters()
PARAMS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# MODIFIED: Increase max_num_hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Homography
h_matrix_history = []
MAX_H_HISTORY = 5
averaged_h_matrix = None
last_marker_id = None
frames_since_last_detection = 0

# NEW: Manual Offset for Homography
BL_OFFSET_X = 0
BL_OFFSET_Y = 0

TL_OFFSET_X = 0
TL_OFFSET_Y = 0

BR_OFFSET_X = 0
BR_OFFSET_Y = 0


# Finger smoothing
MAX_POS_HISTORY = 5
TARGET_DISPLAY_HEIGHT = 720

# NEW: State for multiple hands
# Key: hand_index (0 to MAX_HANDS-1), Value: {'pos_history': list, 'last_color': str, 'channel': Channel}
hand_states = {} 

last_animal_status = {}  # key: marker_id, value: "correct", "wrong", or None
last_animal_position = {}

hands_present = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use MAP_IMAGE, which is updated inside calculate_homography
    map_display = MAP_IMAGE.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- ArUco detection & Filtering
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)
    
    MIN_MARKER_SIZE_PX = 30
    
    filtered_corners = []
    filtered_ids = []
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0] 
            top_side_len = np.linalg.norm(marker_corners[0] - marker_corners[1])
            left_side_len = np.linalg.norm(marker_corners[0] - marker_corners[3])
            min_side_len = min(top_side_len, left_side_len)
            
            if min_side_len >= MIN_MARKER_SIZE_PX:
                filtered_corners.append(corners[i])
                filtered_ids.append(marker_id)
            
    ids = np.array(filtered_ids).reshape(-1, 1) if filtered_ids else None
    corners = tuple(filtered_corners)
    
    current_h_matrix = None
    HOMOGRAPHY_MARKER_IDS = [3, 4, 9, 11]
    
    # NEW: Variables to store the currently detected map marker (for key handler use)
    map_marker_present = False
    target_map_marker_id = None
    target_map_marker_corners = None
    
    # Check for homography update condition
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in HOMOGRAPHY_MARKER_IDS:
                map_marker_present = True
                target_map_marker_id = marker_id
                target_map_marker_corners = corners[i][0]

                # Update if NEW map marker ID is detected OR homography is missing (e.g., manual reset)
                if last_marker_id is None or last_marker_id != marker_id or averaged_h_matrix is None:
                    # RECALCULATE HOMOGRAPHY
                    current_h_matrix = calculate_homography(
                        marker_id, 
                        target_map_marker_corners, 
                        TL_OFFSET_X, TL_OFFSET_Y, 
                        BR_OFFSET_X, BR_OFFSET_Y, 
                        BL_OFFSET_X, BL_OFFSET_Y
                    )
                    update_averaged_homography(current_h_matrix)
                    print("Homography recalculated due to new/missing marker.")
                
                break # Process only the first detected map marker

    # ---- Visualization (Draw Markers)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].astype(int)[0] 
            cv2.polylines(frame, [marker_corners], isClosed=True, color=(0, 255, 0), thickness=2)
            corner_tl = marker_corners[0]
            cv2.putText(frame, str(marker_id), (corner_tl[0], corner_tl[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    
    # ---- Draw Map Border
    H_status = "Status: Calculating..."
    H_color = (0, 165, 255) # Orange for warning/awaiting

    if averaged_h_matrix is not None:
        # Use the global MAP_W and MAP_H which are updated in calculate_homography
        map_corners = np.float32([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]]).reshape(-1, 1, 2)
        # Use try-except for robust inverse calculation
        try:
            H_inverse = cv2.invert(averaged_h_matrix)[1] 
            camera_corners = cv2.perspectiveTransform(map_corners, H_inverse)
            cv2.polylines(frame, [np.int32(camera_corners)], isClosed=True, color=(255, 0, 0), thickness=3)
            H_status = f"Status: OK (Map {ACTIVE_MAP_ID})"
            H_color = (0, 255, 0) # Green
        except cv2.error as e:
            # Handle case where homography might be singular 
            print(f"Error in perspectiveTransform or cv2.invert: {e}")
            reset_homography()
            H_status = "Status: ERROR (Singular H)"
            H_color = (0, 0, 255) # Red
    else:
        H_status = "Status: Awaiting Marker"
        H_color = (0, 165, 255) # Orange
        
    # --- DEBUGGING OUTPUT ---
    # Display the Homography Status
    cv2.putText(frame, H_status, (50, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, H_color, 2)
    # Display the Marker Detection Status
    cv2.putText(frame, f"Marker Found: {'Yes' if map_marker_present else 'No'} (ID: {target_map_marker_id})", 
                (50, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # --- END DEBUGGING OUTPUT ---


    # ---- Finger detection 
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# ... (rest of the finger detection logic is unchanged) ...

# ... (rest of the animal detection logic is unchanged) ...

    # ---- Display camera + map
    # MAP_W and MAP_H must be the correct global variables here
    if MAP_H > 0 and MAP_W > 0:
        frame_resized = cv2.resize(frame, (int(frame.shape[1]*(TARGET_DISPLAY_HEIGHT/frame.shape[0])), TARGET_DISPLAY_HEIGHT))
        map_resized = cv2.resize(map_display, (int(MAP_W*(TARGET_DISPLAY_HEIGHT/MAP_H)), TARGET_DISPLAY_HEIGHT))
        combined = np.hstack((frame_resized, map_resized))
        cv2.imshow("Camera + Map", combined)
    else:
        # Fallback in case of zero dimensions
        cv2.imshow("Camera", frame)
        print("Warning: MAP_H or MAP_W is zero/invalid. Cannot display map.")


    # ---- Key Press Handling (MODIFIED FOR FASTER UPDATE)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC to quit
        break
        
    # Manual Homography Refresh (Key 'r' or 'R')
    if key == ord('r') or key == ord('R'):
        reset_homography()

    # --- Offset Adjustment Logic ---
    updated = False # Flag to track if any offset was changed
    
    # Manual Bottom-Left Corner Adjustment (Keys: W/A/S/D)
    if key == ord('a'): # Left
        BL_OFFSET_X -= 1
        updated = True
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('d'): # Right
        BL_OFFSET_X += 1
        updated = True
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('s'): # Down (increases Y coordinate)
        BL_OFFSET_Y += 1
        updated = True
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('w'): # Up (decreases Y coordinate)
        BL_OFFSET_Y -= 1
        updated = True
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")

    # NEW: Manual Top-Left Corner Adjustment (Keys: T/F/G/H)
    elif key == ord('f'): # Left
        TL_OFFSET_X -= 1
        updated = True
        print(f"TL_OFFSET: ({TL_OFFSET_X}, {TL_OFFSET_Y})")
    elif key == ord('h'): # Right
        TL_OFFSET_X += 1
        updated = True
        print(f"TL_OFFSET: ({TL_OFFSET_X}, {TL_OFFSET_Y})")
    elif key == ord('g'): # Down
        TL_OFFSET_Y += 1
        updated = True
        print(f"TL_OFFSET: ({TL_OFFSET_X}, {TL_OFFSET_Y})")
    elif key == ord('t'): # Up
        TL_OFFSET_Y -= 1
        updated = True
        print(f"TL_OFFSET: ({TL_OFFSET_X}, {TL_OFFSET_Y})")

    # NEW: Manual Bottom-Right Corner Adjustment (Keys: I/J/K/L)
    elif key == ord('j'): # Left
        BR_OFFSET_X -= 1
        updated = True
        print(f"BR_OFFSET: ({BR_OFFSET_X}, {BR_OFFSET_Y})")
    elif key == ord('l'): # Right
        BR_OFFSET_X += 1
        updated = True
        print(f"BR_OFFSET: ({BR_OFFSET_X}, {BR_OFFSET_Y})")
    elif key == ord('k'): # Down
        BR_OFFSET_Y += 1
        updated = True
        print(f"BR_OFFSET: ({BR_OFFSET_X}, {BR_OFFSET_Y})")
    elif key == ord('i'): # Up
        BR_OFFSET_Y -= 1
        updated = True
        print(f"BR_OFFSET: ({BR_OFFSET_X}, {BR_OFFSET_Y})")

    # --- IMMEDIATE RECALCULATION IF OFFSET CHANGED ---
    if updated and map_marker_present:
        start_time = time.time() # Start timer
        
        # 1. Calculate a NEW homography with the adjusted offsets
        new_h = calculate_homography(
            target_map_marker_id, 
            target_map_marker_corners, 
            TL_OFFSET_X, TL_OFFSET_Y, 
            BR_OFFSET_X, BR_OFFSET_Y, 
            BL_OFFSET_X, BL_OFFSET_Y
        )
        
        # 2. Reset the history and replace the averaged matrix immediately
        if new_h is not None:
            # FIX: Removed the incorrect global keyword usage here.
            h_matrix_history = [new_h] * MAX_H_HISTORY
            averaged_h_matrix = new_h 
            
            end_time = time.time() # End timer
            print(f"Averaged Homography updated immediately with new offset. Took {end_time - start_time:.4f}s.")
        else:
            print("Could not recalculate Homography: Marker not visible or calculation failed.")

    # Reset ALL offsets
    elif key == ord('z'): 
        BL_OFFSET_X = 0
        BL_OFFSET_Y = 0
        TL_OFFSET_X = 0
        TL_OFFSET_Y = 0
        BR_OFFSET_X = 0
        BR_OFFSET_Y = 0
        # Only reset homography state if a marker is present to trigger immediate update
        if map_marker_present:
            reset_homography()
        print("All Offsets Reset. Homography reset will follow if marker is present.")


cap.release()
cv2.destroyAllWindows()