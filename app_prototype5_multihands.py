import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import pygame
import os

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

MAP_H, MAP_W, _ = MAP_IMAGE_ID4.shape

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
# HELPER
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

# NEW: Manual Offset for Homography (Bottom-Left Corner)
BL_OFFSET_X = 0
BL_OFFSET_Y = 0

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
    
    should_update_homography = False 
    
    # Check for homography update condition
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in HOMOGRAPHY_MARKER_IDS:
                # 1. Update if a NEW map marker ID is detected
                if last_marker_id is None or last_marker_id != marker_id:
                    should_update_homography = True
                    break
                # 2. Update if the SAME map marker ID is visible but homography is currently None (manual reset)
                elif averaged_h_matrix is None and last_marker_id == marker_id:
                    should_update_homography = True
                    break

    # If the conditions are met, recalculate the homography
    if should_update_homography:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in HOMOGRAPHY_MARKER_IDS:
                marker_corners_f = corners[i][0]
                
                # Update map/masks only if the marker ID changed (or if it's the first detection)
                if last_marker_id is None or last_marker_id != marker_id:
                    if marker_id in MAP_CONFIG:
                        MAP_IMAGE = MAP_CONFIG[marker_id]['image']
                        MASKS = MAP_CONFIG[marker_id]['masks']
                        ACTIVE_MAP_ID = marker_id
                        MAP_H, MAP_W, _ = MAP_IMAGE.shape
                        print(f"Map switched to ID {marker_id}.")

                # Define map_marker_pts with the manual offset applied to the bottom-left point
                if marker_id == 3:
                    map_marker_pts = np.array([
                        [MAP_W-490, 0],
                        [MAP_W, 0],
                        [MAP_W, 492],
                        [MAP_W-490 + BL_OFFSET_X, 492 + BL_OFFSET_Y] # Bottom-Left
                    ], dtype=np.float32)
                elif marker_id == 9:
                    map_marker_pts = np.array([
                        [MAP_W-490, 0],
                        [MAP_W, 0],
                        [MAP_W, 492],
                        [MAP_W-490 + BL_OFFSET_X, 492 + BL_OFFSET_Y] # Bottom-Left
                    ], dtype=np.float32)
                elif marker_id == 4:
                    map_marker_pts = np.array([
                        [MAP_W - 344, 30],
                        [MAP_W - 42, 30],
                        [MAP_W - 42, 338],
                        [MAP_W - 344 + BL_OFFSET_X, 338 + BL_OFFSET_Y] # Bottom-Left
                    ], dtype=np.float32)
                elif marker_id == 11:
                    map_marker_pts = np.array([
                        [MAP_W - 277, 0],
                        [MAP_W, 0],
                        [MAP_W, 277],
                        [MAP_W - 277 + BL_OFFSET_X, 277 + BL_OFFSET_Y] # Bottom-Left
                    ], dtype=np.float32)
                        
                current_h_matrix, _ = cv2.findHomography(marker_corners_f, map_marker_pts)
                last_marker_id = marker_id
                
                # Apply smoothing
                if current_h_matrix is not None:
                    h_matrix_history.append(current_h_matrix)
                    if len(h_matrix_history) > MAX_H_HISTORY:
                        h_matrix_history.pop(0)
                    averaged_h_matrix = np.mean(h_matrix_history, axis=0)
                
                break # Process only the first detected map marker

    # ---- Visualization (Draw Markers)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].astype(int)[0] 
            cv2.polylines(frame, [marker_corners], isClosed=True, color=(0, 255, 0), thickness=2)
            corner_tl = marker_corners[0]
            cv2.putText(frame, str(marker_id), (corner_tl[0], corner_tl[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    
    # ---- Draw Map Border
    if averaged_h_matrix is not None:
        map_corners = np.float32([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]]).reshape(-1, 1, 2)
        # Use try-except for robust inverse calculation
        try:
            H_inverse = cv2.invert(averaged_h_matrix)[1] 
            camera_corners = cv2.perspectiveTransform(map_corners, H_inverse)
            cv2.polylines(frame, [np.int32(camera_corners)], isClosed=True, color=(255, 0, 0), thickness=3)
        except cv2.error as e:
            # Handle case where homography might be singular 
            print(f"Error in perspectiveTransform or cv2.invert: {e}")
            reset_homography() 

    # ---- Finger detection 
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Identify currently tracked hands
    current_hand_indices = set(range(len(results.multi_hand_landmarks))) if results.multi_hand_landmarks else set()

    # Stop sound and clear state for hands that disappeared
    hands_to_remove = [idx for idx in hand_states.keys() if idx not in current_hand_indices]
    for idx in hands_to_remove:
        state = hand_states.pop(idx)
        state['channel'].stop()
    
    hands_present = len(current_hand_indices) > 0

    if hands_present and averaged_h_matrix is not None:
        for hand_index, handLms in enumerate(results.multi_hand_landmarks):
            if hand_index >= MAX_HANDS:
                continue 

            # Initialize state for new hand
            if hand_index not in hand_states:
                hand_states[hand_index] = {
                    'pos_history': [], 
                    'last_color': None, 
                    'channel': HAND_CHANNELS[hand_index]
                }
            
            state = hand_states[hand_index]

            hF, wF, _ = frame.shape
            lm_index_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(lm_index_tip.x * wF), int(lm_index_tip.y * hF)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            pt_transformed = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
            mx, my = int(pt_transformed[0]), int(pt_transformed[1])

            # Finger smoothing
            state['pos_history'].append(np.array([mx, my]))
            if len(state['pos_history']) > MAX_POS_HISTORY:
                state['pos_history'].pop(0)
            averaged_mx, averaged_my = np.mean(state['pos_history'], axis=0).astype(int)

            # Check mask
            detected_color = None
            if 0 <= averaged_mx < MAP_W and 0 <= averaged_my < MAP_H:
                cv2.circle(map_display, (averaged_mx, averaged_my), 50, (255, 0, 255), -1)
                for color, mask in MASKS.items():
                    if mask is not None and mask.shape[0] > averaged_my and mask.shape[1] > averaged_mx:
                        if mask[averaged_my, averaged_mx] > 0:
                            detected_color = color
                            cv2.putText(frame, f"Hand {hand_index}: {detected_color}", (50, 50 + 40 * hand_index), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(map_display, f"Hand {hand_index}: {detected_color}", (50, 50 + 40 * hand_index), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            break
                            
                # Sound playing logic
                if detected_color != state['last_color']:
                    state['channel'].stop()
                    if detected_color:
                        sound_key = None
                        if ACTIVE_MAP_ID == 2 and "caption" in detected_color:
                            sound_key = f"{detected_color}_narrator"
                        elif ACTIVE_MAP_ID != 2:
                            sound_key = detected_color
                            
                        if sound_key and sound_key in SOUNDS:
                            state['channel'].play(SOUNDS[sound_key],loops=-1)
                            
                    state['last_color'] = detected_color
            else:
                state['pos_history'] = []
                if state['last_color']:
                    state['channel'].stop()
                    state['last_color'] = None

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
    # If no hands are present, clear all hand states
    elif not hands_present:
        hands_to_clear = list(hand_states.keys())
        for idx in hands_to_clear:
            state = hand_states.pop(idx)
            state['channel'].stop()


    # ---- Animal detection 
    if ids is not None and averaged_h_matrix is not None: 
         
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in ANIMAL_HABITATS:
                c = corners[i][0]
                cx_animal, cy_animal = int(c[:,0].mean()), int(c[:,1].mean())
                pt = np.array([[[cx_animal, cy_animal]]], dtype=np.float32)
                map_pt = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
                mx_animal, my_animal = int(map_pt[0]), int(map_pt[1])
                cv2.circle(map_display, (mx_animal, my_animal), 50, (0, 255, 0), 2)

                region = check_region(mx_animal, my_animal)

                if region and "caption" not in region:
                    current_status = "correct" if region in ANIMAL_HABITATS[marker_id] else "wrong"
                    last_region = last_animal_position.get(marker_id)

                    # Play sound only if the region actually changed AND no hands are pointing
                    if not hands_present and last_region != region:
                        ANIMAL_CHANNEL.stop()
                        ANIMAL_CHANNEL.play(SOUNDS[current_status])
                        last_animal_status[marker_id] = current_status
                        last_animal_position[marker_id] = region
                else:
                    # If the marker leaves the map or undefined zone, reset once
                    if marker_id in last_animal_status:
                        last_animal_status.pop(marker_id)
                        last_animal_position.pop(marker_id)


    # ---- Display camera + map
    frame_resized = cv2.resize(frame, (int(frame.shape[1]*(TARGET_DISPLAY_HEIGHT/frame.shape[0])), TARGET_DISPLAY_HEIGHT))
    map_resized = cv2.resize(map_display, (int(MAP_W*(TARGET_DISPLAY_HEIGHT/MAP_H)), TARGET_DISPLAY_HEIGHT))
    combined = np.hstack((frame_resized, map_resized))
    cv2.imshow("Camera + Map", combined)

    # ---- Key Press Handling (MODIFIED)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC to quit
        break
        
    # Manual Homography Refresh (Key 'r' or 'R')
    if key == ord('r') or key == ord('R'):
        reset_homography()
        
    # Manual Bottom-Left Corner Adjustment (Keys: Up/Down/Left/Right/Reset)
    if key == ord('a'): # Left
        BL_OFFSET_X -= 1
        reset_homography()
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('d'): # Right
        BL_OFFSET_X += 1
        reset_homography()
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('s'): # Down (increases Y coordinate)
        BL_OFFSET_Y += 1
        reset_homography()
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('w'): # Up (decreases Y coordinate)
        BL_OFFSET_Y -= 1
        reset_homography()
        print(f"BL_OFFSET: ({BL_OFFSET_X}, {BL_OFFSET_Y})")
    elif key == ord('z'): # Reset offsets
        BL_OFFSET_X = 0
        BL_OFFSET_Y = 0
        reset_homography()
        print("BL Offsets Reset.")


cap.release()
cv2.destroyAllWindows()