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
# INIT PYGAME SOUNDS
# -----------------------------
pygame.mixer.init()
CHANNEL = pygame.mixer.Channel(0)

# Habitat colors
HABITATS = ["blue", "green", "red", "beige"]

SOUNDS = {
    "blue": pygame.mixer.Sound("sounds/ocean_blue.wav"),
    "green": pygame.mixer.Sound("sounds/ocean_darkgreen.wav"),
    "red": pygame.mixer.Sound("sounds/sand_red.wav"),
    "beige": pygame.mixer.Sound("sounds/sand_beige.wav"),

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
    3: ["blue"], #coral
    4: ["green"], #turtle
    5: ["blue", "beige"], #crab
    6: ["red", "beige"], #ray
    7: ["blue"], #anemone
    8: ["red", "beige"] #shell
}

# -----------------------------
# LOAD MAP & MASKS
# -----------------------------
# MAP_IMAGE = cv2.imread("craved_map/craved_map_color.jpeg")  # main map


MAP_IMAGE_ID10 = cv2.imread("craved_map/craved_map1.jpeg")
MAP_IMAGE_ID9 = cv2.imread("craved_map2/Frame 7_id9.png") 
MAP_IMAGE_ID8 = cv2.imread("craved_map3/Frame 8_id8.png")

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

MAP_CONFIG = {
    10: {'image': MAP_IMAGE_ID10, 'masks': MASKS_ID10},
    9: {'image': MAP_IMAGE_ID9, 'masks': MASKS_ID9},
    2: {'image': MAP_IMAGE_ID8, 'masks': MASKS_ID8},
}

ACTIVE_MAP_ID = 10
MAP_IMAGE = MAP_IMAGE_ID10
MASKS = MASKS_ID10

# -----------------------------
# HELPER
# -----------------------------
def check_region(mx, my):
    """Return the mask key at the given map coordinates."""
    for key, mask in MASKS.items():
        if mask is not None and 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]:
            if mask[my, mx] > 0:
                return key
    return None

# -----------------------------
# CAMERA + ARUCO + MEDIAPIPE
# -----------------------------
cap = cv2.VideoCapture(1)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
PARAMS = aruco.DetectorParameters()
PARAMS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Homography
h_matrix_history = []
MAX_H_HISTORY = 5
averaged_h_matrix = None
last_marker_id = None

# Finger smoothing
finger_pos_history = []
MAX_POS_HISTORY = 5
last_color = None

TARGET_DISPLAY_HEIGHT = 720

last_animal_status = {}  # key: marker_id, value: "correct", "wrong", or None
last_animal_position = {}

hands_present = False



while True:
    ret, frame = cap.read()
    if not ret:
        break

    map_display = MAP_IMAGE.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- ArUco detection
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)
    print(f"Detected marker IDs: {ids.flatten() if ids is not None else 'None'}")
    current_h_matrix = None

    HOMOGRAPHY_MARKER_IDS = [2, 9, 10]

    MIN_MARKER_SIZE_PX = 20
    
    filtered_corners = []
    filtered_ids = []
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0] # corners for the current marker
            
            # Calculate the length of the top side (from corner 0 to 1) and left side (from corner 0 to 3)
            # Corner format: [[top-left, top-right, bottom-right, bottom-left]]
            
            # Side length in pixels:
            top_side_len = np.linalg.norm(marker_corners[0] - marker_corners[1])
            left_side_len = np.linalg.norm(marker_corners[0] - marker_corners[3])
            
            # Use the smaller side length to enforce the 10x10 constraint
            min_side_len = min(top_side_len, left_side_len)
            
            if min_side_len >= MIN_MARKER_SIZE_PX:
                filtered_corners.append(corners[i])
                filtered_ids.append(marker_id)
            # else:
            #     print(f"Marker ID {marker_id} ignored (size: {min_side_len:.2f}px < {MIN_MARKER_SIZE_PX}px)")
    
    # Use the filtered results for the rest of the logic
    ids = np.array(filtered_ids).reshape(-1, 1) if filtered_ids else None
    corners = tuple(filtered_corners)
    
    # Ensure ids is None if no markers passed the filter
    if ids is not None and len(ids) == 0:
        ids = None
        
    current_h_matrix = None

    HOMOGRAPHY_MARKER_IDS = [2, 9, 10]

    if ids is not None and len(ids) > 0:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in HOMOGRAPHY_MARKER_IDS:
                marker_corners = corners[i][0]
                if last_marker_id is None or last_marker_id != marker_id:

                    if marker_id in MAP_CONFIG:
                        MAP_IMAGE = MAP_CONFIG[marker_id]['image']
                        MASKS = MAP_CONFIG[marker_id]['masks']
                        ACTIVE_MAP_ID = marker_id
                        MAP_H, MAP_W, _ = MAP_IMAGE.shape
                        # print(f"Switched active map/masks to ID {marker_id}")
                        # print(MAP_IMAGE.shape)
                        # print(MAP_W, MAP_H)
                        # print(MASKS.keys())
                        # print(ACTIVE_MAP_ID)

                    if marker_id == 2:
                        map_marker_pts = np.array([
                            [MAP_W-490, 0],
                            [MAP_W, 0],
                            [MAP_W, 492],
                            [MAP_W-490, 492]
                        ], dtype=np.float32)
                    elif marker_id == 9:
                        map_marker_pts = np.array([
                            [MAP_W-490, 0],
                            [MAP_W, 0],
                            [MAP_W, 492],
                            [MAP_W-490, 492]
                        ], dtype=np.float32)
                    elif marker_id == 10:
                        map_marker_pts = np.array([
                            [MAP_W - 277, 0],       # top-left #or 508 #or 281 #MAP_W - 281, 0
                            [MAP_W, 0],             # top-right #MAP_W, 0
                            [MAP_W, 277],         # bottom-right #MAP_W, 278
                            [MAP_W - 277, 277]            # bottom-left #MAP_W - 281, 278
                        ], dtype=np.float32)
                    current_h_matrix, _ = cv2.findHomography(marker_corners, map_marker_pts)
                    last_marker_id = marker_id
                    print(f"Homography updated for marker {marker_id}")

        if current_h_matrix is not None:
            h_matrix_history.append(current_h_matrix)
            if len(h_matrix_history) > MAX_H_HISTORY:
                h_matrix_history.pop(0)
            averaged_h_matrix = np.mean(h_matrix_history, axis=0)
    else:
        averaged_h_matrix = None
        last_marker_id = None
        h_matrix_history = []

    # ---- Finger detection
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hands_present = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0

    if hands_present and averaged_h_matrix is not None:
        for handLms in results.multi_hand_landmarks:
            hF, wF, _ = frame.shape
            lm_index_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(lm_index_tip.x * wF), int(lm_index_tip.y * hF)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            pt_transformed = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
            mx, my = int(pt_transformed[0]), int(pt_transformed[1])

            # Finger smoothing
            finger_pos_history.append(np.array([mx, my]))
            if len(finger_pos_history) > MAX_POS_HISTORY:
                finger_pos_history.pop(0)
            averaged_mx, averaged_my = np.mean(finger_pos_history, axis=0).astype(int)

            # Check mask
            detected_color = None
            if 0 <= averaged_mx < MAP_W and 0 <= averaged_my < MAP_H:
                cv2.circle(map_display, (averaged_mx, averaged_my), 30, (0, 0, 255), -1)
                for color, mask in MASKS.items():
                    if mask is not None and mask.shape[0] > averaged_my and mask.shape[1] > averaged_mx:
                        if mask[averaged_my, averaged_mx] > 0:
                            detected_color = color
                            cv2.putText(frame, detected_color, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
                            cv2.putText(map_display, detected_color, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
                            break
                if detected_color != last_color:
                    CHANNEL.stop()
                    if detected_color and ACTIVE_MAP_ID != 2:
                        if "caption" in detected_color:
                            CHANNEL.play(SOUNDS[detected_color])
                        else:
                            CHANNEL.play(SOUNDS[detected_color])
                    if detected_color and ACTIVE_MAP_ID == 2:
                        if "caption" in detected_color:
                            caption_narrator_key = f"{detected_color}_narrator"
                            CHANNEL.play(SOUNDS[caption_narrator_key])
                    last_color = detected_color
            else:
                finger_pos_history = []
                if last_color:
                    CHANNEL.stop()
                    last_color = None

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    else:
        finger_pos_history = []
        if last_color:
            CHANNEL.stop()
            last_color = None

    # ---- Animal detection ----
    if ids is not None and averaged_h_matrix is not None: 
         
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in ANIMAL_HABITATS:
                # print(f"Animal {marker_id} detected")
                c = corners[i][0]
                cx_animal, cy_animal = int(c[:,0].mean()), int(c[:,1].mean())
                pt = np.array([[[cx_animal, cy_animal]]], dtype=np.float32)
                map_pt = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
                mx_animal, my_animal = int(map_pt[0]), int(map_pt[1])
                cv2.circle(map_display, (mx_animal, my_animal), 30, (0, 255, 0), -1)

                region = check_region(mx_animal, my_animal)

                if region and "caption" not in region:
                    # print(f"Animal {marker_id} in region {region}")
                    current_status = "correct" if region in ANIMAL_HABITATS[marker_id] else "wrong"
                    last_status = last_animal_status.get(marker_id)
                    last_region = last_animal_position.get(marker_id)

                    # Play sound only if the region actually changed
                    if not hands_present and last_region != region:
                        # print(f"Animal {marker_id} in region {region}")
                        # print(f"Animal {marker_id} moved to {region} ({current_status})")
                        CHANNEL.stop()
                        CHANNEL.play(SOUNDS[current_status])
                        last_animal_status[marker_id] = current_status
                        last_animal_position[marker_id] = region
                else:
                    # If the marker leaves the map or undefined zone, reset once
                    if marker_id in last_animal_status:
                        # print(f"Animal {marker_id} left the map")
                        last_animal_status.pop(marker_id)
                        last_animal_position.pop(marker_id)


    # ---- Animal detection
    # if ids is not None and averaged_h_matrix is not None:
    #     for i, marker_id in enumerate(ids.flatten()):
    #         if marker_id in ANIMAL_HABITATS:
    #             print(f"Animal {marker_id} detected")
    #             c = corners[i][0]
    #             cx_animal, cy_animal = int(c[:,0].mean()), int(c[:,1].mean())
    #             pt = np.array([[[cx_animal, cy_animal]]], dtype=np.float32)
    #             map_pt = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
    #             mx_animal, my_animal = int(map_pt[0]), int(map_pt[1])
    #             cv2.circle(map_display, (mx_animal, my_animal), 8, (0, 255, 0), -1)

    #             region = check_region(mx_animal, my_animal)

    #             # if region and "caption" not in region:
    #             #     if region in ANIMAL_HABITATS[marker_id]:
    #             #         CHANNEL.play(SOUNDS["correct"])
    #             #     else:
    #             #         CHANNEL.play(SOUNDS["wrong"])
    #             if region and "caption" not in region:
    #                 print(f"Animal {marker_id} in region {region}")
    #                 current_status = "correct" if region in ANIMAL_HABITATS[marker_id] else "wrong"
                    
    #                 # Only play sound if this animal's status changed
    #                 if last_animal_status.get(marker_id) != current_status:
    #                     CHANNEL.stop()
    #                     CHANNEL.play(SOUNDS[current_status])
    #                     last_animal_status[marker_id] = current_status
    #             else:
    #                 # Reset when animal leaves map or is in an undefined region
    #                 last_animal_status[marker_id] = None

    # ---- Animal detection
    # if ids is not None and averaged_h_matrix is not None:   
    #     for i, marker_id in enumerate(ids.flatten()):
    #         print(f"Detected marker ID: {marker_id}")
    #         if marker_id in ANIMAL_HABITATS:
    #             print(f"Animal {marker_id} detected")
    #             c = corners[i][0]
    #             cx_animal, cy_animal = int(c[:,0].mean()), int(c[:,1].mean())
    #             pt = np.array([[[cx_animal, cy_animal]]], dtype=np.float32)
    #             map_pt = cv2.perspectiveTransform(pt, averaged_h_matrix)[0][0]
    #             mx_animal, my_animal = int(map_pt[0]), int(map_pt[1])
    #             cv2.circle(map_display, (mx_animal, my_animal), 8, (0, 255, 0), -1)

    #             region = check_region(mx_animal, my_animal)

    #             if region and "caption" not in region:
    #                 print(f"Animal {marker_id} in region {region}")
    #                 current_status = "correct" if region in ANIMAL_HABITATS[marker_id] else "wrong"

    #                 # Compute movement distance
    #                 last_pos = last_animal_position.get(marker_id)
    #                 moved_enough = False
    #                 if last_pos is not None:
    #                     dx = mx_animal - last_pos[0]
    #                     dy = my_animal - last_pos[1]
    #                     distance = (dx**2 + dy**2)**0.5
    #                     moved_enough = distance > 10
    #                 else:
    #                     moved_enough = True  # First time we see it

    #                 # Only play if moved more than 10px and status changed
    #                 if moved_enough and last_animal_status.get(marker_id) != current_status:
    #                     CHANNEL.stop()
    #                     CHANNEL.play(SOUNDS[current_status])
    #                     last_animal_status[marker_id] = current_status

    #                 # Update last position
    #                 last_animal_position[marker_id] = (mx_animal, my_animal)

    #             else:
    #                 # Reset if animal leaves map or undefined region
    #                 last_animal_status[marker_id] = None
    #                 last_animal_position[marker_id] = None



    # ---- Display camera + map
    frame_resized = cv2.resize(frame, (int(frame.shape[1]*(TARGET_DISPLAY_HEIGHT/frame.shape[0])), TARGET_DISPLAY_HEIGHT))
    map_resized = cv2.resize(map_display, (int(MAP_W*(TARGET_DISPLAY_HEIGHT/MAP_H)), TARGET_DISPLAY_HEIGHT))
    combined = np.hstack((frame_resized, map_resized))
    cv2.imshow("Camera + Map", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
