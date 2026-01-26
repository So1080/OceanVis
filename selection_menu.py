import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import pygame
import os
import time
import pygame_widgets

from nicegui import ui
import csv
from datetime import datetime

# -----------------------------
# 1. SETUP & ASSET DEFINITIONS
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pygame.init()
pygame.mixer.init()

MAX_HANDS = 1 
HAND_CHANNELS = [pygame.mixer.Channel(i) for i in range(MAX_HANDS)]
ANIMAL_CHANNEL = pygame.mixer.Channel(MAX_HANDS) 

DESCRIPTIONS = {
    "Sparse live bottom": "sounds/Habitat_Legend-NarratorONLY-Sparse_Live_Bottom.wav",
    "Dense live bottom": "sounds/Habitat_Legend-NarratorONLY-Dense_Live_Bottom.wav",
    "Rippled sand": "sounds/Habitat_Legend-NarratorONLY-Rippled_Sand.wav",
    "Flat sand": "sounds/Habitat_Legend-NarratorONLY-Flat_Sand.wav",
}

SOUND_OPTIONS = {
    "Sparse live bottom": [
        "Sounds_VF/BASIC_v3/BASIC-BASIC-sparse-live-8bars.wav",
        "Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Sparse Live Bottom.wav",
        "Sounds_VF/SPEECH_US_v1_x8_REPEATS/Sparse-x8-100percent-USA_v1.aif"
    ],
    "Dense live bottom": [
        "Sounds_VF/BASIC_v3/BASIC-BASIC-dense-live-8bars-v3.wav",
        "Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Dense Live Bottom.wav",
        "Sounds_VF/SPEECH_US_v1_x8_REPEATS/Dense-x8-100percent-USA_v1.aif"
    ],
    "Flat sand": [
        "Sounds_VF/BASIC_v3/BASIC-BASIC-flat-sand-8bars-v3.wav",
        "Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Flat Sand.wav",
        "Sounds_VF/SPEECH_US_v1_x8_REPEATS/Flat-x8-100percent-USA_v1.aif"
    ],
    "Rippled sand": [
        "Sounds_VF/BASIC_v3/BASIC-BASIC-rippled-sand-8bars-v3.wav",
        "Sounds_VF/COMPLEX/HabitatSynthSounds v4 C Short Loop - Rippled Sand.wav",
        "Sounds_VF/SPEECH_US_v1_x8_REPEATS/Rippled-x8-100percent-USA_v1.aif"
    ]
}

# UPDATED: Mask mapping to match your hardware/map setup
MASK_TO_HABITAT = {
    "blue": "Sparse live bottom",
    "green": "Dense live bottom",
    "red": "Flat sand",        
    "beige": "Rippled sand"    
}

HABITATS_ORDER = ["Sparse live bottom", "Dense live bottom", "Flat sand", "Rippled sand"]
selected_indices = {h: 0 for h in HABITATS_ORDER}
FINAL_SOUNDS = {}

# Animal logic updated to match the new habitat colors
ANIMAL_HABITATS = {
    9: ["blue"],               # coral  
    4: ["blue", "red"],        # crab
    5: ["green"],              # turtle
    6: ["red", "beige"],       # ray
    7: ["blue"],               # anemone
    8: ["red", "beige"]        # shell
}

UI_COLORS = {
    "Sparse live bottom": (95, 254, 201),    # Blue
    "Dense live bottom": (0, 68, 52),     # Green
    "Flat sand": (254, 89, 84),             # Red
    "Rippled sand": (252, 229, 185)       # Beige
}

audio_elements = {} 
buttons = {}
# -----------------------------
# PHASE 1: SELECTION MENU
# -----------------------------s


def run_selection_menu():
    ui.label('Sound Selection')
    
    for current_idx in range(4):
        hab_name = HABITATS_ORDER[current_idx]
        audio_elements[hab_name] = []
        buttons[hab_name] = []
        for val in range(3):
            audio = ui.audio(SOUND_OPTIONS[hab_name][val], controls=False)
            audio_elements[hab_name].append(audio)

            button = ui.button(hab_name + " Option " + str(val+1), on_click= audio_elements[hab_name][val].play)
            buttons[hab_name].append(button)

  

    ui.run()


run_selection_menu()