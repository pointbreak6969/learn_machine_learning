"""
Augmented Reality Shooting Game
==============================
A webcam-based AR shooting game using OpenCV and MediaPipe.

Hand Tracking Logic:
- MediaPipe Hands detects 21 hand landmarks per hand
- We track landmark 8 (INDEX_FINGER_TIP) as the gun barrel/crosshair
- We track landmark 4 (THUMB_TIP) for shooting gesture detection
- When distance between index tip and thumb tip < threshold, a "pinch" is detected
- This pinch gesture triggers a shooting action

Shooting Gesture Detection:
- Calculate Euclidean distance between thumb tip and index finger tip
- Normalize by hand size (distance from wrist to middle finger MCP) for consistency
- When normalized distance drops below threshold (~0.15), register as "shooting"
- Add cooldown to prevent rapid-fire from a single pinch

Future 3D AR Extensions:
- Use depth cameras (Intel RealSense, Azure Kinect) for Z-axis position
- Implement 3D target positioning with perspective projection
- Add ARCore/ARKit integration for mobile devices
- Use SLAM for environment mapping and persistent AR objects
"""

import cv2
import numpy as np
import random
import time
import math
import urllib.request
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

# MediaPipe imports (following gesture_recognizer pattern)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Try to import landmark_pb2 for drawing (may show IDE warning but works at runtime)
try:
    from mediapipe.framework.formats import landmark_pb2
    HAS_LANDMARK_PB2 = True
except ImportError:
    HAS_LANDMARK_PB2 = False

# Access solutions module via mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GameConfig:
    """Game configuration parameters."""
    # Window settings
    WINDOW_NAME: str = "AR Shooting Game"
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Hand detection settings
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7
    MAX_NUM_HANDS: int = 1
    
    # Shooting mechanics
    PINCH_THRESHOLD: float = 0.15  # Normalized distance for pinch detection
    SHOOT_COOLDOWN: float = 0.3    # Seconds between shots
    
    # Target settings
    TARGET_RADIUS: int = 40
    TARGET_SPAWN_INTERVAL: float = 2.0  # Seconds between spawns
    MAX_TARGETS: int = 5
    TARGET_LIFETIME: float = 5.0  # Seconds before target disappears
    
    # Visual settings
    CROSSHAIR_SIZE: int = 25
    CROSSHAIR_THICKNESS: int = 2
    HIT_FLASH_DURATION: float = 0.15  # Visual feedback duration
    
    # Colors (BGR format)
    COLOR_CROSSHAIR: Tuple[int, int, int] = (0, 255, 0)
    COLOR_CROSSHAIR_SHOOTING: Tuple[int, int, int] = (0, 0, 255)
    COLOR_TARGET: Tuple[int, int, int] = (0, 100, 255)
    COLOR_TARGET_INNER: Tuple[int, int, int] = (0, 50, 200)
    COLOR_HIT_FLASH: Tuple[int, int, int] = (255, 255, 255)
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
    COLOR_SCORE_BG: Tuple[int, int, int] = (0, 0, 0)


# ============================================================================
# TARGET CLASS
# ============================================================================

class Target:
    """Represents a shootable target in the game."""
    
    def __init__(self, x: int, y: int, radius: int, lifetime: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.spawn_time = time.time()
        self.lifetime = lifetime
        self.is_alive = True
        self.pulse_phase = random.uniform(0, 2 * math.pi)  # For animation
    
    def update(self, current_time: float) -> bool:
        """Update target state. Returns False if target should be removed."""
        age = current_time - self.spawn_time
        if age > self.lifetime:
            self.is_alive = False
        return self.is_alive
    
    def get_current_radius(self) -> int:
        """Get animated radius for pulsing effect."""
        pulse = math.sin(time.time() * 4 + self.pulse_phase) * 5
        return int(self.radius + pulse)
    
    def check_hit(self, crosshair_x: int, crosshair_y: int) -> bool:
        """Check if crosshair position hits this target."""
        distance = math.sqrt((self.x - crosshair_x) ** 2 + (self.y - crosshair_y) ** 2)
        return distance < self.radius
    
    def draw(self, frame: np.ndarray, config: GameConfig):
        """Draw the target on the frame."""
        if not self.is_alive:
            return
        
        current_radius = self.get_current_radius()
        
        # Outer ring
        cv2.circle(frame, (self.x, self.y), current_radius, config.COLOR_TARGET, 3)
        # Middle ring
        cv2.circle(frame, (self.x, self.y), current_radius // 2, config.COLOR_TARGET, 2)
        # Inner circle (bullseye)
        cv2.circle(frame, (self.x, self.y), current_radius // 4, 
                   config.COLOR_TARGET_INNER, -1)
        
        # Draw remaining time indicator
        age = time.time() - self.spawn_time
        remaining_ratio = 1 - (age / self.lifetime)
        arc_radius = current_radius + 10
        end_angle = int(360 * remaining_ratio)
        cv2.ellipse(frame, (self.x, self.y), (arc_radius, arc_radius),
                    -90, 0, end_angle, (100, 100, 100), 2)


# ============================================================================
# HAND TRACKER CLASS
# ============================================================================

class HandTracker:
    """
    Handles hand detection and landmark tracking using MediaPipe Tasks API.
    
    MediaPipe Hand Landmarks:
    - 0: WRIST
    - 4: THUMB_TIP
    - 8: INDEX_FINGER_TIP
    - 9: INDEX_FINGER_MCP (for hand size normalization)
    """
    
    # Model download URL
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    
    def __init__(self, config: GameConfig):
        self.config = config
        
        # Download model if not exists
        self._ensure_model_exists()
        
        # Initialize MediaPipe Hand Landmarker with new Tasks API
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Landmark indices
        self.INDEX_TIP = 8
        self.THUMB_TIP = 4
        self.WRIST = 0
        self.MIDDLE_MCP = 9
    
    def _ensure_model_exists(self):
        """Download the hand landmarker model if it doesn't exist."""
        if not os.path.exists(self.MODEL_PATH):
            print("Downloading hand landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("Model downloaded successfully!")
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Process a frame and extract hand landmarks.
        
        Returns:
            Dictionary with 'index_tip', 'thumb_tip', 'is_pinching', 'landmarks'
            or None if no hand detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        results = self.hand_landmarker.detect(mp_image)
        
        if not results.hand_landmarks or len(results.hand_landmarks) == 0:
            return None
        
        # Get first detected hand
        hand_landmarks = results.hand_landmarks[0]
        h, w, _ = frame.shape
        
        # Extract key landmarks (convert normalized coords to pixel coords)
        index_tip = self._get_pixel_coords(hand_landmarks[self.INDEX_TIP], w, h)
        thumb_tip = self._get_pixel_coords(hand_landmarks[self.THUMB_TIP], w, h)
        wrist = self._get_pixel_coords(hand_landmarks[self.WRIST], w, h)
        middle_mcp = self._get_pixel_coords(hand_landmarks[self.MIDDLE_MCP], w, h)
        
        # Calculate hand size for normalization (wrist to middle finger base)
        hand_size = self._calculate_distance(wrist, middle_mcp)
        
        # Calculate pinch distance (normalized by hand size)
        pinch_distance = self._calculate_distance(index_tip, thumb_tip)
        normalized_pinch = pinch_distance / hand_size if hand_size > 0 else 1.0
        
        # Detect pinch gesture
        is_pinching = normalized_pinch < self.config.PINCH_THRESHOLD
        
        return {
            'index_tip': index_tip,
            'thumb_tip': thumb_tip,
            'is_pinching': is_pinching,
            'pinch_distance': normalized_pinch,
            'landmarks': hand_landmarks
        }
    
    def _get_pixel_coords(self, landmark, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized landmark coordinates to pixel coordinates."""
        return (int(landmark.x * width), int(landmark.y * height))
    
    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks):
        """Draw hand skeleton on frame using MediaPipe drawing utilities."""
        if HAS_LANDMARK_PB2:
            # Convert landmarks to NormalizedLandmarkList proto (as in gesture_recognizer notebook)
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in landmarks
            ])
            
            # Draw using MediaPipe's drawing utilities
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            # Fallback: manual drawing
            h, w, _ = frame.shape
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
            for i, point in enumerate(points):
                color = (0, 255, 0) if i in [4, 8] else (0, 200, 200)
                radius = 6 if i in [4, 8] else 4
                cv2.circle(frame, point, radius, color, -1)
    
    def close(self):
        """Release MediaPipe resources."""
        self.hand_landmarker.close()


# ============================================================================
# GAME CLASS
# ============================================================================

class ARShootingGame:
    """Main game class that orchestrates all game logic."""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        
        # Initialize components
        self.hand_tracker = HandTracker(self.config)
        self.cap = None
        
        # Game state
        self.score = 0
        self.targets: List[Target] = []
        self.last_spawn_time = 0
        self.last_shot_time = 0
        self.is_shooting = False
        self.was_pinching = False  # For edge detection
        
        # Visual feedback
        self.hit_flash_time = 0
        self.show_hit_flash = False
        self.muzzle_flash_time = 0
        self.show_muzzle_flash = False
        
        # Crosshair position (smoothed)
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.smoothing_factor = 0.7  # Higher = smoother but more lag
        
        # FPS tracking
        self.prev_frame_time = time.time()
        self.fps = 0
    
    def initialize_camera(self) -> bool:
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        return True
    
    def spawn_target(self, frame_width: int, frame_height: int):
        """Spawn a new target at a random position."""
        margin = self.config.TARGET_RADIUS + 50
        x = random.randint(margin, frame_width - margin)
        y = random.randint(margin + 100, frame_height - margin)  # Leave space for UI
        
        target = Target(x, y, self.config.TARGET_RADIUS, self.config.TARGET_LIFETIME)
        self.targets.append(target)
    
    def update_targets(self, current_time: float):
        """Update all targets and remove expired ones."""
        self.targets = [t for t in self.targets if t.update(current_time)]
    
    def check_shooting(self, hand_data: dict, current_time: float) -> bool:
        """
        Check for shooting gesture and handle shot.
        
        Uses edge detection: only triggers on the transition from
        not-pinching to pinching (prevents continuous firing).
        """
        is_pinching = hand_data['is_pinching']
        
        # Edge detection: only shoot on pinch start
        if is_pinching and not self.was_pinching:
            # Check cooldown
            if current_time - self.last_shot_time >= self.config.SHOOT_COOLDOWN:
                self.last_shot_time = current_time
                self.was_pinching = True
                self.show_muzzle_flash = True
                self.muzzle_flash_time = current_time
                return True
        
        if not is_pinching:
            self.was_pinching = False
        
        return False
    
    def process_shot(self, crosshair_pos: Tuple[int, int], current_time: float):
        """Process a shot and check for target hits."""
        for target in self.targets[:]:  # Iterate copy to allow removal
            if target.check_hit(crosshair_pos[0], crosshair_pos[1]):
                target.is_alive = False
                self.score += 10
                self.show_hit_flash = True
                self.hit_flash_time = current_time
                # Remove hit target
                self.targets.remove(target)
                break  # Only hit one target per shot
    
    def update_crosshair(self, new_x: int, new_y: int):
        """Smooth crosshair movement to reduce jitter."""
        self.crosshair_x = int(self.crosshair_x * self.smoothing_factor + 
                                new_x * (1 - self.smoothing_factor))
        self.crosshair_y = int(self.crosshair_y * self.smoothing_factor + 
                                new_y * (1 - self.smoothing_factor))
    
    def draw_crosshair(self, frame: np.ndarray, is_shooting: bool):
        """Draw the crosshair at current position."""
        color = (self.config.COLOR_CROSSHAIR_SHOOTING if is_shooting 
                 else self.config.COLOR_CROSSHAIR)
        size = self.config.CROSSHAIR_SIZE
        thickness = self.config.CROSSHAIR_THICKNESS
        
        x, y = self.crosshair_x, self.crosshair_y
        
        # Draw crosshair lines
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
        
        # Draw center dot
        cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw outer circle
        cv2.circle(frame, (x, y), size, color, thickness)
        
        # Muzzle flash effect
        if self.show_muzzle_flash:
            cv2.circle(frame, (x, y), size + 15, (0, 200, 255), 3)
            cv2.circle(frame, (x, y), size + 25, (0, 100, 255), 2)
    
    def draw_ui(self, frame: np.ndarray, hand_detected: bool, pinch_distance: float = 0):
        """Draw game UI elements."""
        h, w = frame.shape[:2]
        
        # Score display with background
        score_text = f"SCORE: {self.score}"
        cv2.rectangle(frame, (10, 10), (200, 60), self.config.COLOR_SCORE_BG, -1)
        cv2.rectangle(frame, (10, 10), (200, 60), self.config.COLOR_TEXT, 2)
        cv2.putText(frame, score_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, self.config.COLOR_TEXT, 2)
        
        # FPS display
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(frame, fps_text, (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (150, 150, 150), 2)
        
        # Hand status
        status = "Hand Detected" if hand_detected else "No Hand Detected"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2)
        
        # Pinch indicator bar
        if hand_detected:
            bar_width = 150
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = h - 50
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Fill based on pinch distance (inverted: closer = more fill)
            fill_ratio = max(0, min(1, 1 - pinch_distance / 0.5))
            fill_width = int(bar_width * fill_ratio)
            fill_color = (0, 0, 255) if fill_ratio > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (200, 200, 200), 2)
            cv2.putText(frame, "PINCH", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
        
        # Instructions
        instructions = "Pinch thumb & index finger to SHOOT | Press 'Q' to quit"
        cv2.putText(frame, instructions, (w // 2 - 250, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Target count
        target_text = f"Targets: {len(self.targets)}"
        cv2.putText(frame, target_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)
    
    def draw_hit_effect(self, frame: np.ndarray):
        """Draw hit visual feedback."""
        if self.show_hit_flash:
            # Create flash overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                         (100, 255, 100), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw "HIT!" text
            cv2.putText(frame, "HIT!", (frame.shape[1] // 2 - 60, frame.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    
    def update_fps(self, current_time: float):
        """Calculate and update FPS."""
        delta_time = current_time - self.prev_frame_time
        self.fps = 1.0 / delta_time if delta_time > 0 else 0
        self.prev_frame_time = current_time
    
    def run(self):
        """Main game loop."""
        if not self.initialize_camera():
            return
        
        print("=" * 50)
        print("AR SHOOTING GAME")
        print("=" * 50)
        print("Controls:")
        print("  - Point with your index finger to aim")
        print("  - Pinch index finger and thumb together to shoot")
        print("  - Press 'Q' to quit")
        print("=" * 50)
        
        try:
            while True:
                current_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Update FPS
                self.update_fps(current_time)
                
                # Process hand tracking
                hand_data = self.hand_tracker.process_frame(frame)
                
                # Update visual effects timers
                if self.show_hit_flash and current_time - self.hit_flash_time > self.config.HIT_FLASH_DURATION:
                    self.show_hit_flash = False
                if self.show_muzzle_flash and current_time - self.muzzle_flash_time > 0.1:
                    self.show_muzzle_flash = False
                
                # Spawn new targets periodically
                if (current_time - self.last_spawn_time > self.config.TARGET_SPAWN_INTERVAL 
                    and len(self.targets) < self.config.MAX_TARGETS):
                    self.spawn_target(w, h)
                    self.last_spawn_time = current_time
                
                # Update targets
                self.update_targets(current_time)
                
                # Process hand input
                is_shooting = False
                pinch_distance = 0.5
                if hand_data:
                    # Update crosshair position
                    index_tip = hand_data['index_tip']
                    self.update_crosshair(index_tip[0], index_tip[1])
                    pinch_distance = hand_data['pinch_distance']
                    
                    # Check for shooting
                    if self.check_shooting(hand_data, current_time):
                        is_shooting = True
                        self.process_shot((self.crosshair_x, self.crosshair_y), current_time)
                    
                    # Draw hand landmarks
                    self.hand_tracker.draw_landmarks(frame, hand_data['landmarks'])
                
                # Draw game elements
                for target in self.targets:
                    target.draw(frame, self.config)
                
                if hand_data:
                    self.draw_crosshair(frame, is_shooting or self.show_muzzle_flash)
                
                # Draw effects
                if self.show_hit_flash:
                    self.draw_hit_effect(frame)
                
                # Draw UI
                self.draw_ui(frame, hand_data is not None, pinch_distance)
                
                # Display frame
                cv2.imshow(self.config.WINDOW_NAME, frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print(f"\nFinal Score: {self.score}")
        print("Thanks for playing!")
        
        if self.cap:
            self.cap.release()
        self.hand_tracker.close()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Entry point for the AR shooting game."""
    # Create custom config if needed
    config = GameConfig()
    
    # You can customize settings here:
    # config.TARGET_SPAWN_INTERVAL = 1.5  # Faster spawns
    # config.PINCH_THRESHOLD = 0.12       # More sensitive shooting
    # config.TARGET_LIFETIME = 3.0        # Shorter target life
    
    # Create and run game
    game = ARShootingGame(config)
    game.run()


if __name__ == "__main__":
    main()
