import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from Pop import Pilot, Camera, IMU  # AutoCar III G Libraries

class AutoCarBrain:
    def __init__(self):
        # 1. Hardware Initialization
        self.car = Pilot() 
        self.cam = Camera(width=400, height=300)
        self.imu = IMU()
        
        # 2. Load Specialized Models for Status 3
        print("Loading models onto NVIDIA Brain Board...")
        self.model_left = load_model('left_model.h5')
        self.model_right = load_model('right_model.h5')
        
        # 3. FSM State Management
        self.status = 1  # 1: Straight, 2: Decision Point (Intersection), 3: Turn Execution
        self.loop_count = 0
        self.start_heading = self.imu.get_angle()[2] # Initial "Forward"
        
    def apply_warp(self, frame):
        """Applies Bird's Eye Warp to fix perspective on gray track."""
        src = np.float32([[100, 120], [300, 120], [400, 270], [0, 270]])
        dst = np.float32([[0, 0], [400, 0], [400, 150], [0, 150]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, matrix, (400, 150))

    def drive(self):
        try:
            while True:
                frame = self.cam.value
                if frame is None: continue
                
                # Pre-processing
                warped = self.apply_warp(frame)
                input_data = np.expand_dims(warped, axis=0)
                current_heading = self.imu.get_angle()[2]

                # --- FSM LOGIC ---
                
                # STATUS 1: STRAIGHT (Standard Line Following)
                if self.status == 1:
                    # Simple centering logic or a basic straight model
                    # If heading drifts, force serial servomotor to 0
                    self.car.steering = 0 
                    self.car.speed = 50
                    
                    # Transition Trigger: Detect wide intersection branching
                    # (Simplified: if camera sees lines diverging significantly)
                    if self.detect_intersection(warped):
                        self.status = 2

                # STATUS 2: DECISION POINT
                elif self.status == 2:
                    print(f"Intersection Found! Loop sequence: {self.loop_count}")
                    # Brief slow-down to allow the servomotor to adjust
                    self.car.speed = 20
                    self.status = 3 # Move to turn execution

                # STATUS 3: SPECIALIZED MODEL EXECUTION
                elif self.status == 3:
                    # Select model based on even/odd loop count for Digital 8
                    if self.loop_count % 2 == 0:
                        prediction = self.model_left.predict(input_data, verbose=0)[0]
                    else:
                        prediction = self.model_right.predict(input_data, verbose=0)[0]
                    
                    # Apply steering from model (De-normalized)
                    self.car.steering = prediction[0] * 400
                    self.car.speed = 30 # Maintain constant turn speed
                    
                    # EXIT CONDITION: Use IMU to detect completion (~180 or 360 degrees)
                    relative_yaw = abs(current_heading - self.start_heading)
                    if relative_yaw > 110: # Threshold for exiting the intersection area
                        print("Turn complete. Returning to Straight.")
                        self.status = 1
                        self.loop_count += 1
                        # Reset heading for the next straightaway
                        self.start_heading = current_heading 

        except KeyboardInterrupt:
            self.car.stop()

    def detect_intersection(self, warped):
        """Checks for the branching lines shown in your photos."""
        # Check pixel density in the top half of the warped ROI
        # If white pixels are found on both far-left and far-right, it's a crossroad
        return np.sum(warped[0:50, :]) > threshold_value # Simplified trigger

if __name__ == "__main__":
    brain = AutoCarBrain()
    brain.drive()