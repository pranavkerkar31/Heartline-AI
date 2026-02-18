"""
Smart Camera with Region of Interest (ROI) Highlighting
This script displays a camera feed with a highlighted ROI frame where users
should position their object before capturing.
"""

import cv2
import numpy as np
import os
from datetime import datetime


class SmartCamera:
    def __init__(self, roi_percentage=0.7, camera_index=0):
        """
        Initialize Smart Camera
        
        Args:
            roi_percentage: Size of ROI relative to frame (0.0 to 1.0)
            camera_index: Camera device index (default 0 for primary camera)
        """
        self.roi_percentage = roi_percentage
        self.camera_index = camera_index
        self.cap = None
        self.roi_rect = None
        self.frame_width = None
        self.frame_height = None
        self.output_dir = "captured_images"
        self.capture_count = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def initialize_camera(self):
        """Initialize camera and get frame dimensions"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate ROI rectangle coordinates
        roi_width = int(self.frame_width * self.roi_percentage)
        roi_height = int(self.frame_height * self.roi_percentage)
        
        x1 = (self.frame_width - roi_width) // 2
        y1 = (self.frame_height - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        
        self.roi_rect = (x1, y1, x2, y2)
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        print(f"ROI: ({x1}, {y1}) to ({x2}, {y2})")
    
    def draw_roi_frame(self, frame):
        """
        Draw the ROI frame and overlay on the camera feed
        
        Args:
            frame: Current camera frame
            
        Returns:
            frame: Frame with ROI overlay
        """
        x1, y1, x2, y2 = self.roi_rect
        
        # Create semi-transparent overlay for areas outside ROI
        overlay = frame.copy()
        
        # Darken areas outside ROI
        cv2.rectangle(overlay, (0, 0), (self.frame_width, y1), (0, 0, 0), -1)  # Top
        cv2.rectangle(overlay, (0, y2), (self.frame_width, self.frame_height), (0, 0, 0), -1)  # Bottom
        cv2.rectangle(overlay, (0, y1), (x1, y2), (0, 0, 0), -1)  # Left
        cv2.rectangle(overlay, (x2, y1), (self.frame_width, y2), (0, 0, 0), -1)  # Right
        
        # Blend overlay with original frame
        alpha = 0.5
        frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)
        
        # Draw ROI rectangle with thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw corner markers for better visibility
        corner_length = 30
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), (0, 255, 0), corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), (0, 255, 0), corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), (0, 255, 0), corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), (0, 255, 0), corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), (0, 255, 0), corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), (0, 255, 0), corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), (0, 255, 0), corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), (0, 255, 0), corner_thickness)
        
        return frame
    
    def add_instructions(self, frame):
        """
        Add instruction text to the frame
        
        Args:
            frame: Current camera frame
            
        Returns:
            frame: Frame with instructions
        """
        # Add semi-transparent background for better text visibility
        text_bg_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, text_bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add instruction text
        instructions = [
            "Place object within the GREEN FRAME",
            "Press SPACE to capture | Press 'Q' to quit",
            f"Captured: {self.capture_count} images"
        ]
        
        y_offset = 25
        for i, text in enumerate(instructions):
            y_position = y_offset + (i * 30)
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def check_roi_content(self, frame):
        """
        Check if there's sufficient content in ROI (optional validation)
        
        Args:
            frame: Current camera frame
            
        Returns:
            bool: True if ROI has sufficient content
        """
        x1, y1, x2, y2 = self.roi_rect
        roi = frame[y1:y2, x1:x2]
        
        # Calculate edge density in ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Return True if edge density is above threshold (indicating object presence)
        return edge_density > 0.02
    
    def capture_image(self, frame):
        """
        Capture and save the current frame
        
        Args:
            frame: Frame to capture
            
        Returns:
            str: Path to saved image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{self.capture_count:04d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the original frame (without overlays)
        cv2.imwrite(filepath, frame)
        self.capture_count += 1
        
        print(f"Image captured: {filename}")
        return filepath
    
    def show_capture_feedback(self, frame):
        """
        Show visual feedback when image is captured
        
        Args:
            frame: Current frame
            
        Returns:
            frame: Frame with feedback overlay
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), 
                     (255, 255, 255), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Add "CAPTURED!" text
        text = "CAPTURED!"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 2
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                   (0, 255, 0), thickness)
        
        return frame
    
    def run(self):
        """Main loop to run the smart camera"""
        try:
            self.initialize_camera()
            
            print("\n=== Smart Camera Started ===")
            print("Instructions:")
            print("  - Position your object within the green frame")
            print("  - Press SPACE to capture image")
            print("  - Press 'Q' to quit\n")
            
            capture_feedback_frames = 0
            original_frame = None
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                # Store original frame for capture
                original_frame = frame.copy()
                
                # Draw ROI frame
                display_frame = self.draw_roi_frame(frame)
                
                # Add instructions
                display_frame = self.add_instructions(display_frame)
                
                # Show capture feedback if recently captured
                if capture_feedback_frames > 0:
                    display_frame = self.show_capture_feedback(display_frame)
                    capture_feedback_frames -= 1
                
                # Display the frame
                cv2.imshow('Smart Camera', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("Quitting...")
                    break
                elif key == ord(' '):  # Space bar
                    # Capture image
                    self.capture_image(original_frame)
                    capture_feedback_frames = 15  # Show feedback for ~15 frames
                
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal images captured: {self.capture_count}")
        print("Camera released and windows closed.")


def main():
    """Main entry point"""
    # Initialize smart camera with 70% ROI size
    camera = SmartCamera(roi_percentage=0.7, camera_index=0)
    
    # Run the camera
    camera.run()


if __name__ == "__main__":
    main()
