"""
Advanced Smart Camera with Object Presence Validation
This version includes validation to ensure an object is present in the ROI before allowing capture.
"""

import cv2
import numpy as np
import os
from datetime import datetime


class AdvancedSmartCamera:
    def __init__(self, roi_percentage=0.7, camera_index=0, validation_threshold=0.02):
        """
        Initialize Advanced Smart Camera with object detection
        
        Args:
            roi_percentage: Size of ROI relative to frame (0.0 to 1.0)
            camera_index: Camera device index
            validation_threshold: Edge density threshold for object detection
        """
        self.roi_percentage = roi_percentage
        self.camera_index = camera_index
        self.validation_threshold = validation_threshold
        self.cap = None
        self.roi_rect = None
        self.frame_width = None
        self.frame_height = None
        self.output_dir = "captured_images"
        self.capture_count = 0
        self.object_detected = False
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def initialize_camera(self):
        """Initialize camera and get frame dimensions"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        roi_width = int(self.frame_width * self.roi_percentage)
        roi_height = int(self.frame_height * self.roi_percentage)
        
        x1 = (self.frame_width - roi_width) // 2
        y1 = (self.frame_height - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        
        self.roi_rect = (x1, y1, x2, y2)
    
    def detect_object_in_roi(self, frame):
        """
        Detect if object is present in ROI using edge detection and motion
        
        Args:
            frame: Current camera frame
            
        Returns:
            bool: True if object detected, False otherwise
            float: Confidence score
        """
        x1, y1, x2, y2 = self.roi_rect
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate variance (higher variance = more detail/object present)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combined score
        confidence = (edge_density * 100 + variance) / 2
        
        # Object detected if confidence above threshold
        detected = edge_density > self.validation_threshold and variance > 50
        
        return detected, confidence
    
    def draw_roi_frame(self, frame, object_detected):
        """
        Draw ROI frame with color based on object detection
        
        Args:
            frame: Current camera frame
            object_detected: Whether object is detected in ROI
            
        Returns:
            frame: Frame with ROI overlay
        """
        x1, y1, x2, y2 = self.roi_rect
        
        # Choose color based on detection
        if object_detected:
            roi_color = (0, 255, 0)  # Green when object detected
            overlay_alpha = 0.3
        else:
            roi_color = (0, 165, 255)  # Orange when no object
            overlay_alpha = 0.5
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, y1), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, y2), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, y1), (x1, y2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x2, y1), (self.frame_width, y2), (0, 0, 0), -1)
        
        frame = cv2.addWeighted(frame, 1, overlay, overlay_alpha, 0)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)
        
        # Draw corner markers
        corner_length = 30
        corner_thickness = 4
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), roi_color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), roi_color, corner_thickness)
        
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), roi_color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), roi_color, corner_thickness)
        
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), roi_color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), roi_color, corner_thickness)
        
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), roi_color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), roi_color, corner_thickness)
        
        return frame
    
    def add_status_overlay(self, frame, object_detected, confidence):
        """
        Add status information overlay
        
        Args:
            frame: Current camera frame
            object_detected: Whether object is detected
            confidence: Detection confidence score
            
        Returns:
            frame: Frame with status overlay
        """
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Status text
        if object_detected:
            status = "READY TO CAPTURE"
            status_color = (0, 255, 0)
        else:
            status = "Position object in frame"
            status_color = (0, 165, 255)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Instructions
        instructions = [
            f"Confidence: {confidence:.1f}%",
            "SPACE: Capture | 'Q': Quit",
            f"Captured: {self.capture_count}"
        ]
        
        y_offset = 60
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def capture_image(self, frame):
        """Capture and save image with ROI highlighted"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full frame
        full_filename = f"capture_full_{timestamp}_{self.capture_count:04d}.jpg"
        full_path = os.path.join(self.output_dir, full_filename)
        cv2.imwrite(full_path, frame)
        
        # Extract and save ROI only
        x1, y1, x2, y2 = self.roi_rect
        roi = frame[y1:y2, x1:x2]
        roi_filename = f"capture_roi_{timestamp}_{self.capture_count:04d}.jpg"
        roi_path = os.path.join(self.output_dir, roi_filename)
        cv2.imwrite(roi_path, roi)
        
        self.capture_count += 1
        print(f"Images captured: {full_filename} and {roi_filename}")
        
        return full_path, roi_path
    
    def show_capture_feedback(self, frame):
        """Show capture feedback animation"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), 
                     (255, 255, 255), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        text = "CAPTURED!"
        font = cv2.FONT_HERSHEY_BOLD
        text_size = cv2.getTextSize(text, font, 2, 4)[0]
        text_x = (self.frame_width - text_size[0]) // 2
        text_y = (self.frame_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 2, (0, 255, 0), 4)
        
        return frame
    
    def run(self):
        """Main camera loop with object detection"""
        try:
            self.initialize_camera()
            
            print("\n=== Advanced Smart Camera Started ===")
            print("Features:")
            print("  - Automatic object detection in ROI")
            print("  - Green frame = Object detected, ready to capture")
            print("  - Orange frame = Position object in frame")
            print("  - Saves both full frame and ROI crop")
            print("\nControls:")
            print("  - SPACE: Capture (when object detected)")
            print("  - 'Q': Quit\n")
            
            capture_feedback_frames = 0
            original_frame = None
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                original_frame = frame.copy()
                
                # Detect object in ROI
                object_detected, confidence = self.detect_object_in_roi(frame)
                self.object_detected = object_detected
                
                # Draw ROI frame
                display_frame = self.draw_roi_frame(frame, object_detected)
                
                # Add status overlay
                display_frame = self.add_status_overlay(display_frame, object_detected, confidence)
                
                # Show capture feedback
                if capture_feedback_frames > 0:
                    display_frame = self.show_capture_feedback(display_frame)
                    capture_feedback_frames -= 1
                
                cv2.imshow('Advanced Smart Camera', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):
                    if object_detected:
                        self.capture_image(original_frame)
                        capture_feedback_frames = 15
                    else:
                        print("Warning: No object detected in ROI. Please position object properly.")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal images captured: {self.capture_count}")


def main():
    # Initialize with object detection validation
    camera = AdvancedSmartCamera(
        roi_percentage=0.7,
        camera_index=0,
        validation_threshold=0.02
    )
    
    camera.run()


if __name__ == "__main__":
    main()
