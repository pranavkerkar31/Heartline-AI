"""
Configuration Examples for Smart Camera
This file shows different configurations for various use cases
"""

from smart_camera import SmartCamera
from advanced_smart_camera import AdvancedSmartCamera


def document_scanner_mode():
    """
    Configuration for document scanning
    - Larger ROI for A4/Letter size documents
    """
    print("Starting Document Scanner Mode...")
    camera = SmartCamera(
        roi_percentage=0.8,  # Larger frame for documents
        camera_index=0
    )
    camera.run()


def id_card_capture_mode():
    """
    Configuration for ID card or credit card capture
    - Smaller, centered ROI
    """
    print("Starting ID Card Capture Mode...")
    camera = SmartCamera(
        roi_percentage=0.5,  # Smaller frame for cards
        camera_index=0
    )
    camera.run()


def product_photography_mode():
    """
    Configuration for product photography
    - Medium ROI with validation
    """
    print("Starting Product Photography Mode...")
    camera = AdvancedSmartCamera(
        roi_percentage=0.65,
        camera_index=0,
        validation_threshold=0.03  # Higher threshold for product detection
    )
    camera.run()


def quality_inspection_mode():
    """
    Configuration for quality inspection/defect detection
    - Precise ROI with high sensitivity validation
    """
    print("Starting Quality Inspection Mode...")
    camera = AdvancedSmartCamera(
        roi_percentage=0.6,
        camera_index=0,
        validation_threshold=0.015  # Lower threshold for sensitive detection
    )
    camera.run()


def full_frame_guide_mode():
    """
    Configuration with full frame ROI
    - Just corner markers, minimal overlay
    """
    print("Starting Full Frame Guide Mode...")
    camera = SmartCamera(
        roi_percentage=0.95,  # Almost full frame
        camera_index=0
    )
    camera.run()


def custom_configuration():
    """
    Example of custom configuration
    """
    print("Starting Custom Configuration...")
    
    # Create camera instance
    camera = AdvancedSmartCamera(
        roi_percentage=0.7,
        camera_index=0,
        validation_threshold=0.02
    )
    
    # Customize output directory
    camera.output_dir = "my_custom_images"
    
    # Run camera
    camera.run()


if __name__ == "__main__":
    print("=== Smart Camera Configuration Examples ===\n")
    print("Choose a mode:")
    print("1. Document Scanner")
    print("2. ID Card Capture")
    print("3. Product Photography")
    print("4. Quality Inspection")
    print("5. Full Frame Guide")
    print("6. Custom Configuration")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    modes = {
        "1": document_scanner_mode,
        "2": id_card_capture_mode,
        "3": product_photography_mode,
        "4": quality_inspection_mode,
        "5": full_frame_guide_mode,
        "6": custom_configuration
    }
    
    if choice in modes:
        modes[choice]()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice. Please run again and select a valid option.")
