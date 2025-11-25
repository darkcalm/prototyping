from pygrabber.dshow_graph import FilterGraph

def list_cameras():
    try:
        graph = FilterGraph()
        devices = graph.get_input_devices()
        
        print("\nAvailable Camera Devices:")
        print("-" * 30)
        for i, device in enumerate(devices):
            print(f"Index {i}: {device}")
        print("-" * 30)
        print("\nUse the index number with the --camera argument.")
        print("Example: python yolov8_webcam.py --camera 1\n")
        
    except Exception as e:
        print(f"Error listing cameras: {e}")

if __name__ == "__main__":
    list_cameras()
