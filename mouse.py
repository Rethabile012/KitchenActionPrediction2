from pynput.mouse import Button, Controller
import time

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print("Clicked to keep Colab alive")  # Optional: For logging
    time.sleep(30)  # Adjust interval as needed (e.g., 60 for every minute)