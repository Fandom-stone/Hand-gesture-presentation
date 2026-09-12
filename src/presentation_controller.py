import time
import pyautogui


class PresentationController:
    def __init__(self, cooldown=0.8):
        self.cooldown = cooldown
        self.last_command_time = 0

    def execute(self, command):
        if command is None:
            return False

        current_time = time.perf_counter()

        if current_time - self.last_command_time < self.cooldown:
            return False

        if command == "NEXT":
            pyautogui.press("right")

        elif command == "PREVIOUS":
            pyautogui.press("left")

        elif command == "START":
            pyautogui.hotkey("ctrl", "f5")

        elif command == "END":
            pyautogui.press("esc")

        else:
            return False

        self.last_command_time = current_time

        return True