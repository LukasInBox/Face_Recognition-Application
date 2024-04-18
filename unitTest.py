
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from tkinter import Tk, messagebox
from main import App  # Running main.py file for testing

class TestApp(unittest.TestCase):
    def setUp(self):
        self.root = Tk()
        self.app = App(self.root, "Unit Test")
    
    def test_initialization(self):
        """Test if the app initializes correctly with the provided title."""
        self.assertEqual(self.app.window.title(), "Unit Test")

    @patch('cv2.dnn.Net')
    @patch('cv2.VideoCapture')
    def test_face_detection(self, mock_video_capture, mock_dnn_net):
        """Test face detection handling in the video stream."""
        mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        detections = np.zeros((1, 1, 1, 7))
        detections[0, 0, 0, 2] = 1.0  # High confidence
        detections[0, 0, 0, 3:7] = np.array([0.1, 0.1, 0.9, 0.9])
        mock_dnn_net.return_value.forward.return_value = detections
        
        with patch('tkinter.Canvas.create_image') as mock_create_image:
            self.app.update()
            mock_create_image.assert_called_once()

    @patch('cv2.imwrite')
    @patch('cv2.VideoCapture')
    def test_save_snapshot(self, mock_video_capture, mock_imwrite):
        """Test saving of snapshot when a face is detected."""
        mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        self.app.save_snapshot()
        mock_imwrite.assert_called_once()

    @patch('builtins.open')
    def test_load_user_statuses(self, mock_open):
        """Test loading user statuses from file."""
        mock_open.return_value.__enter__.return_value = MagicMock(spec=['read', 'write'])
        self.app.load_user_statuses()
        mock_open.assert_called_once_with('user_status.txt', 'r')

    @patch('builtins.open')
    def test_save_user_status(self, mock_open):
        """Test saving user status to file."""
        mock_open.return_value.__enter__.return_value = MagicMock(spec=['read', 'write'])
        self.app.save_user_status('user1', 'clocked_in')
        mock_open.assert_called_once_with('user_status.txt', 'w')

    @patch('builtins.open')
    def test_log_time(self, mock_open):
        """Test logging time function."""
        mock_open.return_value.__enter__.return_value = MagicMock(spec=['read', 'write'])
        self.app.log_time('user1', 'clocked_in')
        mock_open.assert_called_once()

    def tearDown(self):
        self.root.destroy()

if __name__ == '__main__':
    unittest.main()
