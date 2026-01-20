"""
    Real-time homography estimation demo using deep-matching-wrapper.
    Adapted from XFeat CVPR 2024 demo.
"""

import cv2
import numpy as np
import torch
import os
import sys
from time import time, sleep
import argparse
import threading
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from matcher.base_matcher import get_matcher, AVAILABLE_MATCHERS

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
    parser.add_argument('--matcher', type=str, choices=AVAILABLE_MATCHERS, default='xfeat', help='Local feature matcher to use.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu).')
    return parser.parse_args()


class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
                break
            self.frame = frame
            sleep(0.01)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

    def get_last_frame(self):
        return self.frame


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        # Use V4L2 backend explicitly to avoid GStreamer warnings
        # Try to open with V4L2 first, fallback to default if unavailable
        try:
            self.cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        except:
            self.cap = cv2.VideoCapture(args.cam)
            
        self.width = args.width
        self.height = args.height
        
        # Camera setup
        self.setup_camera()
        
        self.ref_frame = None
        self.corners = [[50, 50], [self.width-50, 50], [self.width-50, self.height-50], [50, self.height-50]]
        self.current_frame = None
        self.H = None

        # Init frame grabber thread
        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()

        # Matcher init
        print(f"Initializing matcher: {args.matcher}...")
        self.matcher = get_matcher(args.matcher, device=args.device)
        
        # Homography params
        self.min_inliers = 15
        self.ransac_thr = 4.0

        # FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 # avg FPS over this number of frames

        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.line_type = cv2.LINE_AA
        self.line_color = (0, 255, 0)
        self.line_thickness = 2

        self.window_name = f"Real-time {args.matcher} - Press 's' to set ref, 'q' to quit."

        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        # Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def setup_camera(self):
        # Check if camera is available first
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.args.cam}")
            print("Possible solutions:")
            print("  1. Check if camera is connected and accessible")
            print("  2. Try a different camera index (--cam 1, --cam 2, etc.)")
            print("  3. Check camera permissions: ls -l /dev/video*")
            print("  4. Install v4l2: sudo apt install v4l-utils")
            sys.exit(1)
        
        # Set camera backend to V4L2 to avoid GStreamer warnings
        # Note: This needs to be set before opening the camera
        # so we'll check if we can get frames properly
        
        # Try to set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test if we can read a frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print(f"Error: Camera {self.args.cam} opened but cannot read frames")
            print(f"Actual resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print("Try using default resolution instead of --width/--height")
            self.cap.release()
            sys.exit(1)

    def draw_quad(self, frame, point_list):
        if len(point_list) > 1:
            for i in range(len(point_list) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(point_list) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        if H is None: return []
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        try:
            warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
            warped_points_np[:, 0] += x_offset
            warped_points = warped_points_np.astype(int).tolist()
            return warped_points
        except:
            return []

    def create_top_frame(self):
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 4)
        
        # Adding captions
        self.putText(canvas=top_frame, text="Reference Frame (Press 's' to update)", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(255,255,255), borderColor=(0,0,0), thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame, text="Current Frame", org=(self.width + 10, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(255,255,255), borderColor=(0,0,0), thickness=1, lineType=self.line_type)
        
        # Draw original corners on ref frame
        self.draw_quad(top_frame, self.corners)
        
        # Draw warped corners on current frame
        if self.H is not None and len(self.corners) == 4:
            warped = self.warp_points(self.corners, self.H, self.width)
            if warped:
                 self.draw_quad(top_frame, warped)
        
        return top_frame

    def match_and_draw(self):
        # Run matcher
        # Pass numpy arrays, BaseMatcher now handles them
        res = self.matcher(self.ref_frame, self.current_frame)
        
        points1 = res['matched_kpts0']
        points2 = res['matched_kpts1']
        inlier_mask = np.ones(len(points1), dtype=bool) # Default if no H found
        
        self.H = res['H']
        num_inliers = res['num_inliers']
        
        inlier_pts1 = res['inlier_kpts0']
        inlier_pts2 = res['inlier_kpts1']
        
        # Prepare for drawMatches
        kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in inlier_pts1]
        kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in inlier_pts2]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

        # If it's a dense matcher, we might want to sample if there are too many points
        if len(kp1) > 500:
            indices = np.random.choice(len(kp1), 500, replace=False)
            kp1 = [kp1[i] for i in indices]
            kp2 = [kp2[i] for i in indices]
            matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

        matched_frame = cv2.drawMatches(self.ref_frame, kp1, self.current_frame, kp2, matches, None, 
                                      matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        color = (240, 89, 169)
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 4)

        # Labels
        self.putText(canvas=matched_frame, text=f"{self.args.matcher} Inliers: {num_inliers}", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(255,255,255), borderColor=(0,0,0), thickness=1, lineType=self.line_type)
        
        self.putText(canvas=matched_frame, text="FPS: {:.1f}".format(self.FPS), org=(self.width + 10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(255,255,255), borderColor=(0,0,0), thickness=1, lineType=self.line_type)

        return matched_frame

    def main_loop(self):
        # Warm up
        frame = self.frame_grabber.get_last_frame()
        while frame is None:
            frame = self.frame_grabber.get_last_frame()
            sleep(0.1)
            
        self.current_frame = frame
        self.ref_frame = self.current_frame.copy()

        while True:
            t0 = time()
            
            # Match and create bottom frame
            bottom_frame = self.match_and_draw()
            
            # Create top frame (uses updated self.H from match_and_draw)
            top_frame = self.create_top_frame()

            # Stack and show
            canvas = np.vstack((top_frame, bottom_frame))
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()
            
            # Update frame for next iteration
            next_frame = self.frame_grabber.get_last_frame()
            if next_frame is not None:
                self.current_frame = next_frame

            # Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparser()
    demo = MatchingDemo(args)
    demo.main_loop()