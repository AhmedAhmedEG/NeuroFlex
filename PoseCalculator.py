from Props import draw_pose_landmarks, draw_hand_landmarks, calc_pose_angles, calc_hand_angles, TherapyLevel, \
    draw_arm_landmarks, calc_points, calc_palm_orientation, calc_pose_orientation

from itertools import chain
from pathlib import Path
import mediapipe as mp
import pickle
import cv2
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# image = cv2.imread('Poses/Pose 3/2.jpg')
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pose_path = Path('Poses') / '9'
cap = cv2.VideoCapture(str(pose_path / '2.mp4'))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, static_image_mode=False, model_complexity=2) as holistic:
    therapy_level = TherapyLevel(right_hand=True, right_arm=True)

    while cap.isOpened():
        success, frame = cap.read()
        # success, frame = True, deepcopy(image)

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = holistic.process(frame)
        if results.left_hand_landmarks:
            left_hand_points = calc_points(results.left_hand_landmarks.landmark, w, h)
            left_hand_3d_points = calc_points(results.left_hand_landmarks.landmark, w, h, use_z=True)

            therapy_level.left_hand_angles = calc_hand_angles(left_hand_points)
            therapy_level.left_palm_orientation = calc_palm_orientation(left_hand_3d_points)

            draw_hand_landmarks(frame, left_hand_points)

        if results.right_hand_landmarks:
            right_hand_points = calc_points(results.right_hand_landmarks.landmark, w, h)
            right_hand_3d_points = calc_points(results.right_hand_landmarks.landmark, w, h, use_z=True)

            therapy_level.right_hand_angles = calc_hand_angles(right_hand_points)
            therapy_level.right_palm_orientation = calc_palm_orientation(right_hand_3d_points)

            draw_hand_landmarks(frame, right_hand_points)

        if results.pose_landmarks:
            pose_points = calc_points(results.pose_landmarks.landmark, w, h, visibility=True)
            pose_3d_points = calc_points(results.pose_landmarks.landmark, w, h, visibility=True, use_z=True)

            therapy_level.pose_angles = calc_pose_angles(pose_points)
            therapy_level.pose_angles.update(calc_pose_orientation(pose_3d_points))

            draw_pose_landmarks(frame, pose_points)
            draw_arm_landmarks(frame, pose_points)

        # Flip the image horizontally for a selfie-view display.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.flip(frame, 1)

        os.system('cls')
        print('-' * 100)

        print('\nLeft Hand Angles:-')
        for k, v in therapy_level.left_hand_angles.items():
            print(f'{k}:', v)

        for k, v in therapy_level.left_palm_orientation.items():
            print(f'{k}:', v)

        print('\nRight Hand Angles:-')
        for k, v in therapy_level.right_hand_angles.items():
            print(f'{k}:', v)

        for k, v in therapy_level.right_palm_orientation.items():
            print(f'{k}:', v)

        print('\nPose Angles:-')
        for k, v in therapy_level.pose_angles.items():
            print(f'{k}:', v)

        print('-' * 100)

        cv2.imshow('MediaPipe Holistic', frame)
        if cv2.waitKey(5) == ord('q'):
            break

    with open(pose_path / 'TherapyLevel.pkl', 'wb') as f:
        pickle.dump(therapy_level, f)

cap.release()