import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
import csv
import os
import numpy as np

import cv2
import mediapipe as mp

# Initialize holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initiate video capture
cap = cv2.VideoCapture(0)

# # Initiate holistic model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#
#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#
#             if not ret:
#                 print("Failed to read from camera.")
#                 break
#
#             # Recolor feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#
#             # Make detections
#             results = holistic.process(image)
#
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             # Draw face landmarks
#             mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
#                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                                        )
#
#             # Draw right hand landmarks
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                                        )
#
#             # Draw left hand landmarks
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                                        )
#
#             # Draw pose detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                        )
#
#             cv2.imshow('Raw Webcam Feed', image)
#
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#
#     except KeyboardInterrupt:
#         pass
#
# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
#
# results.face_landmarks.landmark[0].visibility
#
#

# num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
#
#
# landmarks = ['class']
# for val in range(1, num_coords+1):
#     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
#
#
# with open('coords.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(landmarks)

#####------------------------Training part -------------------------

#
# class_name = "Love"
#
# import cv2
# import mediapipe as mp
# import csv
# import numpy as np
#
# # Initialize holistic model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
# # Initiate video capture
# cap = cv2.VideoCapture(0)
#
# # Initiate holistic model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#
#             if not ret:
#                 print("Failed to read from camera.")
#                 break
#
#             # Recolor feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#
#             # Make detections
#             results = holistic.process(image)
#
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             # Draw face landmarks
#             mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
#                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                                        )
#
#             # Draw right hand landmarks
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
#                                        )
#
#             # Draw left hand landmarks
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
#                                        )
#
#             # Draw pose detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                        )
#             # Export coordinates
#             try:
#                 # Extract Pose landmarks
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
#
#                 # Extract Face landmarks
#                 face = results.face_landmarks.landmark
#                 face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
#
#                 # Concate rows
#                 row = pose_row + face_row
#
#                 # Append class name
#                 # Assuming class_name is defined elsewhere in your code
#                 row.insert(0, class_name)
#
#                 # Export to CSV
#                 with open('coords.csv', mode='a', newline='') as f:
#                     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                     csv_writer.writerow(row)
#
#             except Exception as e:
#                 print("Error exporting coordinates:", e)
#
#             cv2.imshow('Raw Webcam Feed', image)
#
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#
#     except KeyboardInterrupt:
#         print("Keyboard interrupt detected. Exiting...")
#
# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

#-----------------------------trained model
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# df = pd.read_csv('coords.csv')
#
# print(df.head())
#
# print(df.tail())
#
#
# X = df.drop('class', axis=1) # features
# y = df['class'] # target value
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
#
#
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
#
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#
#
# pipelines = {
#     'lr':make_pipeline(StandardScaler(), LogisticRegression()),
#     'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
#     'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
#     'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
# }
#
# fit_models = {}
# for algo, pipeline in pipelines.items():
#     model = pipeline.fit(X_train, y_train)
#     fit_models[algo] = model
#
#
#
# fit_models['rc'].predict(X_test)
# from sklearn.metrics import accuracy_score # Accuracy metrics
# import pickle
#
# for algo, model in fit_models.items():
#     yhat = model.predict(X_test)
#     print(algo, accuracy_score(y_test, yhat))
#
# fit_models['rf'].predict(X_test)
#
# with open('body_language.pkl', 'wb') as f:
#     pickle.dump(fit_models['rf'], f)

#------------------------------------------  Model open here  ----------------------------

#
import pickle
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initiate video capture
cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to read from camera.")
                break

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # Draw right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # Draw left hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # Draw pose detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Convert the row to DataFrame for prediction
                X = pd.DataFrame([row])

                # Make Detections
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                # Get left ear coordinates
                left_ear_coords = tuple(np.multiply(
                    np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                    [640, 480]).astype(int))

                # Display class and probability
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display on the image
                cv2.rectangle(image, (left_ear_coords[0], left_ear_coords[1] + 5),
                              (left_ear_coords[0] + len(body_language_class) * 20, left_ear_coords[1] - 30),
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, left_ear_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)

            except Exception as e:
                print("Error:", e)
                pass

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()



