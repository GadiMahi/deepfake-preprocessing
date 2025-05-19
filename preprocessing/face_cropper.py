import cv2
import mediapipe as mp
import os

def crop_faces_from_folder(frame_folder, output_folder):
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(frame_folder):
        if not filename.lower().endswith(".jpg"):
            continue
        path = os.path.join(frame_folder, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                cropped = image[y:y+height, x:x+width]
                out_path = os.path.join(output_folder, f"{filename}")
                cv2.imwrite(out_path, cropped)
                break  # only crop the first face

    print(f"✅ Cropped faces saved to: {output_folder}")