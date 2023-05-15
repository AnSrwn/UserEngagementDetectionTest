# Sources:
# https://github.com/The-revolutionary-army/Engagement-and-comprehension-level-detection
# https://github.com/yogesh-kamat/EduMeet

from flask import Flask, render_template, Response
from camera import VideoCameraModel
import tensorflow as tf
import keras
import cv2
import numpy as np
import dlib

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        image = np.array(frame)

        try:
            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            roi = []
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                # if no face skip predictions
                if len(image[y1:y2, x1:x2]) <= 0 or len(image[y1-100:y2+100, x1-100:x2+100]) <= 0:
                    print('no face')
                    continue
                # append faces
                roi.append(cv2.resize(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (48,48)))
                # get predictions
                predictions = []
                if len(roi)>0:
                    test_images = np.expand_dims(roi, axis=3)
                    predictions = parallel_model.predict(test_images)
    
                    boredom = round(predictions[0][0][1],3)
                    engagement = round(predictions[1][0][1],3)
                    confusion = round(predictions[2][0][1],3)
                    frustration = round(predictions[3][0][1],3)
                    
                    text_up = 'Bored: '+str(boredom)+' Engaged: '+str(engagement)
                    text_down = 'Confused: '+str(confusion)+' Frustrated: '+str(frustration)

            # Draw rect
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Write label Up
            cv2.putText(frame, text_up, (x1 - 100, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # Write label Down
            cv2.putText(frame, text_down, (x1 - 100, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except UnboundLocalError:
            # if no predictions
            continue
        except:
            continue

        try:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except cv2.error as e:
            print("No frame: {e}")


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCameraModel()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parallel_model = keras.models.load_model('models/parallel_model.h5')
    # detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat') # better but slower
    # http://dlib.net/cnn_face_detector.py.html

    app.run(host='0.0.0.0', debug=True)
