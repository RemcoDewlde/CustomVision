import cv2
import sys
import numpy as np
import pafy as pafy
import tensorflow as tf
from classes.object_detection import ObjectDetection
from classes.box import Box

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""

    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :,
                 (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def get_vid(vid_id):
    """Expects a youtube video id, returns the video url for cv2.VideoCapture"""
    url = "https://youtu.be/" + vid_id
    link = pafy.new(url)
    video = link.getbest()
    print(f'Playing: {link.title}')
    return video.url


def load_labels():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    return labels


def gen_colors(labels):
    """generate a random color for every label in labels.txt"""
    return np.random.uniform(0, 255, size=(len(labels), 3))


def get_interpreter():
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILENAME)
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()[0]['index']
    input_index = interpreter.get_input_details()[0]['index']

    return interpreter, input_index, output


def main(vid):
    print("starting...")
    video = get_vid(vid)
    labels = load_labels()
    colors = gen_colors(labels)
    interpreter = get_interpreter()

    input_index = interpreter[1]
    output_index = interpreter[2]

    interpreter = interpreter[0]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    cap = cv2.VideoCapture(video)

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interpreter.resize_tensor_input(input_index, (1, height, width, 3))
        interpreter.allocate_tensors()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        inputs = np.array(frame, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]

        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        result = interpreter.get_tensor(output_index)[0]

        predictions = od_model.postprocess(result)

        for prediction in predictions:
            # check if probability is higher than 25%
            p = int(cv2.getTrackbarPos('%', 'frame')) / 100
            if prediction["probability"] > p:

                # Create a instance of the box class
                box = Box(prediction, frame)

                # Get the color associated with the tagId
                color = colors[prediction["tagId"]]
                thickness = 2
                probability = str(round(prediction["probability"], 2))
                probability = probability[2:]

                # Draw a rectangle around the detected object
                frame = cv2.rectangle(frame, box.get_start_point(), box.get_end_point(), color, thickness)

                # Show the label associated with the object
                cv2.putText(frame, str(prediction["tagName"]) + " | Probability:" + probability + "%",
                            (int(prediction["boundingBox"]["left"] * width),
                             int(prediction["boundingBox"]["top"] * height) - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Our operations on the frame come here

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} Youtube video id'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
