# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import tensorflow as tf
import numpy as np
from classes.object_detection import ObjectDetection
import cv2
import pafy as pafy

MODEL_FILENAME = 'model/model.tflite'
LABELS_FILENAME = 'model/labels.txt'


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

        print(inputs.shape)
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def nothing():
    pass


def main():
    url = 'https://youtu.be/XAXwmMu8otM'
    vlink = pafy.new(url)
    play = vlink.getbest()

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(play.url)
    cv2.createTrackbar('%', 'frame', 0, 100, nothing)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channel = frame.shape

        predictions = od_model.predict_image(frame)

        for prediction in predictions:
            # check if probability is higher than 25%
            p = int(cv2.getTrackbarPos('%', 'frame')) / 100
            if prediction["probability"] > p:
                # Start point of the rectangle (top left)
                start_point = (
                    int(prediction["boundingBox"]["left"] * width), int(prediction["boundingBox"]["top"] * height))

                # end point of of the rectangle (bottom right)
                end_point = (
                    int(prediction["boundingBox"]["left"] * width) + int(prediction["boundingBox"]["width"] * width),
                    int(prediction["boundingBox"]["top"] * height) + int(prediction["boundingBox"]["height"] * height))

                # Get the color associated with the tagId
                color = colors[prediction["tagId"]]
                thickness = 2
                probability = str(round(prediction["probability"], 2))
                probability = probability[2:]

                # Draw a rectangle around the detected object
                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

                # Show the label associated with the object
                cv2.putText(frame, str(prediction["tagName"]) + " | Probability:" + probability + "%",
                            (int(prediction["boundingBox"]["left"] * width),
                             int(prediction["boundingBox"]["top"] * height) - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
