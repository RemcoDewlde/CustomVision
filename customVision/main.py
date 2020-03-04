import sys
import tensorflow as tf
import numpy as np
from classes.object_detection import ObjectDetection
import cv2
import pafy as pafy
from classes.timer import Timer
from classes.box import Box

MODEL_FILENAME = 'model/model.pb'
LABELS_FILENAME = 'model/labels.txt'


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]


def main(vidId):
    # Load youtube Vided
    url = "https://youtu.be/" + vidId
    # url = 'https://youtu.be/XAXwmMu8otM'
    # url = 'https://youtu.be/BRMK77NUsyU'
    vlink = pafy.new(url)
    play = vlink.getbest()

    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    # generate random color
    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    od_model = TFObjectDetection(graph_def, labels)

    cap = cv2.VideoCapture(play.url)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channel = frame.shape

        t = Timer()
        t.start()
        predictions = od_model.predict_image(frame)
        t.stop()

        for prediction in predictions:
            # check if probability is higher than 25%
            if prediction["probability"] > 0.25:
                box = Box(prediction, frame)

                # Get the color associated with the tagId
                color = colors[prediction["tagId"]]
                thickness = 2
                probability = str(round(prediction["probability"], 2))
                probability = probability[2:]

                # # Draw a rectangle around the detected object
                frame = cv2.rectangle(frame, box.get_start_point(), box.get_end_point(), color, thickness)

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
    if len(sys.argv) <= 1:
        print('USAGE: {} Youtube video id'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
