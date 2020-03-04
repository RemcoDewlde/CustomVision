import sys
import cv2
import numpy as np
import tensorflow as tf
import pafy
from classes.object_detection import ObjectDetection
from classes.box import Box

MODEL_FILENAME = 'model/model.pb'
LABELS_FILENAME = 'model/labels.txt'


def get_vid(vid_id):
    """Expects a youtube video id, returns the video url for cv2.VideoCapture"""
    url = "https://youtu.be/" + vid_id
    link = pafy.new(url)
    video = link.getbest()
    print(f'Playing: {link.title}')
    return video.url


def load_labels():
    """load labels from labels.txt in same dir as file"""
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
        return labels


def gen_colors(labels):
    """generate a random color for every label in labels.txt"""
    return np.random.uniform(0, 255, size=(len(labels), 3))


def load_graph_def():
    """Load Tensorflow graph"""
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def get_tf_graph(graph_def):
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
        tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")
    return graph


def make_session(graph):
    """make a tensorflow session and return this session"""
    return tf.compat.v1.Session(graph=graph)


def predict(sess, frame):
    inputs = np.array(frame, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR
    output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
    outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
    return outputs[0]


def main(vid):
    print("Starting....")
    video = get_vid(vid)
    labels = load_labels()
    colors = gen_colors(labels)
    graphdef = load_graph_def()
    graph = get_tf_graph(graphdef)
    sess = make_session(graph)

    od_model = ObjectDetection(labels)

    cap = cv2.VideoCapture(video)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channel = frame.shape

        result = predict(sess, frame)
        predictions = od_model.postprocess(result)

        for prediction in predictions:
            # check if probability is higher than 25%
            if prediction["probability"] > 0.25:
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
