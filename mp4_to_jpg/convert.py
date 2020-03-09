import cv2
import sys
import os
import progressbar
import datetime


def main(path):
    # open video with cv2
    cap = cv2.VideoCapture(path)

    # get length of the video in frames for the progressbar
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get current directory and make a new folder
    # the name of the folder will be the current timestamp
    work_dir = os.path.dirname(os.path.realpath(__file__))
    new_dir_name = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(work_dir, new_dir_name)
    os.mkdir(path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is None:
            print("Finished")
            print("Location:{}".format(path))
            print("Saved: {} images".format(count))
            break
        else:
            # get the current frame number
            count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progressbar.printProgressBar(count, length, prefix='Progress:', suffix='Complete', autosize=True)

            # convert a frame to a jpg and save it
            cv2.imwrite('{}/{}.jpg'.format(path, count), frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])
