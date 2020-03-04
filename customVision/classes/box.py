class Box:
    """
     A class used to represent two points

    """

    def __init__(self, prediction, frame):
        """
        @param prediction: Object
        @param frame: numpy array / cv2 frame array
        """
        self.prediction = prediction
        self.frame = frame
        self.height, self.width, self.channel = frame.shape

    def get_start_point(self):
        """
        @return Tuple with (x,y) coordinates
        """
        return (int(self.prediction["boundingBox"]["left"] * self.width), int(
            self.prediction["boundingBox"]["top"] * self.height))

    def get_end_point(self):
        """
        @return: Tuple with (x,y) coordinates
        """
        return (int(self.prediction["boundingBox"]["left"] * self.width) + int(
            self.prediction["boundingBox"]["width"] * self.width),
                int(self.prediction["boundingBox"]["top"] * self.height) + int(
                    self.prediction["boundingBox"]["height"] * self.height))
