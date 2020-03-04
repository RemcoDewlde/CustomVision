class Box:

    def __init__(self, prediction, frame):
        self.prediction = prediction
        self.frame = frame
        self.height, self.width, self.channel = frame.shape

    def get_start_point(self):
        return (int(self.prediction["boundingBox"]["left"] * self.width), int(
            self.prediction["boundingBox"]["top"] * self.height))

    def get_end_point(self):
        return (int(self.prediction["boundingBox"]["left"] * self.width) + int(
            self.prediction["boundingBox"]["width"] * self.width),
                int(self.prediction["boundingBox"]["top"] * self.height) + int(
                    self.prediction["boundingBox"]["height"] * self.height))