import get_detector
import learning_face
import data

class faceEmotion():
    def get_detector(self):
        get_detector.get_detector(self)

    def learning_face(self):
        learning_face.learning_face(self)
    def data(self):
        data.data(self)

if __name__ == "__main__":
    face_emotion = faceEmotion()
    face_emotion.get_detector()
    face_emotion.learning_face()
