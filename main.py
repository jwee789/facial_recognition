import face_recognition
import cv2
import numpy as np
import datetime
import winsound

#Colors we're using
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

#Add faces of admins into an array
jaden_img = face_recognition.load_image_file("known\jaden.jpg")
jaden_face = face_recognition.face_encodings(jaden_img)
admins = [
    jaden_face
]


font = cv2.FONT_HERSHEY_COMPLEX


def centerText(img, text, color):
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) // 2

    # add text centered on image
    cv2.putText(img, text, (textX, textY), font, 1, color, 1)

    return img


def main():
    cap = cv2.VideoCapture(1)

    frame = 0

    access = "NO FACE DETECTED"
    color = BLUE

    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()

        img = cv2.putText(img, 'Time: ' + str(datetime.datetime.now()), (30,20), font, 0.35, (0,0,0), 1, cv2.LINE_AA)

        #Find faces
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        #Check if faces detected
        if not face_locations:
            access = "NO FACE DETECTED"
            color = BLUE
            winsound.PlaySound(None, winsound.SND_PURGE)

        #Draw rectangles around all faces
        for face in face_locations:
            cv2.rectangle(img, (face[1], face[0]), (face[3], face[2]), BLACK, 3)

        if frame % 5 == 0:
            #check if jaden is present
            for face in face_encodings:
                matches = face_recognition.compare_faces(admins, face, 0.1)
                if matches[0].all() == True:
                    access = "FACE RECOGNIZED"
                    color = GREEN
                    winsound.PlaySound(None, winsound.SND_PURGE)
                else:
                    access = "INTRUDER"
                    color = RED
                    winsound.PlaySound('police.wav', winsound.SND_ASYNC)

        #Put text on screen
        img = centerText(img, access, color)

        #Display the resulting frame
        img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)

        cv2.imshow('frame', img)
        frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()