import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('img/test.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print(faces)
crop_img = []
FORMAT_WIDTH = 128
FORMAT_HEIGHT = 128
dim = (FORMAT_WIDTH, FORMAT_HEIGHT)
# Draw rectangle around the faces
for face_id, (x, y, w, h) in enumerate(faces):
    temp_crop = img[y:y+h, x:x+w].copy()
    # resize to format size
    temp_crop = cv2.resize(temp_crop, dim)
    cv2.imwrite("output/c_{}.jpg".format(face_id), temp_crop)

COLOR_BLUE = (255, 0, 0)
person_id = ["person_0", "person_1", "person_2", "person_3", "person_4", "person_5", "person_6"]
for face_id, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_BLUE, 2)
    cv2.putText(img, person_id[face_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLUE, 2)


# Display the output
cv2.imshow('img', img)
cv2.waitKey(0)
