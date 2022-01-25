# Import
import numpy as np
import cv2
import os


######### KNN Code ##############

def distance(v1 , v2) :
    return np.sqrt(((v1-v2) ** 2).sum())

def knn(train , test , k = 5) :
    dist = []

    for i in range(train.shape[0]) :
        # Get the vector and label
        ix = train[i,:-1]
        iy = train[i,-1]
        # Compute the distance from test point
        d = distance(test , ix)
        dist.append([d,iy])

    # Sort based on distance and get top k
    dk = sorted(dist , key = lambda x : x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:,-1]

    #Get frequencies of each label
    output = np.unique(labels , return_counts = True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

##########################################################################

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("C:\\Users\\prach\\PycharmProjects\\FaceRecognitns\\GeneratingSelfieTrainingData\\haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = 'C:\\Users\\prach\\PycharmProjects\\FaceRecognitns\\GeneratingSelfieTrainingData\\data\\'

# train data
face_data = []

# label of data
labels = []

# labelling of different files
class_id = 0
# mapping between id and name
names = {}


###############################################################################

# i will iterate over my directory
# u will get all the files in this path
for fx in os.listdir(dataset_path) :
    if fx.endswith('npy') :
        # Creating a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        # here we are ggiving file name along with the path
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create labels for the class
        # eg 10 images of a person
        # then hum ones array me vo class multiply karke -- we got pure array pe of size (10,) - class_id
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)


face_dataset = np.concatenate(face_data , axis = 0)
face_labels = np.concatenate(labels , axis = 0).reshape(-1,1)

print(face_dataset.shape)
print(face_labels.shape)

# our trainset has both features and labels
trainset = np.concatenate((face_dataset , face_labels) , axis = 1)
print(trainset.shape)


################################################################################


while True :
    ret , frame = cap.read()
    if  ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame , 1.3 , 5)

    for face in faces :
        x , y , w , h = face

        # get the face region of interst
        offset = 10
        print(frame.shape)
        face_section = frame[y-offset : y+h+offset , x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # Predicted label(out)
        out = knn(trainset , face_section.flatten())
        print(out)

        # Display on the screen the anme and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame , pred_name , (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,0,0) , 2 , cv2.LINE_AA)
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,255) , 2)

    cv2.imshow("Faces" , frame)

    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')) :
        break

cap.release()
cv2.destroyAllWindows()