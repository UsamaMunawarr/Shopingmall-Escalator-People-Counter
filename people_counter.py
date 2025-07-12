from ultralytics import YOLO # Import the YOLO class from the ultralytics package
import cv2 # Import OpenCV for image processing
import cvzone # Import cvzone for additional computer vision functionalities
import math # Import math for mathematical operations
from sort import * # Import SORT for object tracking(FROM SORT IMPORT EVERYTHING)

#######################################
### Webcam Setup and Model Loading ###
#######################################
# cap = cv2.VideoCapture(2)  # For webcam
# cap.set(3, 720)  # Set the width of the webcam feed
# cap.set(4, 480)   # Set the height of the webcam feed
    ##################################################
    #### Code to check which webcam i am using####
    ##################################################
    # for i in range(3):  # Try indexes 0, 1, and 2 manually
#     print(f"Trying camera index {i}")
#     cap = cv2.VideoCapture(i)
#     if not cap.isOpened():
#         print(f"Camera {i} not available")
#         continue

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Failed to grab frame from camera {i}")
#             break

#         cv2.imshow(f"Camera {i}", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()





######################################
## For Videos Use this code instead ##
########################################
cap = cv2.VideoCapture(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\04_People_Counter\Peoples_Counter\people.mp4")# For video file

model = YOLO(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\04_People_Counter\Peoples_Counter\yolov8n.pt")  # Load the YOLOv8 model weights

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mask = cv2.imread(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\04_People_Counter\Peoples_Counter\mask-1.png")

###############################
### Car Tracker Setup ######
################################
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # Initialize SORT tracker

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]  # Define limits for counting people

totalCountUp = []  # Initialize total count of people
totalCountDown = []  # Initialize total count of people

while True:
    success, img = cap.read()  # Read a frame from the webcam
    #####################################
    #####################################
    ### Apply Mask to Image Region #####
    ######################################
    mask_resized = cv2.resize(mask, (1280, 720))  # Resize the mask to match the car video size
    imgRegion = cv2.bitwise_and(img, mask_resized)  # Apply the mask to the image

    ##################################################
    #### Overlay Graphics on the Image Region ####
    ################################################

    imageGraphics = cv2.imread(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\04_People_Counter\Peoples_Counter\graphics-1.png", cv2.IMREAD_UNCHANGED)

    if imageGraphics is not None:
        # Resize the graphic to a good size (e.g., width=400, height=120)
        imageGraphics = cv2.resize(imageGraphics, (550, 122))

        # Overlay at top-left corner
        img = cvzone.overlayPNG(img, imageGraphics, pos=(730, 260))

        # Display the total count of people going up and down
        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_SIMPLEX,
            2, (139, 195, 74), 5)  # RED color in BGR: (0, 0, 255)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_SIMPLEX,
            2, (50, 50, 230), 5)


    #######################
    results = model(imgRegion, stream=True)  # Run inference on the webcam feed

    ######################
    ## Initialize Detections Array ##
    ###############################
    detections = np.empty((0, 5))  # Initialize an empty array for detections


    for r in results:
        boxes = r.boxes  # Get the bounding boxes from the results
        for box in boxes:
            ##########################################
            # Get the coordinates of the bounding box
            ##########################################
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            w, h = x2-x1,y2-y1  # Calculate width and height from coordinates

            #############################################
            # if we press control and right click on cornerRect blow chunk it will show the code of 
            # cornerRect and we can change the corner of rectangles accordingly 
            ##############################################
            # cvzone.cornerRect(img, (x1, y1, w, h), l=8)  # Draw a rectangle around the detected object

            ################################################
            #### Display Class Name and Confidence Score ###
            ##################################################
            conf = math.ceil((box.conf[0]*100))/100  # Get the confidence score of the detection
            # print(f'Confidence: {conf}')  # Print the confidence score
            # cvzone.putTextRect(img, f'Conf: {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2) # Display the confidence score on the image

            ###########################################
            #### Class Name Display on Image ######
            ##########################################
            cls = int(box.cls[0]) # Get the className Variable id suppose 0 == person, 1 == bicycle, etc.
            currentClass = classNames[cls]  # Get the class name from the classNames list

            if currentClass == "person" and conf > 0.3:  # Check if the detected object is a person
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                #                     scale=0.6, thickness=1, offset=3)  # Display the class name and confidence score on the image
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)  # Draw a rectangle around the detected object
                currentArray = np.array([x1, y1, x2, y2, conf])  # Create an array with the bounding box coordinates and confidence score
                detections = np.vstack((detections, currentArray))  # Append the current detection to the detections array

    resultsTracker = tracker.update(detections)  # Update the tracker with the new detections
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)  # Draw the counting line on the image (red line for going up)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)  # Draw the counting line on the image (red line for going down)

    ##################################################
    #### Draw Bounding Boxes and Count Cars ######
    ##################################################
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)  # Print the tracking result
        w, h = x2-x1,y2-y1  # Calculate width and height from coordinates
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))  # Draw a rectangle around the tracked object
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                                    scale=2, thickness=1, offset=3)  # Display the class name and confidence score on the image

        cx, cy = x1 + w // 2, y1 + h // 2  # Calculate the center coordinates of the bounding box
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        ##########################################
        #### Count People Going Up and Down ######
        ##########################################

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[3] + 15:  # Check if the center of the bounding box is within the counting line
            if totalCountUp.count(id) == 0:  # Check if the ID is not already counted
                totalCountUp.append(id)  # Increment the total count of people
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)  # Draw the counting line on the image

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[3] + 15:  # Check if the center of the bounding box is within the counting line
            if totalCountDown.count(id) == 0:  # Check if the ID is not already counted
                totalCountDown.append(id)  # Increment the total count of people
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    

    cv2.imshow("Image", img)  # Display the webcam feed
    # cv2.imshow("ImageRegion", imgRegion)  # Display the masked image region
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for a key press for 1 millisecond (Display main image)
        break





