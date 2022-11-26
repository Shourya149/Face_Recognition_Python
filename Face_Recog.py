from platform import release
from datetime import datetime
import face_recognition as fr
import numpy as np
import csv
import cv2
import os


#Part-1 : Activating the webcam and capturing input
webcam_input=cv2.VideoCapture(0)


#Part-2 : Retrieving Employee Details
path="Images"
image_list=os.listdir(path)

employee_ids=[]
employee_names=[]
employee_encodings=[]

for image in image_list :
    current_image=cv2.imread(f'{path}/{image}')
    current_image_encoding=fr.face_encodings(current_image)[0]
    employee_encodings.append(current_image_encoding)
    image_name=os.path.splitext(image)[0]
    employee_ids.append(image_name.split("_")[0])
    employee_names.append(image_name.split("_")[1])


#Part-3 : Opening CSV file and retrieving previously recorded data
current_date=datetime.now().strftime("%d-%m-%Y")
try :
    file=open(current_date+".csv","r",newline="")
except :
    file=open(current_date+".csv","w+",newline="")

file_reader=csv.reader(file)
present_employees=[]

for line in file_reader :
    present_employees.append(line)

if len(present_employees)==0 :
    present_employees.append(["Id","Name","Timing"])

file.close()

#Part-4 : Face Detection , Face Matching and Appendinng Data in CSV fle
#absent_employee=employee_names.copy()
file=open(current_date+".csv","w+",newline="")
file_writer=csv.writer(file)

captured_face_location=[]
capptured_face_encodings=[]

name=""
id=""

while True :
    #Part-4.1 : Capturing frame from webcam  
    dummy,frame=webcam_input.read()
    smaller_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    smaller_frame_rgb=smaller_frame[:,:,::-1]


    #Part-4.2 : Capturing Face from the frame
    captured_face_location=fr.face_locations(smaller_frame_rgb)
    capptured_face_encodings=fr.face_encodings(smaller_frame_rgb,captured_face_location)

    
    #Part-4.3 : Recognizing the face  
    for encoding in capptured_face_encodings :
        possible_matches=fr.compare_faces(employee_encodings,encoding)
        similarity_index=fr.face_distance(employee_encodings,encoding)
        best_match=np.argmin(similarity_index)

        if possible_matches[best_match] :
            id=employee_ids[best_match]
            name=employee_names[best_match]

        
        #Part-4.4 : Storing Updated Data in List
        flag=1
        for employee in present_employees:
            if id==employee[0] and name==employee[1] :
                flag=0
                break

        if flag :
            current_time=datetime.now().strftime("%H-%M-%S")
            if len(name)>0 and len(id)>0 :
                present_employees.append([id,name,current_time])
            print("Face captured")
            print("Face Recognized.....")
            print("Id :",id,"\tName :",name)
            print("Data is written into the file successfully\n\n\n")

        else :
            print(name," ,Your Attendance is already marked")
        
    #Part-4.5 : Loop Exit condition
    cv2.imshow("Attendance System ",frame)
    if cv2.waitKey(1) & 0xFF in [ord('e'),ord('E')] :
        break


#Part-5 : Writng data in CSV file
file_writer.writerows(present_employees)

#Part-6 : Closing webcam inputs
webcam_input.release()
cv2.destroyAllWindows()
file.close()










