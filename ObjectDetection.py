import tensorflow as tf
import cv2

class detection:
    def __init__(self):
        self.height=320
        self.width=320
        self.interpreter = tf.contrib.lite.Interpreter(model_path="model.tflite")
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        
        with open('labels.txt') as p:
            self.z=p.readlines()
    
    def draw_rect(self,image, box, category):
        y_min = int(max(1, (box[0] * self.height)))
        x_min = int(max(1, (box[1] * self.width)))
        y_max = int(min(self.height, (box[2] * self.height)))
        x_max = int(min(self.width, (box[3] * self.width)))
       
        cv2.putText(image,category,(x_min-10,y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) 
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

    def objectDetect(self,img):
        category=[]
        new_img = cv2.resize(img, (320, 320))
        self.interpreter.set_tensor(self.input_details[0]['index'], [new_img])
        self.interpreter.invoke()   
        rects = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        
        for index, score in enumerate(scores[0]):
            if score > 0.5:
                class_id = detection_classes[0, index]
                category.append(self.z[int(class_id)])
                self.draw_rect(new_img,rects[0][index],self.z[int(class_id)])
         
        return new_img,category,
    
img = cv2.imread('1.jpg')
de=detection()
image,category=de.objectDetect(img)
if len(category)<1:
    print("no object")
elif 'stop' in category:
    print("STOP there")
    #no further way
elif 'red' in category:
    print("STOP there")
    stop=True
    while stop:
        image,category=de.objectDetect(img)
        if 'yellow' in category:
            print("Ready to go")
        if 'green' in category:
            print("go")
            stop=False
else:
    if 'person' or 'car' in category:
        #check distancse using ultrasonic sensor
        #if person is on the way stop immediately
        print("slow down speed between 15 to 25")
    if 'school' or 'crossing' or 'menAtWork' in category:
        print("slow down speed between 15 to 20")

    if 'right_side' in category:
        print("turn right")
    if 'left_side' in category:
        print("turn left")