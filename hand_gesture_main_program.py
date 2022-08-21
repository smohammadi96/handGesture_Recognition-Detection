from tkinter import PhotoImage
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2 ,threading
from threading import Thread
from say_numbers.predictgest import say_numbers
from direction.main import direction
from pos_net.Predictor import pos_net
import cv2
from i_love_you.Recognize_Gesture import i_love_you


def main_function(hand_gesture):
    global GUI , filter
    video_capture = cv2.VideoCapture(0)
    cv2.putText(img = video_capture, 
                        text = str(10),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                        fontScale = 3, 
                        color = (255,255,0),
                        thickness = 2)
    while hand_gesture.is_set():
        ret, image = video_capture.read()
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # conerts to PIL format3
        image = Image.fromarray(image)
        # Converts to a TK format to visualize it in t


        # GUI
        image = ImageTk.PhotoImage(image)
        # Actualize the image in the panel to show it
        GUI.configure(image=image)
        GUI.image = image
    video_capture.release()
    
hand_gesture_window = Tk()
hand_gesture_window.title("hand gesture recognition")
logo_gesture = PhotoImage(file='./hand.png')
hand_gesture_window.tk.call('wm', 'iconphoto', hand_gesture_window, logo_gesture)

button_1 = Button(hand_gesture_window, activebackground="silver",bd="50",bg="pink",text="say i love you", fg="black" , command =lambda :  i_love_you())
button_2 = Button(hand_gesture_window, activebackground="pink",bd="50",bg="silver",text="say direction my hand", fg="black" , command = lambda: direction() )
button_3 = Button(hand_gesture_window, activebackground="pink",bd="50",bg="violet",text="say numbers", fg="black" , command = lambda: say_numbers() )
button_4 = Button(hand_gesture_window, activebackground="silver",bd="50",bg="sky blue",text="simple-hand-gesture", fg="black" , command =lambda :  pos_net())
button_1.flash()
button_1.pack(side="top", fill="both", padx="50",pady="4")
button_2.pack(side="top", fill="both", padx="50",pady="4")
button_3.pack(side="top", fill="both", padx="50",pady="4")
button_4.pack(side="top", fill="both", padx="50",pady="4")

GUI = Label(hand_gesture_window)
GUI.pack()
buttons = [button_1, button_2, button_3]
#run hand_gesture GUI
run_hand_gesture = threading.Event()
run_hand_gesture.set()
Thread(target=main_function, args=(run_hand_gesture,)).start()
def end_of_hand_gesture():
        global hand_gesture_window, run_hand_gesture
        run_hand_gesture.clear()
        hand_gesture_window.destroy()
        print ("you close the hand_gesture recognition app...")
hand_gesture_window.protocol("terminate_window", end_of_hand_gesture)
hand_gesture_window.mainloop() #creates loop of GUI
