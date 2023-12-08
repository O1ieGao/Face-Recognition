import cv2
import time
import threading
import numpy as np

################################################################################
# Configuration variables
################################################################################
is_windows = True # Set to False for MacOS/Linux

esc_key = 27
window_name = "Face detection"

text_font = cv2.FONT_HERSHEY_DUPLEX
text_pos_line1 = (4, 16)
text_pos_line2 = (4, 36)
text_scale = 0.5
line_color_in = (63, 63, 255)
line_color_out = (0, 0, 0)
line_thick_in = 1
line_thick_out = 3

framerate_res = 10

# Global data variables
faces = []
img = []

# Set up the classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

################################################################################
# Face detection function (run as separate thread)
################################################################################
def face_detection(stop_sig, img_lock, face_cond):
    # Use global variable for faces list
    global faces

    # Keep looping until recieving a signal to stop
    while not stop_sig.is_set():
        # Try to do the face detection
        try:
            # Require grayscale image, need access to image
            with img_lock:
                g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Require that this thread is the only one accessing faces list
            with face_cond:
                # Do face detection
                faces = classifier.detectMultiScale(g_img, scaleFactor=1.1, minNeighbors=6) # Replace this line with the actual face detection [150, 150, 150, 150]
                print(faces)
                # Add a little bit of a delay to simulate being on the robot
                time.sleep(0.15) # Remember to remove this once you move your code to the robot
                # Block thread until woken up - makes sure main thread can copy data before writing new data back
                face_cond.wait()

        # If an exception is thrown, skip this loop iteration and try again
        except:
            pass


################################################################################
# Main code, sets up worker threads and runs main thread
################################################################################
if __name__ == '__main__':
    # Set up camera
    if is_windows:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set width
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Set height

    # Create the window
    cv2.namedWindow(window_name)

    # Create event for notifying worker threads to stop
    stop_sig = threading.Event()

    # Create mutual exclusion locks
    img_lock = threading.Lock()
    face_cond = threading.Condition()

    # Create and start the separate thread for doing face detection
    face_thread = threading.Thread(target=face_detection, args=(stop_sig, img_lock, face_cond, ))
    face_thread.start()

    # Variables for handling frames
    framerate = 0.0
    fps_counter = 0
    start = time.time() # First frame time

    # Buffers for storing copies of critical data
    img_out = []
    faces_out = []

    # The main loop
    while True:
        # Get the frame, need image mutex lock
        with img_lock:
            read_success, img = camera.read()
            if read_success and img is not None:
                img_out = img.copy()

        # Frame is good
        if read_success and img_out is not None:
            # Clone faces if mutex lock is available
            if face_cond.acquire(blocking = False):
                # If there's at least one face, copy the list, otherwise reset to empty list
                if len(faces) > 0:
                    faces_out = faces.copy()
                else:
                    faces_out = []
                # Notify waiting threads and then release the lock
                face_cond.notifyAll()
                face_cond.release()

            # Render boxes around most recent set of faces
            for face in faces_out:
                cv2.rectangle(img_out, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), line_color_out, line_thick_out)
                cv2.rectangle(img_out, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), line_color_in, line_thick_in)

            # Add framerate to output image
            cv2.putText(img_out, '{:.2f} FPS - Press ESC to quit'.format(framerate), text_pos_line1, text_font, text_scale, line_color_out, line_thick_out, cv2.LINE_AA)
            cv2.putText(img_out, '{:.2f} FPS - Press ESC to quit'.format(framerate), text_pos_line1, text_font, text_scale, line_color_in, line_thick_in, cv2.LINE_AA)

            # Add number of faces detected to image
            cv2.putText(img_out, 'Number of faces detected: {}'.format(len(faces_out)), text_pos_line2, text_font, text_scale, line_color_out, line_thick_out, cv2.LINE_AA)
            cv2.putText(img_out, 'Number of faces detected: {}'.format(len(faces_out)), text_pos_line2, text_font, text_scale, line_color_in, line_thick_in, cv2.LINE_AA)

            # Render image to screen
            cv2.imshow(window_name, img_out)

            # Compute FPS every so often
            fps_counter += 1
            if fps_counter >= framerate_res:
                end = time.time()
                framerate = (framerate_res / (end - start))
                fps_counter = 0
                start = end
        
        # Error: getting frame failed, skip frame        
        else:
            pass

        # Check to see if a key was pressed this frame
        k = cv2.waitKey(1)
        # ESC pressed, exit loop
        if k % 256 == esc_key:
            print("ESC pressed, closing...")
            break

    # Signal to other threads that program is ending
    stop_sig.set()

    # Notify face detection thread to stop waiting
    with face_cond:
        face_cond.notifyAll()
    face_thread.join()

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()