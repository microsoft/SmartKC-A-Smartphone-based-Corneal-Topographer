import cv2 
  
  
# read image 
img = cv2.imread('592930_right_2.jpg') 
#img = cv2.resize(img, (img.shape[1]//8, img.shape[0]//8), interpolation=cv2.INTER_LINEAR)
  
# show image 
cv2.imshow('image', img) 
   
#define the events for the 
# mouse_click. 
def mouse_click(event, x, y,  
                flags, param): 
      
    # to check if left mouse  
    # button was clicked 
    if event == cv2.EVENT_LBUTTONDOWN: 
          
        # font for left click event 
        font = cv2.FONT_HERSHEY_TRIPLEX 
        LB = 'Left Button'
          
        # display that left button  
        # was clicked. 
        cv2.putText(img, LB, (x, y),  
                    font, 1,  
                    (255, 255, 0),  
                    2)  
        cv2.imshow('image', img) 
        print(x, y)
          
  
cv2.setMouseCallback('image', mouse_click) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 