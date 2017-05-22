import cv2
import numpy as np


img = cv2.imread('./zdjecia/real.jpg')


class CiscoRecognizer:
    def __init__(self, photo):
        self.hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        self.photo = photo
        self.min_blue = np.array([90, 150, 0])
        self.max_blue = np.array([110, 255, 90])

        self.min_red = np.array([170, 40, 30])
        self.max_red = np.array([10, 255, 210])
        self.height, self.width , self.depth = self.photo.shape
        self.red_colors = np.zeros((self.height, self.width, 3), np.uint8)
        self.blue_colors = np.zeros((self.height, self.width, 3), np.uint8)


    def is_red(self, point):
        if ((point[0] <= self.max_red[0]) or (point[0] >= self.min_red[0])) and  (self.min_red[1] < point[1] <= self.max_red[1]) and (self.min_red[2] < point[2] <= self.max_red[2]):
            return True
        else:
            return False

    def is_blue(self, point):
        if (self.min_blue[0] < point[0] <= self.max_blue[0]) and  (self.min_blue[1] < point[1] <= self.max_blue[1]) and (self.min_blue[2] < point[2] <= self.max_blue[2]):
            return True
        return False


    def get_red(self):
        for y in range(self.height):
            for x in range(self.width):
                try:
                    if(self.is_red(self.hsv[x,y])):
                        self.photo[x,y][0] = 0
                        self.photo[x,y][1] = 255
                        self.photo[x,y][2] = 132
                except IndexError:
                    print(x,y)
                    continue

    def get_blue(self):
        for y in range(self.height):
            for x in range(self.width):
                try:
                    if(self.is_blue(self.hsv[x,y])):
                        self.photo[x,y][0] = 0
                        self.photo[x,y][1] = 255
                        self.photo[x,y][2] = 132
                except IndexError:
                    print(x,y)
                    continue


    def get_colors(self):
        for x in range(self.height):
            for y in range(self.width):
                try:
                    if(self.is_blue(self.hsv[x,y])):

                        self.red_colors[x, y][0] = 0
                        self.red_colors[x, y][1] = 0
                        self.red_colors[x, y][2] = 0

                        self.blue_colors[x, y][0] = 255
                        self.blue_colors[x, y][1] = 255
                        self.blue_colors[x, y][2] = 255
                    elif (self.is_red(self.hsv[x, y])):
                        self.red_colors[x, y][0] = 255
                        self.red_colors[x, y][1] = 255
                        self.red_colors[x, y][2] = 255

                        self.blue_colors[x, y][0] = 0
                        self.blue_colors[x, y][1] = 0
                        self.blue_colors[x, y][2] = 0
                    else:
                        self.red_colors[x, y][0] = 0
                        self.red_colors[x, y][1] = 0
                        self.red_colors[x, y][2] = 0

                        self.blue_colors[x, y][0] = 0
                        self.blue_colors[x, y][1] = 0
                        self.blue_colors[x, y][2] = 0
                except IndexError:
                    print(x,y)
                    continue


                # if 20 < img[x,y][0] < 50 and 22 < img[x,y][1] < 45 and 90 < img[x,y][2] < 190:
                #     img[x, y][0] = 0
                #     img[x, y][1] = 255
                #     img[x, y][2] = 132
                #
                # elif 30 < img[x,y][0] < 160 and 37 < img[x,y][1] < 83 and 10 < img[x,y][2] < 60:
                #     img[x, y][0] = 0
                #     img[x, y][1] = 255
                #     img[x, y][2] = 132

    def show_photo(self):
        cv2.imshow('image', self.red_colors)
        # cv2.imshow('image', self.blue_colors)

        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('messigray.png',self.photo)
            cv2.destroyAllWindows()


ciscoRecognizer = CiscoRecognizer(img)
ciscoRecognizer.get_colors()
ciscoRecognizer.show_photo()



#
# img = cv2.imread('./zdjecia/new2.jpg', cv2.COLOR_BGR2HSV)
# height, width, depth = img.shape
#
# # img.itemset((10,10,2),100)
# # img[0:height, 0:width, 1] =
# for x in range(height):
#     for y in range(width):
#         if 20 < img[x,y][0] < 50 and 22 < img[x,y][1] < 45 and 90 < img[x,y][2] < 190:
#             img[x, y][0] = 0
#             img[x, y][1] = 255
#             img[x, y][2] = 132
#
#         elif 30 < img[x,y][0] < 160 and 37 < img[x,y][1] < 83 and 10 < img[x,y][2] < 60:
#             img[x, y][0] = 0
#             img[x, y][1] = 255
#             img[x, y][2] = 132
#
#
#     # for x in img.height:
# #     print(img[x, 1])
# #
#
# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()

#
#
# while(1):
#     # Take each frame
#     img = cv2.imread('./zdjecia/rotated2.jpg')
#     hsv = cv2.imread('./zdjecia/rotated2.jpg')
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # define range of blue color in HSV
#     lower_blue = np.array([90,50,0])
#     upper_blue = np.array([110,255,105])
#
#     # Threshold the HSV image to get only blue colors
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(img,img, mask= mask)
#
#     # cv2.imshow('frame',img)
#     # cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()