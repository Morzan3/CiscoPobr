import cv2
import numpy as np
from random import randint
from invariant_counter import InvariantCounter
from pprint import pprint
import math


MAX_LINE_SIZE = 10000
MIN_LINE_SIZE = 190
MAX_LETTER_SIZE = 10000
MIN_LETTER_SIZE = 200

img = cv2.imread('./final/medium/4.jpg')


class CiscoRecognizer:
    def __init__(self, photo):
        self.hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        self.photo = photo

        self.min_blue = np.array([90, 60, 40])
        self.max_blue = np.array([150, 255, 180])

        self.min_red = np.array([160, 90, 90])
        self.max_red = np.array([20, 255, 250])
        self.height, self.width, self.depth = self.photo.shape
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


    def is_pixel_of_given_color(self, pixel, color):
        if pixel[0] == color[2] and pixel[1] == color[1] and pixel[2] == color[0]:
            return True
        else:
            return False

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
    def find_logo(self):
        self.get_colors()
        self.calculate_photo_segments_invariant()
        letter_segments, logo_segments = self.find_similar_segments()
        self.group_up_segments(letter_segments, logo_segments)
        self.show_photo()


    def group_up_segments(self, letter_segments, logo_segments):
        s_segments = [(x,y) for x,y in letter_segments if x == 's']
        rest = [(x,y) for x,y in letter_segments if x != 's']

        grouped_letter_segments = []
        for segment in s_segments:
            grouped_letter_segment = self.add_nearest_segments(segment, rest)
            grouped_letter_segments.append(grouped_letter_segment)
            rest = [x for x in rest if x not in grouped_letter_segments]


        self.mark_grouped_segments(grouped_letter_segments)
        print(grouped_letter_segments)


    def mark_grouped_segments(self, grouped_letter_segments):
        for group in grouped_letter_segments:
            x_min, x_max, y_min, y_max = self.get_segment_boarder_points(group)

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    self.photo[x][y] = [255, 0, 0]

    def get_segment_boarder_points(self, group):
        x_min = self.height
        y_min = self.width
        x_max =0
        y_max= 0

        for segment in group:
            for point in segment[1].segment:
                if point[0] > x_max:
                    x_max = point[0]
                elif point[0] < x_min:
                    x_min = point[0]
                if point[1] > y_max:
                    y_max = point[1]
                elif point[1] < y_min:
                    y_min = point[1]

        return x_min, x_max, y_min, y_max


    def add_nearest_segments(self, s_segment, rest_segments):
        grouped_segments = [s_segment]
        i_segments = [(x,y) for x,y in rest_segments if x == 'i']
        c_segments = [(x,y) for x,y in rest_segments if x == 'c']
        o_segments = [(x,y) for x,y in rest_segments if x == 'o']

        i_segment = self.find_nearest_segment(s_segment, i_segments)
        o_segment = self.find_nearest_segment(s_segment, o_segments)
        c_segment_first = self.find_nearest_segment(s_segment, c_segments)
        c_segments.remove(c_segment_first)
        c_segment_second = self.find_nearest_segment(s_segment, c_segments)

        grouped_segments = grouped_segments + [i_segment, o_segment, c_segment_first, c_segment_second]
        return grouped_segments

    def find_nearest_segment(self, main_segment, rest):
        nearest_segment = rest[0]

        for segment in rest:
            if self.calculate_distance(main_segment, segment) < self.calculate_distance(main_segment, nearest_segment):
                nearest_segment = segment

        return nearest_segment

    def calculate_distance(self, first_point, second_point):
        distance = math.sqrt(np.power([first_point[1].center_i - second_point[1].center_i,
                                       first_point[1].center_j - second_point[1].center_j], 2).sum())
        return distance

    def mark_and_add_segment(self, photo, segment_list, starting_point, segment_color):
        target_color = [randint(0,255), randint(0,255), randint(0,255)]
        photo[starting_point[0], starting_point[1]] = target_color

        coordinates_to_check = []
        coordinates_set = set()
        coordinates_set.add(starting_point)
        coordinates_to_check.append(starting_point)

        while True:

            try:
                coordinates = coordinates_to_check.pop()
            except IndexError:
                break

            (x_value, y_value) = coordinates

            try:

                if self.is_pixel_of_given_color(photo[x_value + 1, y_value], segment_color):
                    photo[x_value + 1, y_value] = target_color
                    coordinates_set.add((x_value + 1, y_value))
                    coordinates_to_check.append((x_value + 1, y_value))

                if self.is_pixel_of_given_color(photo[x_value - 1, y_value], segment_color):
                    photo[x_value - 1, y_value] = target_color
                    coordinates_set.add((x_value - 1, y_value))
                    coordinates_to_check.append((x_value - 1, y_value))

                if self.is_pixel_of_given_color(photo[x_value, y_value + 1], segment_color):
                    photo[x_value, y_value + 1] = target_color
                    coordinates_set.add((x_value, y_value + 1))
                    coordinates_to_check.append((x_value, y_value + 1))

                if self.is_pixel_of_given_color(photo[x_value, y_value - 1], segment_color):
                    photo[x_value, y_value - 1] = target_color
                    coordinates_set.add((x_value, y_value - 1))
                    coordinates_to_check.append((x_value, y_value - 1))

            except IndexError:
                continue


        segment_list.append(list(coordinates_set))
        return 1




    def extract_segments(self, sanitize = False):
        red_segments = []
        for x in range(self.height):
            for y in range(self.width):
                if self.is_pixel_of_given_color(self.red_colors[x, y], [255, 255, 255]):
                    self.mark_and_add_segment(self.red_colors, red_segments, (x,y), [255, 255, 255])

        if sanitize:
            red_segments = self.delete_red_segments(red_segments)

        blue_segments = []
        for x in range(self.height):
            for y in range(self.width):
                if self.is_pixel_of_given_color(self.blue_colors[x, y], [255, 255, 255]):
                    self.mark_and_add_segment(self.blue_colors, blue_segments, (x, y), [255, 255, 255])

        if sanitize:
            blue_segments = self.delete_blue_segments(blue_segments)


        return red_segments, blue_segments


    def delete_red_segments(self, segment_list):
        filtered_list = []
        for segment in segment_list:
            if len(segment) > MAX_LETTER_SIZE or len(segment) < MIN_LETTER_SIZE:
                for point in segment:
                    self.red_colors[point[0], point[1]] = [0, 0, 0]
            else:
                filtered_list.append(segment)
        return filtered_list

    def delete_blue_segments(self, segment_list):
        filtered_list = []
        for segment in segment_list:
            if len(segment) > MAX_LINE_SIZE or len(segment) < MIN_LINE_SIZE:
                for point in segment:
                    self.blue_colors[point[0], point[1]] = [0, 0, 0]
            else:
                filtered_list.append(segment)
        return filtered_list

    def calculate_photo_segments_invariant(self):
        red_segments, blue_segments = self.extract_segments(True) # *****************************

        self.letter_segments = []
        self.logo_segments = []

        for red_segment in red_segments:
            invariant_counter = InvariantCounter(red_segment)
            invariant_counter.calculate_needed_invariants()
            self.letter_segments.append(invariant_counter)

        for blue_segment in blue_segments:
            invariant_counter = InvariantCounter(blue_segment)
            invariant_counter.calculate_needed_invariants()
            if invariant_counter.NM1 != 0 and invariant_counter.NM2 != 0 and invariant_counter.NM7 !=0:
                self.logo_segments.append(invariant_counter)


    def find_similar_segments(self):
        letters = []

        logo_segments = []

        for segment in self.letter_segments:

            # find C
            if 0.28 < segment.NM1 < 0.39 and 0.014 < segment.NM2 < 0.03 and 0.005 < segment.NM3 < 0.013 and 0.018 < segment.NM7 < 0.03:
                print("found C")
                letters.append(('c',segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find I
            elif 0.29 < segment.NM1 < 0.4 and 0.065 < segment.NM2 < 0.13 and 0.006 < segment.NM7 < 0.0073:
                print("found I")
                letters.append(('i', segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find S
            elif 0.24 < segment.NM1 < 0.34 and 0.017 < segment.NM2 < 0.034 and 0.00002 < segment.NM3 < 0.00183 and 0.011 < segment.NM7 < 0.021:
                print("found S")
                letters.append(('s', segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find 0
            elif 0.24 < segment.NM1 < 0.33 and 3.78e-06 < segment.NM2 < 0.0003 and 4.4e-08 < segment.NM3 < 6e-05 and 0.014 < segment.NM7 < 0.027:
                letters.append(('o', segment))
                print("found O")
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            else:
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [0, 0, 0]


        for segment in self.logo_segments:
            # Smallest segments
            if 0.17 < segment.NM1 < 0.243 and 0.007 < segment.NM2 < 0.031 and 0 <= segment.NM3 < 0.0098 and 0.0064 < segment.NM7 < 0.009:
                            print("Small segments")
                            logo_segments.append(segment)
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]


            #Medium segments
            elif 0.27 < segment.NM1 < 0.46 and 0.04 < segment.NM2 < 0.19 and 0 <= segment.NM3 < 0.008 and 0.006 < segment.NM7 < 0.01:
                            print("Medium")
                            logo_segments.append(segment)
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]


            #Large segments
            elif 0.5 < segment.NM1 < 1.28 and 0.24 < segment.NM2 < 1.6 and 1.6e-07 <= segment.NM3 < 0.0098 and 0.0063 < segment.NM7 < 0.0099:
                            print("Long")
                            logo_segments.append(segment)
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]

            else:
                for point in segment.segment:
                    self.blue_colors[point[0], point[1]] = [0, 0, 0]



        return letters, logo_segments

    def show_photo(self):
        # cv2.imshow('image', self.red_colors)
        cv2.imwrite('final/red.jpg', self.red_colors)
        cv2.imwrite('final/blue1.jpg', self.blue_colors)
        cv2.imwrite('final/marked.jpg', self.photo)


        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('messigray.png',self.photo)
            cv2.destroyAllWindows()


ciscoRecognizer = CiscoRecognizer(img)
ciscoRecognizer.find_logo()

















































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