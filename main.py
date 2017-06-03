import cv2
import numpy as np
import math
import configparser
import json
import os
from random import randint
from invariant_counter import InvariantCounter

config = configparser.ConfigParser()
config.read('config.ini')

MIN_LETTER_SIZE = config.getint('Segments', 'MinLetterSegmentSize')
MAX_LETTER_SIZE = config.getint('Segments', 'MaxLetterSegmentSize')
MIN_LINE_SIZE = config.getint('Segments', 'MinLineSegmentSize')
MAX_LINE_SIZE = config.getint('Segments', 'MaxLineSegmentSize')
CHECK_LETTER_TRESHOLD = config.getint('Segments', 'LetterCheckTreshold')

class CiscoRecognizer:
    def __init__(self):
        photo_filepath = config.get('Photo', 'FilePath')
        if not os.path.isfile(photo_filepath):
            print("Invalid file path")
            exit(0)
        try:
            self.photo = cv2.imread(photo_filepath)
        except:
            print("Error while loading the photo")
            exit(0)

        self.min_blue = np.array(json.loads(config.get('Colors', 'MinBlueValues')))
        self.max_blue = np.array(json.loads(config.get('Colors', 'MaxBlueValues')))

        self.min_red = np.array(json.loads(config.get('Colors', 'MinRedValues')))
        self.max_red = np.array(json.loads(config.get('Colors', 'MaxRedValues')))
        self.height, self.width, self.depth = self.photo.shape
        self.red_colors = np.zeros((self.height, self.width, 3), np.uint8)
        self.blue_colors = np.zeros((self.height, self.width, 3), np.uint8)



    def find_logo(self):
        self.convert_to_hsv()
        self.separate_colors()
        self.calculate_photo_segments_invariant()
        letter_segments, logo_segments = self.find_all_letter_and_logo_segments()
        self.group_up_segments(letter_segments, logo_segments)
        self.save_photos()

    def convert_to_hsv(self):
        # self.hsv_photo = cv2.cvtColor(self.photo, cv2.COLOR_BGR2HSV)

        self.hsv_photo = np.zeros((self.height, self.width, 3), np.uint8)
        for x in range(self.height):
            for y in range(self.width):
                h, s, v = self.bgrToHsc(self.photo[x][y])
                self.hsv_photo[x,y][0] = h
                self.hsv_photo[x,y][1] = s
                self.hsv_photo[x,y][2] = v

    def bgrToHsc(self, point):
        b = point[0]
        g = point[1]
        r = point[2]
        max_color_value = max(point)
        min_color_value = min(point)
        v = max_color_value
        if min_color_value == max_color_value:
            return 0.0, 0.0, v
        s = (max_color_value - min_color_value) / max_color_value
        max_min = max_color_value - min_color_value
        rc = (max_color_value - r) / max_min
        gc = (max_color_value - g) / max_min
        bc = (max_color_value - b) / max_min
        if b == max_color_value:
            h = 4.0 + gc - rc
        elif g == max_color_value:
            h = 2.0 + rc - bc
        else:
            h = bc - gc

        h = (h / 6.0) % 1.0

        h = int(h*180)
        s = int(s*255)
        v = int(v)
        return (h, s, v)


    def is_red(self, point):
        return True if ((point[0] <= self.max_red[0]) or (point[0] >= self.min_red[0])) \
                    and (self.min_red[1] < point[1] <= self.max_red[1]) \
                    and (self.min_red[2] < point[2] <= self.max_red[2]) \
                else False

    def is_blue(self, point):
        return True if (self.min_blue[0] < point[0] <= self.max_blue[0]) \
                    and (self.min_blue[1] < point[1] <= self.max_blue[1]) \
                    and (self.min_blue[2] < point[2] <= self.max_blue[2]) \
            else False


    def is_pixel_of_given_color(self, pixel, color):
        return True if pixel[0] == color[2] and pixel[1] == color[1] and pixel[2] == color[0] else False

    def separate_colors(self):
        """
        Function separates the original photo into 2 separate photos based on the color boarded values
        """
        for x in range(self.height):
            for y in range(self.width):
                try:
                    if self.is_blue(self.hsv_photo[x, y]):
                        self.red_colors[x, y] = [0, 0, 0]
                        self.blue_colors[x, y] = [255, 255, 255]

                    elif self.is_red(self.hsv_photo[x, y]):
                        self.red_colors[x, y] = [255, 255, 255]
                        self.blue_colors[x, y] = [0, 0, 0]

                    else:
                        self.red_colors[x, y] = [0, 0, 0]
                        self.blue_colors[x, y] = [0, 0, 0]

                except IndexError:
                    continue


    def group_up_segments(self, letter_segments, logo_segments):
        """
        Function grouping the logo segments to each 's' segment
        """
        s_segments = [(x, y) for x, y in letter_segments if x == 's']
        rest = [(x, y) for x, y in letter_segments if x != 's']

        grouped_letter_segments = []
        for segment in s_segments:
            grouped_letter_segment = self.add_nearest_segments(segment, rest)
            if grouped_letter_segment == None:
                continue
            grouped_letter_segments.append(grouped_letter_segment)

        self.assign_logo_to_grouped_letters(grouped_letter_segments, logo_segments)

    def assign_logo_to_grouped_letters(self, group_letters, logo_segments):
        """
        Function is assignign logo segments to each letter group
        """
        for letter_group in group_letters:
            self.assign_logo_segments_to_letter_group(letter_group, logo_segments)


    def assign_logo_segments_to_letter_group(self,letter_group, logo_segments):
        """
        Function deciding in which direction to look for the logo
        """
        x_min, x_max, y_min, y_max = self.get_segment_center_boarder_points(letter_group)
        if (y_max - y_min) > (x_max - x_min):
            self.look_for_logo_up_or_down(letter_group, logo_segments)
        else:
            self.look_for_logo_left_or_right(letter_group, logo_segments)


    def look_for_logo_up_or_down(self, letter_group, logo_segments):
        """
        Function deciding in which direction to look for the logo
        """
        s_letter = None
        o_letter = None
        for letter in letter_group:
            if letter[0] == 's':
                s_letter = letter
            elif letter[0] == 'o':
                o_letter = letter

        if s_letter[1].center_j < o_letter[1].center_j:
            self.look_for_logo_in_direction(letter_group, logo_segments, 'u')
        else:
            self.look_for_logo_in_direction(letter_group, logo_segments, 'd')

    def look_for_logo_left_or_right(self, letter_group, logo_segments):
        """
        Function deciding in which direction to look for the logo
        """
        s_letter = None
        o_letter = None
        for letter in letter_group:
            if letter[0] == 's':
                s_letter = letter
            elif letter[0] == 'o':
                o_letter = letter

        if s_letter == None or o_letter == None:
            print("group marked as noise")
            return

        if s_letter[1].center_i > o_letter[1].center_i:
            self.look_for_logo_in_direction(letter_group, logo_segments, 'l')
        else:
            self.look_for_logo_in_direction(letter_group, logo_segments, 'r')


    def look_for_logo_in_direction(self, letter_group, logo_segments, direction):
        """
        Function is filtering possible segment list to the ones located in specific direction and adding the closest
        to the groupped letter segments.
        """
        s_letter = [(x,y)for x,y in letter_group if x == 's'][0]
        direction_segments = []
        if direction == 'u':
            direction_segments = [(x,y) for x,y in logo_segments if y.center_i < s_letter[1].center_i]
        elif direction == 'd':
            direction_segments = [(x, y) for x, y in logo_segments if y.center_i > s_letter[1].center_i]
        elif direction == 'l':
            direction_segments = [(x, y) for x, y in logo_segments if y.center_j < s_letter[1].center_j]
        elif direction == 'r':
            direction_segments = [(x, y) for x, y in logo_segments if y.center_j > s_letter[1].center_j]

        if len(direction_segments) < 9:
            print("Logo segments identified as noise")
            self.mark_grouped_segments(letter_group)
            return

        closest_segments = []
        for segment in direction_segments:
            distance = self.calculate_distance_between_segments(s_letter, segment)
            closest_segments.append((distance, segment))

        sorted_by_distance = sorted(closest_segments, key=lambda x: x[0])
        sorted_by_distance = [y for x,y in sorted_by_distance]
        letter_group = letter_group + sorted_by_distance[:9]
        self.mark_grouped_segments(letter_group)



    def mark_grouped_segments(self, group_of_segments):
        """
        Function is marking the group_of_segments on the photograph
        """
        x_min, x_max, y_min, y_max = self.get_segment_boarder_points(group_of_segments)
        for x in range(x_min, x_max):
            self.photo[x][y_min] = [0, 0, 255]
        for x in range(x_min, x_max):
            self.photo[x][y_max] = [0, 0, 255]
        for y in range(y_min, y_max):
            self.photo[x_min][y] = [0, 0, 255]
        for y in range(y_min, y_max):
            self.photo[x_max][y] = [0, 0, 255]

    def get_segment_boarder_points(self, group):
        """
        Function is finding the extreme values of points in given group
        """
        x_min = self.height
        y_min = self.width
        x_max = 0
        y_max = 0

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

    def get_segment_center_boarder_points(self, group):
        """
        Function finding the extreme values of center point segments coordinates
        """
        xes = [segment[1].center_i for segment in group]
        yes = [segment[1].center_j for segment in group]

        min_x = min(xes)
        max_x = max(xes)
        min_y = min(yes)
        max_y = max(yes)

        return min_x, max_x, min_y, max_y


    def add_nearest_segments(self, s_segment, rest_segments):
        """
        Function is finding the nearest letters segments to the given 's' segment
        """
        i_segments = [(x, y) for x, y in rest_segments if x == 'i']
        c_segments = [(x, y) for x, y in rest_segments if x == 'c']
        o_segments = [(x, y) for x, y in rest_segments if x == 'o']

        if len(i_segments) == 0 or len(c_segments) == 0 or len(o_segments) == 0:
            print("Not enought segments to qualifie for a logo")
            return None

        i_segment = self.find_nearest_segment(s_segment, i_segments)
        o_segment = self.find_nearest_segment(s_segment, o_segments)
        c_segment_first = self.find_nearest_segment(s_segment, c_segments)
        c_segments.remove(c_segment_first)
        c_segment_second = self.find_nearest_segment(s_segment, c_segments)

        if self.check_letter_segments([s_segment, i_segment, o_segment, c_segment_first, c_segment_second]):
            return [s_segment, i_segment, o_segment, c_segment_first, c_segment_second]
        else:
            return None

    def check_letter_segments(self, letters_to_check):
        """
        Function is checking the group of letter segments for substantial changes in the center point coordinate
        differences
        """
        xes = [y.center_i for x, y in letters_to_check]
        yes = [y.center_j for x, y in letters_to_check]

        x_diff = max(xes) - min(xes)
        y_diff = max(yes) - min(yes)

        smaller_diff = min(x_diff, y_diff)
        print(smaller_diff)
        return True if smaller_diff < CHECK_LETTER_TRESHOLD else False

    def find_nearest_segment(self, main_segment, rest):
        """
        Function finding the nearest segment to the main_segment out of list of given segments
        """
        nearest_segment = rest[0]

        for segment in rest:
            if self.calculate_distance_between_segments(main_segment, segment) < self.calculate_distance_between_segments(main_segment, nearest_segment):
                nearest_segment = segment

        return nearest_segment

    def calculate_distance_between_segments(self, first_segment, second_segment):
        """
        Function calculating distance between two segments based on their center points
        """
        return math.sqrt(np.power([first_segment[1].center_i - second_segment[1].center_i,
                                   first_segment[1].center_j - second_segment[1].center_j], 2).sum())

    def mark_and_add_segment(self, photo, segment_list, starting_point, segment_color):
        """
        Flood fill algorithm for discovering information about segment based on one of its points
        """
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
        """
        Function is analysing all the pixels in blue and red photos, gathering information about segments
        """
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
        """
        Function is filtering all the segments based on their size
        """
        filtered_list = []
        for segment in segment_list:
            if len(segment) > MAX_LETTER_SIZE or len(segment) < MIN_LETTER_SIZE:
                for point in segment:
                    self.red_colors[point[0], point[1]] = [0, 0, 0]
            else:
                filtered_list.append(segment)
        return filtered_list

    def delete_blue_segments(self, segment_list):
        """
        Function is filtering all the segments based on their size
        """
        filtered_list = []
        for segment in segment_list:
            if len(segment) > MAX_LINE_SIZE or len(segment) < MIN_LINE_SIZE:
                for point in segment:
                    self.blue_colors[point[0], point[1]] = [0, 0, 0]
            else:
                filtered_list.append(segment)
        return filtered_list

    def calculate_photo_segments_invariant(self):
        """
        Function is taking all segments and counts their invariants
        """
        red_segments, blue_segments = self.extract_segments(True)

        self.letter_segments = []
        self.logo_segments = []

        for red_segment in red_segments:
            invariant_counter = InvariantCounter(red_segment)
            invariant_counter.calculate_needed_invariants()
            if invariant_counter.NM1 != 0 and invariant_counter.NM2 != 0 and invariant_counter.NM7 !=0:
                self.letter_segments.append(invariant_counter)

        for blue_segment in blue_segments:
            invariant_counter = InvariantCounter(blue_segment)
            invariant_counter.calculate_needed_invariants()
            if invariant_counter.NM1 != 0 and invariant_counter.NM2 != 0 and invariant_counter.NM7 !=0:
                self.logo_segments.append(invariant_counter)



    def find_all_letter_and_logo_segments(self):
        """
        Function analyses red and blue segments and based on their invariants categorizes them.
        """
        letters = []
        logo_segments = []

        for segment in self.letter_segments:
            # find C
            if 0.28 < segment.NM1 < 0.39 and 0.014 < segment.NM2 < 0.03 and 0.005 < segment.NM3 < 0.014 and 0.018 < segment.NM7 < 0.03:
                letters.append(('c',segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find I
            elif 0.29 < segment.NM1 < 0.4 and 0.065 < segment.NM2 < 0.13 and 0.006 < segment.NM7 < 0.0073:
                letters.append(('i', segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find S
            elif 0.24 < segment.NM1 < 0.34 and 0.017 < segment.NM2 < 0.034 and 0.00002 < segment.NM3 < 0.00183 and 0.011 < segment.NM7 < 0.021:
                letters.append(('s', segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            # find 0
            elif 0.24 < segment.NM1 < 0.33 and 3.38e-06 < segment.NM2 < 0.0003 and 4.4e-08 < segment.NM3 < 6e-05 and 0.014 < segment.NM7 < 0.027:
                letters.append(('o', segment))
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [255, 2555, 255]

            else:
                for point in segment.segment:
                    self.red_colors[point[0], point[1]] = [0, 0, 0]


        for segment in self.logo_segments:
            # Smallest segments
            if 0.17 < segment.NM1 < 0.243 and 0.007 < segment.NM2 < 0.031 and 0 <= segment.NM3 < 0.0098 and 0.0064 < segment.NM7 < 0.009:
                            logo_segments.append(('small', segment))
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]


            #Medium segments
            elif 0.27 < segment.NM1 < 0.46 and 0.04 < segment.NM2 < 0.19 and 0 <= segment.NM3 < 0.008 and 0.006 < segment.NM7 < 0.01:
                            logo_segments.append(('medium',segment))
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]


            #Large segments
            elif 0.5 < segment.NM1 < 1.28 and 0.24 < segment.NM2 < 1.6 and 1.6e-08 <= segment.NM3 < 0.0098 and 0.0063 < segment.NM7 < 0.0099:
                            logo_segments.append(('long',segment))
                            for point in segment.segment:
                                self.blue_colors[point[0], point[1]] = [255, 2555, 255]

            else:
                for point in segment.segment:
                    self.blue_colors[point[0], point[1]] = [0, 0, 0]

        return letters, logo_segments

    def save_photos(self):
        """
        Function saves the photos to the disk
        """

        destination_folder = config.get('Photo', 'DestinationFolderFilepath')
        cv2.imwrite(destination_folder + 'red.jpg', self.red_colors)
        cv2.imwrite(destination_folder + 'blue.jpg', self.blue_colors)
        cv2.imwrite(destination_folder + 'marked.jpg', self.photo)

ciscoRecognizer = CiscoRecognizer()
ciscoRecognizer.find_logo()

