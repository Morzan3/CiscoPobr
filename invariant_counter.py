import math

class InvariantCounter:
    def __init__(self, segment):
        self.segment = segment

    def calculate_m(self, p, q):
        result = 0
        for point in self.segment:
            result += math.pow(point[0], p) * math.pow(point[1], q)

        return result

    def calculate_needed_invariants(self):
        segment_area = len(self.segment)
        m01 = self.calculate_m(0, 1)
        m10 = self.calculate_m(1, 0)
        m20 = self.calculate_m(2, 0)
        m02 = self.calculate_m(0, 2)
        m30 = self.calculate_m(3, 0)
        m03 = self.calculate_m(0, 3)
        m21 = self.calculate_m(2, 1)
        m11 = self.calculate_m(1, 1)
        m12 = self.calculate_m(1, 2)

        self.center_i = m10 / segment_area;
        self.center_j = m01 / segment_area;

        M01 = m01 - (m01 / segment_area) * segment_area;
        M10 = m10 - (m10 / segment_area) * segment_area;
        M11 = m11 - m10 * m01 / segment_area;
        M20 = m20 - m10 * m10 / segment_area;
        M02 = m02 - m01 * m01 / segment_area;
        M21 = m21 - 2 * m11 * self.center_i - m20 * self.center_j + 2 * m01 * self.center_i * self.center_i;
        M12 = m12 - 2 * m11 * self.center_j - m02 * self.center_i + 2 * m10 * self.center_j * self.center_j;
        M30 = m30 - 3 * m20 * self.center_i + 2 * m10 * self.center_i * self.center_i;
        M03 = m03 - 3 * m02 * self.center_j + 2 * m01 * self.center_j * self.center_j;

        self.NM1 = (M20 + M02) / pow(segment_area, 2);
        self.NM2 = ((M20 - M02) * (M20 - M02) + 4 * M11 * M11) / pow(segment_area, 4);
        self.NM3 = ((M30 - 3 * M12) * (M30 - 3 * M12) + (3 * M21 - M03) * (3 * M21 - M03)) / pow(segment_area, 5);
        self.NM7 = (M20 * M02 - M11 * M11) / pow(segment_area, 4);