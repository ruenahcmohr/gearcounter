#!.venv/bin/python3
import random as rng
import sys

import cv2
import math
import numpy as np


class NoGearFoundError(Exception):
    def __init__(self, message):
        self.message = message


class Gear:

    def __init__(self, image, contour):
        self.image = image
        self.contour = contour
        if len(contour) < 10:
            raise NoGearFoundError("because its got less than 10 points...")

        center, radius = cv2.minEnclosingCircle(self.contour)
        self.center_x = int(center[0])
        self.center_y = int(center[1])
        self.radius = int(radius)

        if (self.radius < 15) or (self.radius > 972):
            raise NoGearFoundError(f"because its radius {self.radius} is too ... wrong...", )

        self.hull = cv2.convexHull(self.contour)
        convex_hull_length = cv2.arcLength(self.hull, True)
        circumference = math.pi * 2 * self.radius
        # the convex hull and the min enclosing circle should have the same circumference
        if convex_hull_length / circumference < 0.9 or convex_hull_length / circumference > 1.1:
            raise NoGearFoundError(
                f"ditching non circle length={cv2.arcLength(self.contour, True)} hull={cv2.arcLength(self.hull, True)} c={math.pi * 2 * self.radius}")

    # make greyscale waves of the inside edge of the circle.
    def __make_ring_waves(self):
        intenWave = []
        # image_gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = self.image[:, :, 2]  # red channel for white gears
        color = 255
        angles = np.linspace(-3.14, 3.14, 512)

        micro_grey = image_gray[
                     self.center_y - self.radius:self.center_y + self.radius,
                     self.center_x - self.radius:self.center_x + self.radius
                     ]

        mask = micro_grey.copy()

        # cv2.imshow('Binary image', micro_grey)

        p0 = (self.radius * 2 - 5, self.radius)  # middle
        mask[:] = 0
        cv2.circle(mask, (p0[0], p0[1]), 5, color, -1)  # make sample size mask

        maskBase = mask.sum()

        for a in angles:
            mask[:] = 0
            p1 = self.__rotate([p0], a, (self.radius, self.radius))[0]
            cv2.circle(mask, (int(p1[0]), int(p1[1])), 5, color, -1)  # circle along edge-5pix
            sample = cv2.bitwise_and(micro_grey, micro_grey, mask=mask)
            bri = sample.sum() / maskBase
            intenWave.append(bri * 100)

        # cv2.imshow('Binary image', sample)
        # cv2.waitKey(6000)

        return intenWave

    def __fft_tooth_counter(self, curve):
        fft_points = np.array(curve)
        # delta = []
        # for i in range(0, len(fft_points)-1):
        #     delta.append(fft_points[i+1] - fft_points[i])
        # delta.append(fft_points[0] - fft_points[-1])
        #
        # delta = 30 * np.array(delta)
        # plt.plot(list(range(0, len(delta))), delta)

        fft_result = np.fft.fft(fft_points)
        # ditch the positive fequency values (and the dc value)
        fft_result = fft_result[0:250]

        # print("Hi!", fft_points)
        # fft_result[0] = 0

        #    plt.plot(list(range(0, len(fft_points))), fft_points)
        #    plt.show()

        # reduce to amplitudes
        fft_result = abs(fft_result)
        fft_result[0] = 0
        fft_result[1] = 0
        fft_result[2] = 0
        fft_result[3] = 0
        fft_result[4] = 0

        #    plt.plot(list(range(0, len(fft_result))), fft_result)
        #    plt.show()

        # ignore low peaks
        peak = np.max(fft_result)
        if peak < 500:  # 1200):
            raise NoGearFoundError(f"low FFT amplitude: {peak}")

        print("max is: ", peak)

        peaks = [(i, fft_result[i]) for i, item in enumerate(fft_result) if item > peak * 0.5]
        peaks.sort(key=(lambda x: x[1]), reverse=True)
        if len(peaks) > 4:
            raise NoGearFoundError(f"too much noise {peaks}")

        # inverse fft of the first 4 highest components
        x = np.linspace(0, math.pi, num=1000, endpoint=False)
        y = np.cos(x * peaks[0][0]) * peaks[0][1]
        if len(peaks) > 1:
            y = y + np.cos(x * peaks[1][0]) * peaks[1][1]
        if len(peaks) > 2:
            y = y + np.cos(x * peaks[2][0]) * peaks[2][1]
        if len(peaks) > 3:
            y = y + np.cos(x * peaks[3][0]) * peaks[3][1]

        # find their number of zero-crossings
        # see https://kitchingroup.cheme.cmu.edu/blog/2013/02/27/Counting-roots/
        self.count = (np.sum(y[0:-2] * y[1:-1] < 0))

        #    print (">>> peaks are: ", fft_result[peaks] , "at", peaks, "<<<<")
        print(f">>> peaks are: {peaks}, teeths={self.count}")

        # indexes = list(range(0, len(fft_result)))
        # plt.plot(indexes[0:200], fft_result[0:200], [0, len(fft_result[0:200])], [peak, peak])
        # plt.plot(x, y)

        # plt.show()

    def count_tooths(self):
        curve = self.__make_ring_waves()
        self.__fft_tooth_counter(curve)

    # it looks like the lowest overhead way to do a rotation is with a port of the function from my C library...
    @staticmethod
    def __rotate(points, angle, center=(0, 0)):
        points_out = []
        for i, p in enumerate(points):
            c = np.cos(angle)
            s = np.sin(angle)
            x = center[0] + (p[0] - center[0]) * c - (p[1] - center[1]) * s
            y = center[1] + (p[0] - center[0]) * s + (p[1] - center[1]) * c
            points_out.append((x, y))
        return points_out

    def draw_count(self, drawing):
        font = cv2.FONT_HERSHEY_SIMPLEX

        color = (rng.randint(128, 256), rng.randint(10, 256), rng.randint(10, 256))
        p0 = (self.center_x, self.center_y)

        cv2.circle(drawing, p0, int(self.radius), color, 3)  # circle along edge
        cv2.circle(drawing, p0, 20, color, -1)  # filled circle in middle

        xf = (p0[0] < 1300)
        yf = (p0[1] < 980)

        p1 = np.add(p0, [self.radius + 60, 0])

        t1 = np.add(p0, [self.radius, 0])
        t2 = np.add(p0, [self.radius + 40, 10])
        t3 = np.add(p0, [self.radius + 40, -10])

        if [xf, yf] == [True, True]:  # go right down
            angle = rng.randint(0, 45)
        if [xf, yf] == [True, False]:  # go right up
            angle = rng.randint(315, 360)
        if [xf, yf] == [False, True]:  # go left down
            angle = rng.randint(135, 180)
        if [xf, yf] == [False, False]:  # go left up
            angle = rng.randint(180, 225)

        ps = self.__rotate([p1, t1, t2, t3], np.deg2rad(angle), (p0[0], p0[1]))
        p1 = ps[0]

        triangle = np.array(
            [(int(ps[1][0]), int(ps[1][1])), (int(ps[2][0]), int(ps[2][1])), (int(ps[3][0]), int(ps[3][1]))])
        cv2.drawContours(drawing, [triangle], 0, color, -1)

        # cv2.drawContours(drawing, [self.contour], 0, color, -1)

        p0 = (int(p0[0]), int(p0[1]))
        p1 = (int(p1[0]), int(p1[1]))

        cv2.line(drawing, p0, p1, color, 3)  # takeoff

        if xf:
            x = p1[0] + 50
        else:
            x = p1[0] - 50

        cv2.line(drawing, p1, (x, p1[1]), color, 3)  # takeoff tail

        if xf == 0: x = x - 100

        cv2.putText(drawing, str(self.count), (x, p1[1] + 30), font, 2.5, color, 4, cv2.LINE_AA)  # gear count

        return drawing


def get_contours(image):
    # image_gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image[:, :, 2]  # red channel for white gears
    image_gray = cv2.GaussianBlur(image_gray, (15, 15), 0)

    # cv2.imshow('Binary image', cv2.resize(image_gray, (1024,768)))
    # cv2.waitKey(6000)

    _, image_gray = cv2.threshold(image_gray, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)

    # cv2.imshow('Binary image', cv2.resize(image_gray, (1024,768)))
    # cv2.waitKey(6000)

    contours, _ = cv2.findContours(image_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "contours found")
    return contours


def sort_contours(image, contours):
    # break down the contours for us into circles and polygons
    gears = []

    for i, c in enumerate(contours):
        try:
            gears.append(Gear(image, c))
        except NoGearFoundError as error:
            print(f"Ditching contour {i}: {error.message}")

    # print (len(centers), "Items remaining")
    return gears


def main():
    print("###################################################")
    np.set_printoptions(suppress=True)
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "foo.jpg"

    image = cv2.imread(file_name)
    contours = get_contours(image)

    gears = sort_contours(image, contours)
    for gear in gears:
        try:
            gear.count_tooths()
            image = gear.draw_count(image)
        except NoGearFoundError as error:
            print(f"ditching profile: {error.message}")
            continue
    #
    #   # radii, centers, polar    = unrollContours(radii, centers, contours)
    #   Iradii, Icens, Icounts   = makeRingWaves(image, radii, centers)
    #
    # #  Aradii, Acens, Acounts   = amplitudeToothCounter(radii, centers, polar)
    # #  Fradii, Fcens, Fcounts   = fftToothCounter(radii, centers, polar)
    #   Fradii, Fcens, Fcounts   = fftToothCounter(radii, centers, Icounts)
    #
    #   # image = drawCounts(image, Aradii, Acens, Acounts)
    #   image = drawCounts(image, Fradii, Fcens, Fcounts)

    cv2.imshow('Binary image', cv2.resize(image, (1024, 768)))
    cv2.imwrite(f'counts_{file_name.split("/")[-1]}', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


main()
