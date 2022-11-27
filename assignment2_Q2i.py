# lane detection
# blur > canny > Roi (crop canny) > collect hough lines > display lines

import cv2
import numpy as np


def canny(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgCanny = cv2.Canny(imgBlur, 25,100)
    return imgCanny


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygons = np.array([[(200,height),(550,290),(1100,height)]])
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    # print('display_lines: ',lines)
    # filter empty elements in lines
    filterLines = [[]]
    # print(len(lines[0]))
    for i in lines[0]:
        # print(i)
        if len(i) > 0:
            filterLines[0].append(i)
    # print(filterLines)
    lines = filterLines
    if lines is not None:
        for line in lines:
            # try:
                # print(line)
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1),(x2, y2),(255,255,0),10)
            # except:
            #     continue
    return line_image


def average_slope_intercept(img, lines):
    leftFit = []
    rightFit = []
    leftLine = []
    rightLine = []

    if lines is None:
        return None
    for line in lines:
        # print(line)
        # if line is None:
        #     continue
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope, intercept = fit
            # print(fit)
            if slope < 0:
                leftFit.append((slope, intercept))
            else:
                rightFit.append((slope,intercept))

    if len(leftFit) > 0:
        leftFitAvg = np.average(leftFit,axis=0)
        leftLine = create_points(img, leftFitAvg)
    if len(rightFit) > 0:
        rightFitAvg = np.average(rightFit, axis=0)
        rightLine = create_points(img, rightFitAvg)
    # print(f"left {leftFitAvg}\nrigtht {rightFitAvg}")

    avgLines = [[leftLine,rightLine]]
    return avgLines


def create_points(img, line_parameters):
    slope, intercept = line_parameters
    y1 = int(img.shape[0])
    y2 = int(y1*(2/3))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


video = cv2.VideoCapture("Resources/test2.mp4")
prevLeft, prevRight = [], []
while True:
    ret, frame = video.read()
    Canny = canny(frame)
    croppedCanny = region_of_interest(Canny)
    lines = cv2.HoughLinesP(croppedCanny,2 ,np.pi/180, 100, np.array([]), minLineLength=5, maxLineGap=750)
    lineImg = display_lines(frame, lines)
    averageLines = average_slope_intercept(frame, lines)

    # if left line or right line is None, use corresponding line from previous frame
    if len(averageLines[0][0]) != 0:
        prevLeft = averageLines[0][0]

    if len(averageLines[0][1]) != 0:
        prevRight = averageLines[0][1]

    if len(averageLines[0][0]) == 0:
        averageLines[0][0] = prevLeft

    if len(averageLines[0][1]) == 0:
        averageLines[0][1] = prevRight
    # print(averageLines)
    averageLinesImage = display_lines(frame, averageLines)
    combinedImg = cv2.addWeighted(frame, 0.9, averageLinesImage, 1.0, 1)

    # cv2.imshow("lane detect", frame)
    # cv2.imshow("Canny", Canny)
    cv2.imshow("crop", croppedCanny)
    cv2.imshow("line image", lineImg)
    # cv2.imshow("average lines image", averageLinesImage)
    cv2.imshow("combined image", combinedImg)
    if cv2.waitKey(10) & 0xff == 27:
        break

video.release()
cv2.destroyAllWindows()
