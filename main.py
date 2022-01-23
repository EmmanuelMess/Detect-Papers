import glob
import cv2
import numpy as np


def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


def orderPoints(points):
    x, y = points[:, 0], points[:, 1]

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    angles = np.where((y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r))

    mask = np.argsort(angles)
    mask = np.roll(mask, 2)

    result = np.zeros_like(points)
    result[:, 0] = x[mask]
    result[:, 1] = y[mask]

    return result


def fourPointTransform(image, points):
    ordered = orderPoints(points)

    widthA = cv2.norm(ordered[2] - ordered[3])
    widthB = cv2.norm(ordered[1] - ordered[0])
    maxWidth = np.maximum(widthA, widthB).astype(int)

    heightA = cv2.norm(ordered[1] - ordered[2])
    heightB = cv2.norm(ordered[0] - ordered[3])
    maxHeight = np.maximum(heightA, heightB).astype(int)

    newPositions = np.asarray([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, newPositions)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def findSquares(image):
    THRESH = 20
    N = 11

    blurred = cv2.medianBlur(image, 9)
    gray0 = np.zeros((blurred.shape[0], blurred.shape[1]), dtype=np.uint8)

    squares = []

    for c in range(0, 3):

        cv2.mixChannels([blurred], [gray0], [c, 0])

        while not squares and N < 100:
            for l in range(N):
                if l == 0:
                    gray = cv2.Canny(gray0, 10, THRESH, 3)
                    gray = cv2.dilate(gray, np.ones((3, 3)), (-1, -1))
                else:
                    _, gray = cv2.threshold(gray0, (l + 1) * 255 / N, 255, 0)

                contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)

                    if approx.shape[0] == 4 \
                            and np.fabs(cv2.contourArea(approx)) > 1000 \
                            and cv2.isContourConvex(approx):
                        maxCosine = 0
                        for j in range(2, 5):
                            approx = approx.reshape((-1, 2))
                            cosine = np.fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                            maxCosine = np.fmax(maxCosine, cosine)

                        if maxCosine < 0.2:
                            squares.append(approx)

            N += 10

    return squares


def removeShadows(image):
    def operate(channel):
        dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        channel = cv2.dilate(channel, dilationKernel)
        blurredChannel = cv2.medianBlur(channel, 21)
        differenceChannel = cv2.absdiff(b, blurredChannel)
        differenceChannel = cv2.subtract(255, differenceChannel)
        differenceChannel = cv2.normalize(differenceChannel, channel, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return differenceChannel

    b, g, r = cv2.split(image)
    b, g, r = operate(b), operate(g), operate(r)

    return cv2.merge(np.array([b, g, r]))


def intelliResize(image):
    PROCESS_WIDTH = 512
    PROCESS_HEIGHT = 910

    width = image.shape[1]
    height = image.shape[0]

    if width < PROCESS_WIDTH or height < PROCESS_HEIGHT:
        newWidth, newHeight = width, height
    elif height < width:
        newWidth, newHeight = PROCESS_HEIGHT, PROCESS_WIDTH
    else:
        newWidth, newHeight = PROCESS_WIDTH, PROCESS_HEIGHT

    ratioWidth = width / newWidth
    ratioHeight = height / newHeight

    return cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_AREA), ratioWidth, ratioHeight


WINDOW_NAME = "Square Detection Demo"


def main():
    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.resizeWindow(WINDOW_NAME, 1000, 1000)

    for filename in glob.glob("assets/*.jpg"):
        image = cv2.imread(filename)

        resizedImage, ratioWidth, ratioHeight = intelliResize(image)

        squares = findSquares(resizedImage.copy())

        list.sort(squares, key=cv2.contourArea)

        unresizedSquares = []

        for square in squares:
            unresizedSquare = []
            for i in square:
                unresizedSquare.append([i[0] * ratioWidth, i[1] * ratioHeight])
            unresizedSquares.append(unresizedSquare)

        unresizedSquares = np.asarray(unresizedSquares, dtype=np.float32)
        drawn = cv2.polylines(image.copy(), np.int32(unresizedSquares), True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, drawn)

        if True:
            number = 0

            for unresizedSquare in unresizedSquares:
                if number > 1:
                    break

                wndname1 = "Square Cut:" + str(number)
                number += 1

                cut = image.copy()

                cut = fourPointTransform(cut, unresizedSquare)

                cv2.namedWindow(wndname1)
                cv2.resizeWindow(wndname1, 1000, 1000)

                noShadows = removeShadows(cut)

                cv2.imshow(wndname1, noShadows)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
