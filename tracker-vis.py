import cv2
import numpy as np

video_path = 'videoplayback.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # optional resize
    # frame = cv2.resize(frame, (960, 540))

    # convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # white-ish color range for baseball
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # light dilation and blur to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # find contours (external only)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter by area, circularity, and radius
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area > 5 and perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.80:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if 2 < radius < 30:
                    center = (int(x), int(y))
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 1)
                    cv2.putText(frame, "", (center[0]+5, center[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # second contour check with hierarchy info
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area < 2 or perimeter == 0:
                continue

            has_hole = hierarchy[0][i][2] != -1
            if has_hole:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.75:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < 2 or radius > 30:
                continue

            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 0), 1)
            cv2.putText(frame, "", (center[0]+5, center[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # show output
    cv2.imshow("Baseball Tracking", frame)
    cv2.imshow("Mask", mask)  # optional debug view

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
