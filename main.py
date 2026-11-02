import cv2
import numpy as np

ref = cv2.imread('reference.jpg', cv2.IMREAD_GRAYSCALE)
test = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

if ref is None or test is None:
    print("Error: Photos folder kulla irukaa nu check pannunga!")
    exit()

ref = cv2.resize(ref, (500, 500))
test = cv2.resize(test, (500, 500))

diff = cv2.absdiff(cv2.GaussianBlur(ref, (5, 5), 0), cv2.GaussianBlur(test, (5, 5), 0))
_, thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_img = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
defect_count = 0

for c in contours:
    if cv2.contourArea(c) > 10:
        defect_count += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

status = f"STATUS: FAIL ({defect_count} Defects)" if defect_count > 0 else "STATUS: PASS (Good)"
color = (0, 0, 255) if defect_count > 0 else (0, 255, 0)
cv2.putText(output_img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("Inspection Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()