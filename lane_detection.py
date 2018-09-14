import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


pre_left_m = None
pre_left_b = None
pre_right_m = None
pre_right_b = None

class DetectLane:
    def __init__(self,):
        pass

    def draw_lines(self,img, lines, color=[0, 0, 255], thickness=10):
        global pre_left_m, pre_left_b, pre_right_m, pre_right_b
        prev_ratio = 0.0
        left_m = []
        left_b = []
        right_m = []
        right_b = []

        left_y_min = img.shape[0]
        right_y_min = img.shape[0]

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if (abs(x2 - x1) < 1e-3):
                        continue
                    m = (y2 - y1) / (x2 - x1)
                    b = y2 - m * x2

                    # Left lane
                    if (m <= -0.5 and m >= -0.9 and b >= img.shape[0] and x1 <= img.shape[1] / 2 and
                            x2 <= img.shape[1] / 2):
                        left_m.append(m)
                        left_b.append(b)

                    # Right lane
                    elif (m >= 0.5 and m <= 0.9 and m * img.shape[1] + b >= img.shape[0] and
                          x1 >= img.shape[1] / 2 and
                          x2 >= img.shape[1] / 2):
                        right_m.append(m)
                        right_b.append(b)

            if (len(left_m) > 0):
                left_avg_m = np.mean(left_m)
                left_avg_b = np.mean(left_b)
                if (pre_left_m is not None):
                    left_avg_m = prev_ratio * pre_left_m + (1 - prev_ratio) * left_avg_m
                    left_avg_b = prev_ratio * pre_left_b + (1 - prev_ratio) * left_avg_b
                left_x_min = (img.shape[0] * 3 / 5 - left_avg_b) / left_avg_m
                left_x_max = (img.shape[0] - left_avg_b) / left_avg_m
                cv2.line(img, (int(left_x_min), int(img.shape[0] * 3 / 5)),
                         (int(left_x_max), img.shape[0]), color, thickness)
                pre_left_m = left_avg_m
                pre_left_b = left_avg_b

            if (len(right_m) > 0):
                right_avg_m = np.mean(right_m)
                right_avg_b = np.mean(right_b)
                if (pre_right_m is not None):
                    right_avg_m = prev_ratio * pre_right_m + (1 - prev_ratio) * right_avg_m
                    right_avg_b = prev_ratio * pre_right_b + (1 - prev_ratio) * right_avg_b
                right_x_min = (img.shape[0] * 3 / 5 - right_avg_b) / right_avg_m
                right_x_max = (img.shape[0] - right_avg_b) / right_avg_m
                cv2.line(img, (int(right_x_min), int(img.shape[0] * 3 / 5)),
                         (int(right_x_max), img.shape[0]), color, thickness)
                pre_right_m = right_avg_m
                pre_right_b = right_avg_b


    def lane_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    def find_lanes(self,image):
        low_threshold = 50
        high_threshold = 150
        rho = 1
        theta = np.pi/180
        threshold = 15
        min_line_len = 20
        max_line_gap = 50
        kernel_size = 3

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur_img = cv2.GaussianBlur(hsv_img, (kernel_size, kernel_size), 0)

        #colur range to recognize
        yellow_min = np.array([65, 80, 80], np.uint8)

        yellow_max = np.array([105, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(blur_img, yellow_min, yellow_max)

        white_min = np.array([0, 0, 200], np.uint8)
        white_max = np.array([255, 80, 255], np.uint8)
        white_mask = cv2.inRange(blur_img, white_min, white_max)

        blur = cv2.bitwise_and(blur_img, blur_img, mask=cv2.bitwise_or(yellow_mask, white_mask))

        edges = cv2.Canny(blur, low_threshold, high_threshold)
        height = image.shape[0]
        width = image.shape[1]
        vertices = np.array([[(0,height),(width*2/5, height*3/5),
                              (width*3/5, height*3/5), (width,height)]], dtype=np.int32)

        mask = np.zeros_like(edges)
        if len(edges.shape) > 2:
            channel_count = edges.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        line_image = self.lane_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
        lines_edges = self.weighted_img(line_image, image)
        return lines_edges

    def weighted_img(self,img, initial_img, α=0.8, β=1.0, λ=0.0):
        return cv2.addWeighted(initial_img, α, img, β, λ)



# def main():
#     cap = cv2.VideoCapture('test_vid2.mp4')

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         lobj = DetectLane()
#         lines_edges = lobj.find_lanes(frame)
#         cv2.imshow('frame', lines_edges)


#         k = cv2.waitKey(5) & 0xFF
#         if k == 27:
#             break
#     cv2.destroyAllWindows()
#     cap.release()





# if __name__ == '__main__':
#     main()

#
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture('test_vid1.mp4')
#
# while True:
# 	_, frame = cap.read()
# 	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#
# 	lower_red = np.array([18,94,140], dtype=np.uint8)
# 	upper_red = np.array([40,255,255], dtype=np.uint8)
# 	mask = cv2.inRange(hsv, lower_red, upper_red)
# 	result = cv2.bitwise_and(frame, frame, mask=mask)
#
# 	blur = cv2.GaussianBlur(result, (15,15), 0 )
# 	cv2.imshow('frame',frame)
# 	cv2.imshow('Mask',mask)
# 	cv2.imshow('blur',blur)
#
#
# 	k = cv2.waitKey(5) & 0xFF
# 	if k == 27:
# 		break
# cv2.destroyAllWindows()
# cap.release()
