from lane_detection import DetectLane
import cv2

def main():
    cap = cv2.VideoCapture('test_vid2.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lobj = DetectLane()
        lines_edges = lobj.find_lanes(frame)
        cv2.imshow('frame', lines_edges)


        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()





if __name__ == '__main__':
    main()
