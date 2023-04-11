from sklearn.cluster import DBSCAN
import numpy as np
import cv2


class Detect(object):
    def __init__(self, input):
        self.image = cv2.imread(input)

    def siftDetector(self):
        sift = cv2.SIFT_create()
        # sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def showSiftFeatures(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift_image = cv2.drawKeypoints(
            self.image, self.key_points, self.image.copy())
        return sift_image

    def locateForgery(self, eps=40, min_sample=2):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(
            self.descriptors)
        size = np.unique(clusters.labels_).shape[0]-1
        forgery = self.image.copy()
        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            print('No Forgery Found!!')
            return None
        if size == 0:
            size = 1
        cluster_list = [[] for i in range(size)]
        for idx in range(len(self.key_points)):
            if clusters.labels_[idx] != -1:
                cluster_list[clusters.labels_[idx]].append(
                    (int(self.key_points[idx].pt[0]), int(self.key_points[idx].pt[1])))
        for points in cluster_list:
            if len(points) > 1:
                for idx1 in range(1, len(points)):
                    # Green color in BGR
                    cv2.line(forgery, points[0], points[idx1], (0, 255, 0), 5)
                    # cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 5)
        return forgery
if __name__ == "__main__":
    detector = Detect('/home/hai/Downloads/Image_Manipulation_Detection_System_Python/input/CFA-Artifact Detection/5.jpg')
    key_points, descriptors = detector.siftDetector()
    sift_image = detector.showSiftFeatures()
    forgery_image = detector.locateForgery()

    # Thay đổi kích thước cửa sổ hiển thị
    cv2.namedWindow('Display Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display Window', 1600, 600)

    # Nối hai ảnh theo chiều ngang
    result_image = cv2.hconcat([sift_image, forgery_image])

    # Hiển thị ảnh đã nối trong cửa sổ hiển thị
    cv2.imshow('Display Window', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
