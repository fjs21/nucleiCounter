import cv2 as cv
import numpy as np
import argparse

class tryMorphology():

    def __init__(self):
        self.erosion_size = 0
        self.max_elem = 2
        self.max_kernel_size = 21
        self.max_iterations = 5

        self.title_trackbar_element_type = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
        self.title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
        self.title_trackbar_iteration = 'Iterations'
        self.title_erosion_window = 'Erosion Demo'
        self.title_dilatation_window = 'Dilation Demo'
        self.title_opening_window = 'Opening Demo'
        self.title_closing_window = 'Closing Demo'

    def tryErosion(self, src):
        self.src = src

        cv.namedWindow(self.title_erosion_window)
        cv.createTrackbar(self.title_trackbar_element_type, self.title_erosion_window , 0, self.max_elem, self.erosion)
        cv.createTrackbar(self.title_trackbar_kernel_size, self.title_erosion_window , 0, self.max_kernel_size, self.erosion)
        self.erosion(0)
        cv.waitKey()
        cv.destroyAllWindows()

    def tryDilation(self, src):
        self.src = src
    
        cv.namedWindow(self.title_dilatation_window)
        cv.createTrackbar(self.title_trackbar_element_type, self.title_dilatation_window , 0, self.max_elem, self.dilatation)
        cv.createTrackbar(self.title_trackbar_kernel_size, self.title_dilatation_window , 0, self.max_kernel_size, self.dilatation)
        self.dilatation(0)
        cv.waitKey()
        cv.destroyAllWindows()

    def tryOpening(self, src):
        self.src = src
    
        cv.namedWindow(self.title_opening_window)
        cv.createTrackbar(self.title_trackbar_element_type, self.title_opening_window , 0, self.max_elem, self.opening)
        cv.createTrackbar(self.title_trackbar_kernel_size, self.title_opening_window , 0, self.max_kernel_size, self.opening)
        cv.createTrackbar(self.title_trackbar_iteration, self.title_opening_window , 1, self.max_iterations, self.opening)
        self.opening(0)
        cv.waitKey()
        cv.destroyAllWindows()

    def tryClosing(self, src):
        self.src = src
    
        cv.namedWindow(self.title_opening_window)
        cv.createTrackbar(self.title_trackbar_element_type, self.title_opening_window , 0, self.max_elem, self.opening)
        cv.createTrackbar(self.title_trackbar_kernel_size, self.title_opening_window , 0, self.max_kernel_size, self.opening)
        cv.createTrackbar(self.title_trackbar_iteration, self.title_opening_window , 1, self.max_iterations, self.opening)
        self.opening(0)
        cv.waitKey()
        cv.destroyAllWindows()

    def erosion(self, val):
        self.src = src
    
        erosion_size = cv.getTrackbarPos(self.title_trackbar_kernel_size, self.title_erosion_window)
        erosion_type = 0
        val_type = cv.getTrackbarPos(self.title_trackbar_element_type, self.title_erosion_window)
        if val_type == 0:
            erosion_type = cv.MORPH_RECT
        elif val_type == 1:
            erosion_type = cv.MORPH_CROSS
        elif val_type == 2:
            erosion_type = cv.MORPH_ELLIPSE
        self.element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
        erosion_dst = cv.erode(self.src, self.element)
        cv.imshow(self.title_erosion_window, erosion_dst)

    def dilatation(self, val):
        dilatation_size = cv.getTrackbarPos(self.title_trackbar_kernel_size, self.title_dilatation_window)
        dilatation_type = 0
        val_type = cv.getTrackbarPos(self.title_trackbar_element_type, self.title_dilatation_window)
        if val_type == 0:
            dilatation_type = cv.MORPH_RECT
        elif val_type == 1:
            dilatation_type = cv.MORPH_CROSS
        elif val_type == 2:
            dilatation_type = cv.MORPH_ELLIPSE
        self.element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        dilatation_dst = cv.dilate(self.src, self.element)
        cv.imshow(self.title_dilatation_window, dilatation_dst)

    def opening(self, val):
        opening_size = cv.getTrackbarPos(self.title_trackbar_kernel_size, self.title_opening_window)
        opening_iterations = cv.getTrackbarPos(self.title_trackbar_iteration, self.title_opening_window)
        opening_type = 0
        val_type = cv.getTrackbarPos(self.title_trackbar_element_type, self.title_opening_window)
        if val_type == 0:
            opening_type = cv.MORPH_RECT
        elif val_type == 1:
            opening_type = cv.MORPH_CROSS
        elif val_type == 2:
            opening_type = cv.MORPH_ELLIPSE
        self.element = cv.getStructuringElement(opening_type, (2*opening_size + 1, 2*opening_size+1), (opening_size, opening_size))
        opening_dst = cv.morphologyEx(self.src, cv.MORPH_OPEN, self.element, iterations= opening_iterations)
        cv.imshow(self.title_opening_window, opening_dst)

    def closing(self, val):
        closing_size = cv.getTrackbarPos(self.title_trackbar_kernel_size, self.title_closing_window)
        closing_iterations = cv.getTrackbarPos(self.title_trackbar_iteration, self.title_closing_window)
        closing_type = 0
        val_type = cv.getTrackbarPos(self.title_trackbar_element_type, self.title_closing_window)
        if val_type == 0:
            closing_type = cv.MORPH_RECT
        elif val_type == 1:
            closing_type = cv.MORPH_CROSS
        elif val_type == 2:
            closing_type = cv.MORPH_ELLIPSE
        self.element = cv.getStructuringElement(closing_type, (2*closing_size + 1, 2*closing_size+1), (closing_size, closing_size))
        closing_dst = cv.morphologyEx(self.src, cv.MORPH_CLOSE, self.element, iterations= closing_iterations)
        cv.imshow(self.title_closing_window, closing_dst)

