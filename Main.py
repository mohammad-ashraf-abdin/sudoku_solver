from PreProccessing.Tools import *
from KNN.KNN_Class import *
from imutils.video import WebcamVideoStream
import cv2
from Solver import sudukoSolver

heightImg = 450
widthImg = 450
arr = [3, 3, 3, 3, 0, 1, 0, 0, 2, 2, 0, 1, 0, 3, 0, 6, 0, 4, 0, 0, 0, 2, 0, 4, 0, 0, 0, 8, 0, 9, 0, 0, 0, 1, 0, 6, 0, 6,
       0, 0, 0, 0, 0, 5, 0, 7, 0, 2, 0, 0, 0, 4, 0, 9, 0, 0, 0, 5, 0, 9, 0, 0, 0, 9, 0, 4, 0, 8, 0, 7, 0, 5, 6, 0, 0, 1,
       0, 7, 0, 0, 3]
x = np.array([[0,0], [0,0],[0,0],[0,0]], np.int32)
vs = WebcamVideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = cv2.resize(frame, (widthImg, heightImg))
    if(len(frame)<3):
        gray(frame)


    else:
        imgThreshold = preProcess(frame)
        imgContours = frame.copy()
        imgBigContour = frame.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

        biggest, maxArea = biggestContour(contours)

        if biggest.size != 0:
            biggest = reorder(biggest)

            cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
            imgWarpColored = cv2.warpPerspective(frame, matrix, (widthImg, heightImg))
            f=imgWarpColored.copy()
            imgDetectedDigits = frame.copy()
            imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

            imgSolvedDigits = np.zeros((450,450,3), np.uint8)
            boxes = splitBoxes(imgWarpColored)
            result = []

            q = ''
            itr = 0
            for item in boxes:
                    imgThreshh = cv2.adaptiveThreshold(item[7:45, 10:45],
                                                       255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV,
                                                       11,
                                                       2)
                    ncontours, hierarchyy = cv2.findContours(imgThreshh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(ncontours) > 30:
                        result.append(0)
                    else:
                        kernal = np.ones((2,2),np.uint8)
                        erosion = cv2.erode(item[7:45, 10:45],kernal,iterations=1)
                        q = main(erosion)
                        if q.isdigit():
                            result.append(int(q))
                        else:
                            result.append(0)
                    x = biggest
                    arr = result
            imgDetectedDigits = displayNumbers(imgDetectedDigits, result, color=(255, 0, 255))
            numbers = np.asarray(result)
            posArray = np.where(numbers > 0, 0, 1)
            board = np.array_split(numbers, 9)
            try:
                if Tools.validboard(board):
                    sudukoSolver.solve(board)
            except:
                pass
            flatList = []
            for sublist in board:
                for item in sublist:
                    flatList.append(item)
            solvedNumbers = flatList * posArray
            imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
            pts2 = np.float32(biggest)
            pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
            inv_perspective = cv2.addWeighted(imgInvWarpColored, 0.9, frame, 0.9, 1)
            imgDetectedDigits = drawGrid(imgDetectedDigits)
            imgSolvedDigits = drawGrid(imgSolvedDigits)
            cv2.imshow('img', frame)
            # cv2.imshow('thr', imgThreshold)
            # cv2.imshow('imgcou', imgContours)
            cv2.imshow('imgbigcount', imgBigContour)
            cv2.imshow('imgw', imgWarpColored)
            # cv2.imshow('imgDetectedDigits', imgDetectedDigits)
            cv2.imshow('img', imgSolvedDigits)
            # cv2.imshow('solkk', imgInvWarpColored)
            cv2.imshow('solk', inv_perspective)

        else:
            print("No Sudoku Found")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vs.release()
cv2.destroyAllWindows()