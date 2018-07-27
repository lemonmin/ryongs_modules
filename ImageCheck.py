import cv2
import numpy as np

'''
targetImage가 baseImage에 포함되어있는지를 확인하는 함수

# 주의할 점 #
1) targetImage가 사이즈까지 완전히 동일해야 제대로 판단 가능함
2) baseImage가 targetImage보다 크거나 적어도 같아야 함

# Parameter #
1) baseImgPath : base 이미지가 저장되어 있는 경로
2) targetImgPath : target 이미지가 저장되어 있는 경로
3) percentage : 허용 일치율 하한 (98프로 이상의 일치율일 경우에만 True를 return 하려면 98 을 입력)
'''
def checkImageIncluded(baseImgPath, targetImgPath, percentage):
    baseImg = cv2.imread(baseImgPath, cv2.IMREAD_UNCHANGED)
    targetImg = cv2.imread(targetImgPath, cv2.IMREAD_UNCHANGED)

    baseImgCopy = baseImg.copy()
    # cv2에서 제공하는 matchTemplate 함수 == iconImg와 일치도가 가장 높은 위치를 return
    # res는 matchTemplate 필셀 연산 적용 결과가 담긴 이미지 배열
    # 적용 알고리즘에 따라 일치도가 높은 픽셀일수록 큰 값 혹은 작은 값이 담김
    res = cv2.matchTemplate(baseImgCopy, targetImg, cv2.TM_CCOEFF_NORMED)

    # minMaxLoc을 통해 가장 작은 값, 큰 값의, 그리고 그 위치를 가져옴
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # TM_CCOEFF_NORMED 알고리즘의 경우 가장 일치도가 높은 위치일수록 큰 값을 가지므로 max_val, max_loc을 선택
    # cv2.rectangle을 이용해 찾은 위치에 빨간 사각형으로 표기
    cv2.rectangle(baseImgCopy, max_loc, (max_loc[0] + 100, max_loc[1] + 110), (0,0,255), 2)

    cv2.imshow("baseImgCopy",baseImgCopy)
    cv2.waitKey(0)


    # loc = (array([81, 81, 81], dtype=int64), array([730, 731, 732], dtype=int64))
    # 픽셀 연산 결과 값(일치도)이 0.98보다 큰 좌표들의 정보가 (y좌표, x좌표) 형태로 반환됨
    loc = np.where(res > percentage/100)
    print("loc = ",loc)

    # zip연산을 통해서 (x좌료, y좌표) 형태로 값들을 가져옴
    # 만약 0.98 이상의 일치도를 가지는 좌표가 하나도 없을 경우엔 target 이미지가 base 이미지에 포함되지 않는다고 판단하여
    # return False
    count = 0
    for i in zip(*loc[::-1]):
        print("location = ",i)
        count += 1
    if count>0:
        return True
    else:
        return False

# need to test
def get_matchTemplate(baseImgPath, targetImgPath):
    baseImg = cv2.imread(baseImgPath, cv2.IMREAD_UNCHANGED)
    targetImg = cv2.imread(targetImgPath, cv2.IMREAD_UNCHANGED)

    baseImgCopy = baseImg.copy()
    res = cv2.matchTemplate(baseImgCopy, targetImg, cv2.TM_CCOEFF_NORMED)
    return res

def checkImageIncluded(baseImgPath, targetImgPath, percentage):
    res = get_matchTemplate(baseImgPath, targetImgPath)
    loc = np.where(res > percentage/100)

    count = 0
    for i in zip(*loc[::-1]):
        print("location = ",i)
        count += 1
    if count>0:
        return True
    else:
        return False

def get_img_location(baseImgPath, targetImgPath):
    res = get_matchTemplate(baseImgPath, targetImgPath)
    baseImg = cv2.imread(baseImgPath, cv2.IMREAD_UNCHANGED)
    w, h, _ = baseImg.shape[:2]

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    cv2.rectangle(baseImgCopy, max_loc, (max_loc[0] + 100, max_loc[1] + 110), (0,0,255), 2)
    return (max_loc[0] + w//2, max_loc[1] + 50 + h//2)


'''
TO DO:
이미지 피라미드 적용해서, 동일한 이미지가 다른 사이즈로 포함되어 있는 경우도 찾기
'''

print("result == ",checkImageIncluded("./rsc/base.png","./rsc/target.png",98))
