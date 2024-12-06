import cv2
import numpy as np
import matplotlib.pyplot as plt

def template_matching(query_img, target_img):
    # Применение метода cv2.matchTemplate
    result = cv2.matchTemplate(target_img, query_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Вычисление координат для рамки
    h, w = query_img.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Копируем изображение и рисуем прямоугольник
    output = target_img.copy()
    cv2.rectangle(output, top_left, bottom_right, (0, 0, 255), 2)

    return output

def sift_matching(query_img, target_img):
    # Инициализация SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_img, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)

    # Сопоставление дескрипторов методом FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Применение ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Определение координат сопоставления
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Вычисление гомографии
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = query_img.shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        # Копируем изображение и рисуем рамку
        output = target_img.copy()
        dst = np.int32(dst)
        cv2.polylines(output, [dst], True, (0, 0, 255), 2)
    else:
        output = target_img.copy()
        print("Not enough matches are found.")

    return output

def plot_results(query_img, template_result, sift_result):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Query Image")
    axs[1].imshow(cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Template Matching")
    axs[2].imshow(cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB))
    axs[2].set_title("SIFT Matching")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    query_image_path = "paris/tower_crop.jpg"
    target_image_path = "paris/other.jpg"

    query_img = cv2.imread(query_image_path)
    target_img = cv2.imread(target_image_path)

    template_result = template_matching(query_img, target_img)
    sift_result = sift_matching(query_img, target_img)

    plot_results(query_img, template_result, sift_result)
