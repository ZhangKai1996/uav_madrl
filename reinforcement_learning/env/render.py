import numpy as np
import cv2

color_dict = {0: (0, 0, 255),
              1: (0, 255, 0),
              2: (255, 0, 0)}


def add_ADI(i, image, views, n, width, height, limit, span=5):
    """
    飞行姿态指示仪（Attitude Director Indicator, ADI）
    """
    # horizontal
    height_h = height // 4
    img_h = np.ones((height_h, width, 3), dtype=np.uint8) * 255
    center_x_h, center_y_h = width // 2, height_h // 2
    img_h = cv2.rectangle(img_h,
                          (0, 0),
                          (width - 1, height_h),
                          (0, 0, 0), 1)
    img_h = cv2.line(img_h,
                     (center_x_h, 0),
                     (center_x_h, height_h),
                     (0, 255, 255), 1)
    # vertical
    width_v = height // 4
    img_v = np.ones((height, width_v, 3), dtype=np.uint8) * 255
    center_x_v, center_y_v = width_v // 2, height // 2
    img_v = cv2.rectangle(img_v,
                          (0, 0),
                          (width_v, height - 1),
                          (0, 0, 0), 1)
    img_v = cv2.line(img_v,
                     (0, center_y_v),
                     (width_v, center_y_v),
                     (0, 255, 255), 1)
    # blank area
    img_blank = np.ones((height_h, width_v, 3), dtype=np.uint8) * 255
    img_blank = cv2.putText(img_blank,
                            str(i + 1),
                            (width_v // 2, height_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 0), 1,
                            cv2.LINE_AA)
    # ball
    tmp_h, tmp_v = height_h // n, width_v // n
    radius_h = (tmp_h - span) // 2
    radius_v = (tmp_v - span) // 2
    for j, [horizontal, vertical] in enumerate(views):
        color = color_dict[j]

        if abs(horizontal) >= limit:
            horizontal = abs(horizontal) / horizontal * limit
        delta = int(horizontal / limit * width / 2)
        img_h = cv2.circle(img_h,
                           (center_x_h + delta, int(tmp_h * (j + 1 / 2))),
                           radius_h,
                           color, 1)

        if abs(vertical) >= limit:
            vertical = abs(vertical) / vertical * limit
        delta = int(vertical / limit * height / 2)
        img_v = cv2.circle(img_v,
                           (int(tmp_v * (j + 1 / 2)), center_y_v + delta),
                           radius_v,
                           color, 1)

    return np.hstack([np.vstack([image, img_h]),
                      np.vstack([img_v, img_blank])])
