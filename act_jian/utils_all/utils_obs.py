from act_jian.utils_all.constants import JOINT_POSITIONS_LIMITS


def normalize_image(image):
    image = image / 255.0
    return image


def denormalize_image(image):
    image = image * 255.0
    return image


def normalize_position(position):
    position[:7] = (position[:7] - JOINT_POSITIONS_LIMITS[:, 0]) / (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0])
    return position


def denormalize_position(position):
    position[:7] = position[:7] * (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0]) + JOINT_POSITIONS_LIMITS[:, 0]
    return position
