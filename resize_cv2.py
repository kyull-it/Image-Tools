import cv2

def resize(image):

    height, width = image.shape[:2]
    
    if height > 1000 or width > 1000:
        rw = width / max(height, width)
        rh = height / max(height, width)
        re_width = int(1000 * rw)
        re_height = int(1000 * rh)
        resized_image = cv2.resize(image, (re_width, re_height), cv2.INTER_CUBIC)
        return resized_image
        
    elif height < 100 or width < 100:
        rw = width / min(height, width)
        rh = height / min(height, width)
        re_width = int(100 * rw)
        re_height = int(100 * rh)
        resized_image = cv2.resize(image, (re_width, re_height), cv2.INTER_CUBIC)
        return resized_image
        
    else:
        return image


