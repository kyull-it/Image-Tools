

def vips_resize(image_path, size_tuple, save_or_not):
    
    import pyvips
    import time
    
    """
    help(vips_resize)
    
     - image_path : string
     - size_tuple : 2-tuple (int, int)
     - save_or_not : boolean (True or False)
     
     - resampling filter (upscaling)                         (downscaling)
        # 0 - nearest    (pil보다 훨씬 좋지 않음)               ()
        # 1 - linear     (0보단 좋지만 2보단 좋지 않음)          (가장 용량이 적음)
        # 2 - cubic      (2,3,4,5가 다 비슷)                  (가장 품질이 좋음)
        # 3 - mitchell   ()                                 ()
        # 4 - lanczos2   ()                                 ()
        # 5 - lanczos3   ()                                 ()
     
    """
    
    
    start = time.time()
    
    
    # 이미지 로드
    image_load = pyvips.Image.new_from_file(image_path)
    print("Successfully loaded...", end = "\n")
    
    
    # resize할 pixel tuple 계산       
    template = size_tuple
    size = (image_load.width, image_load.height)

    print("Your optimal size is : ", template)
    print("Your Input size is : ", size)

    print("Resizing image....")
    print("""
    .
    .
    .
    """)
    
    w_ratio = template[0]/size[0]
    h_ratio = template[1]/size[1]
    
    if w_ratio > h_ratio:
        
        scale = w_ratio
        resized_width = int(size[0] * w_ratio)
        resized_height = int(size[1] * w_ratio)
        
    else:
        
        scale = h_ratio
        resized_width = int(size[0] * h_ratio)
        resized_height = int(size[1] * h_ratio)
    
    
    # resize할 filter 선택
    
    # 고효율 upscale filter
    #  - 테스트 이미지 : (영산대 로고 : UPLOAD_1663126115535.webp)
    if scale >= 1:
        resampling = "cubic"
        lossless = False
        
    # 고효율 downscale filter - 화질보단 용량이 가장 작은 것
    #  - 테스트 이미지 : (영화 브로커 이미지 : UPLOAD_1665194032713.webp)
    else:
        resampling = "linear"
        lossless = True
    
    # resizing
    resized_image = image_load.resize(scale, kernel=resampling)
    
    print("Resized image size is : ", (resized_image.width, resized_image.height))
    print("")
    
    print("Cropping image....")
    print("""
    .
    .
    .
    """)
    
       
    # Crop할 이미지 pixel위치 계산 (이미지의 중앙)
    left = (resized_image.width - template[0]) // 2
    top = (resized_image.height - template[1]) // 2
    right = template[0]
    bottom = template[1]
    
    # Crop
    cropped_image = resized_image.crop(left, top, right, bottom)
    print("Cropped size : ", (cropped_image.width, cropped_image.height))
    
    
    
    # Save    
    if save_or_not == True:
        
        image_name = '_'.join(path.split("_")[1:])
    
    save_path = "./room/ROOM/auknotwork/vips_resize"
    cropped_image.webpsave(f"{save_path}/resized_vips_{image_name}_{size}_{template}_{resampling}.webp", lossless=lossless)
    
    print("Duration : ", time.time() - start)
    
    return cropped_image


if __name__ == '__main__':
    vips_resize()
    

# # 이미지 경로
# path = "./room/ROOM/auknotwork/origin/origin_UPLOAD_1665307.jpeg"

# # 실행
# vips_resize(path, (150, 150), True)


# 결과
# Successfully loaded...
# Your optimal size is :  (1080, 680)
# Your Input size is :  (388, 194)
# Resizing image....

#     .
#     .
#     .
    
# Resized image size is :  (1360, 680)

# Cropping image....

#     .
#     .
#     .
    
# Cropped size :  (1080, 680)
# Duration : 0.03664708137512207
