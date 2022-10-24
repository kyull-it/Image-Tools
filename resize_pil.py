
import PIL.Image as Image
import time

def pil_resize(image_path, size_tuple, save_or_not):
    
    """
    help(pil_resize)
    
     - image_path : string
     - size_tuple : 2-tuple (int, int)
     - save_or_not : boolean (True or False)
     
     - resampling filter (upscaling)                         (downscaling)
        # 0 - NEAREST    (default filter - 가장 깨짐이 심함)    (default filter - 가장 깨짐이 심함)
        # 1 - LANCZOS    (3보다 약간 깨지고 선명함)               (3보다 약간 깨지고 선명함)
        # 2 - BILINEAR   (깨짐이 없지만 흐릿함)                  (깨짐이 없지만 흐릿함)
        # 3 - BICUBIC    (깨짐이 약간 있지만 선명함)              (깨짐이 약간 있지만 선명함)
        # 4 - BOX        (0과 비슷)                           (가장 선명하고 부드러움)
        # 5 - HAMMING    (2에서 좀더 깨진 버전)                  (2보다 흐릿함)
     
    """
    
    start = time.time()
    
    
    # 이미지 로드
    image_load = Image.open(image_path)
    print("Successfully loaded...", end = "\n")

    
    # resize할 pixel tuple 계산       
    template = size_tuple
    size = image_load.size
        
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
        
        resized_width = int(size[0] * w_ratio)
        resized_height = int(size[1] * w_ratio)
        
    else:
        
        resized_width = int(size[0] * h_ratio)
        resized_height = int(size[1] * h_ratio)
    
    
    # resize할 filter 선택
    
    # 고효율 upscale filter  - 용량도 가장 낮음  
    #  - 테스트 이미지 : (영산대 로고 : UPLOAD_1663126115535.webp)
    if resized_width or resized_height >= 1:
        resampling = Image.BICUBIC    
        
    # 고효율 downscale filter  - 용량은 비슷 
    #  - 테스트 이미지 : (영화 브로커 이미지 : UPLOAD_1665194032713.webp)
    else:
        resampling = Image.BOX         
    
    # resizing
    resized_image = image_load.resize((resized_width, resized_height), resampling)
    
    print("Resized image size is : ", resized_image.size)
    print("")
    
    print("Cropping image....")
    print("""
    .
    .
    .
    """)
      
    # Crop할 이미지 pixel위치 계산 (이미지의 중앙)
    left = (resized_image.size[0] - template[0]) // 2
    top = (resized_image.size[1] - template[1]) // 2
    right = left + template[0]
    bottom = top + template[1]
    
    # Crop
    cropped_image = resized_image.crop((left, top, right, bottom))
    print("Cropped size : ", cropped_image.size)
    
    cropped_image.show()
    
    
    # Save    
    if save_or_not == True:
        
        image_name = '_'.join(path.split("_")[1:])
    
    save_path = "./room/ROOM/auknotwork/resize"
    cropped_image.save(f"{save_path}/resized_pil_{image_name}_{size}_{template}_{resampling}.webp")
    
    print("Duration : ", time.time() - start)
    
    return cropped_image
    
    
    
# 이미지 경로
path = "./room/ROOM/auknotwork/origin/origin_UPLOAD_1663126115535.webp"

# 실행
pil_resize(path, (1080, 680), True)


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
# Duration : 0.22581100463867188
