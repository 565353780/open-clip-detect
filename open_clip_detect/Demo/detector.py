from open_clip_detect.Module.detector import Detector

def demo():
    model_file_path = '/home/chli/Model/open_clip/DFN5B-CLIP-ViT-H-14-378.bin'
    device = 'cpu'
    auto_download_model = False
    image_file_path = '/home/chli/Dataset/CapturedImages/y_5_x_3.png'

    detector = Detector(model_file_path, device, auto_download_model)

    clip_feature = detector.detectImageFile(image_file_path)

    print('clip_feature:')
    print(clip_feature)
    print(clip_feature.shape)
    return True
