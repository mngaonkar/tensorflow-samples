import requests
requests.packages.urllib3.disable_warnings()
import ssl
import time

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

import tensorflow as tf
import tensorflow_hub as hub

IMG_PATH = "/Users/mahadev/Downloads/peoples-1.jpeg"

def main():
    if not check_gpu():
        print("no GPU found, exiting...")
        return

    force_cpu()
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    t1 = time.time()
    result = run_detector(detector, IMG_PATH)
    t2 = time.time()
    print("total time taken {}".format(t2-t1))

    print(result)

def force_cpu():
    tf.config.experimental.set_visible_devices([], 'GPU')

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    return img

def run_detector(detector, image_path):
    img = load_image(image_path)
    converted_image = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_image)
    result = {key:value.numpy() for key,value in result.items()}

    return result

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return True
    else:
        return False
        
if __name__ == '__main__':
    main()
