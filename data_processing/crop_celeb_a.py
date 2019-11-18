import cv2
from mtcnn import MTCNN
import argparse
from glob import glob
import os
from collections import defaultdict


def crop(src_file, tar_file, img_size, detector):
    img = cv2.cvtColor(cv2.imread(src_file), cv2.COLOR_BGR2RGB)

    try:
        info_json = detector.detect_faces(img)[0]
        x, y, w, h = info_json["box"]
        cropped = img[y:y + h, x:x + w, ...]
        cropped = cv2.resize(src=cropped, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=tar_file, img=cropped)
        print(f"saved to {tar_file}")
        return

    except IndexError:
        print(f"Could not locate face on {src_file}")
        return

    except cv2.error:
        print(f"cv2.error for cropping happened with bb data of x:{x},y:{y},w{w},h{h}")
        return


def main(main_dir, src_dir, tar_dir, txtfile, cont, img_size):
    source_dir = os.path.join(main_dir, src_dir)
    target_dir = os.path.join(main_dir, tar_dir)
    file = os.path.join(main_dir, txtfile)
    detector = MTCNN()
    with open(file) as f:
        line = f.readline()
        while line:
            filename, id = line.split()
            if int(filename[:-4]) < cont:
                print(int(filename[:-4]))
                line = f.readline()
                continue
            id_dir = os.path.join(target_dir, id)
            if not os.path.isdir(id_dir):
                os.makedirs(id_dir)
            src_file = os.path.join(source_dir, f"{filename}")
            tar_file = os.path.join(id_dir, f"{filename}")
            crop(src_file=src_file, tar_file=tar_file, img_size=img_size, detector=detector)
            line = f.readline()
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", type=str, default="/data/celeba_data/")
    ap.add_argument("--src_dir", type=str, default="img_align_celeba")
    ap.add_argument("--tar_dir", type=str, default="celeba_cropped")
    ap.add_argument("--txtfile", type=str, default="identity_CelebA.txt")
    ap.add_argument("--continue", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=112)
    args = vars(ap.parse_args())
    main(main_dir=args["main_dir"], src_dir=args["src_dir"], tar_dir=args["tar_dir"], txtfile=args["txtfile"],
         cont=args["continue"], img_size=args["img_size"])
