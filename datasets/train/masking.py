from ast import parse
import numpy as np
import sys, os
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--data_dir', dest='data_dir', help="data directory",
                        default=None, type=str)
    parser.add_argument("--imgset", dest='imgset', help='imageset',
                        default='all.txt', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default='./output', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


def mask(img_path, img_name, box_path):
    img = Image.open(img_path)

    normal_box = []
    mask_box = []
    with open(box_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            _l = l.replace('\n', '')
            box_data = _l.split(" ")
            if box_data[0] == '6':
                mask_box += [[float(b) for b in box_data[1:]]]
            else:
                normal_box += [l]

    img_wh = img.size

    if len(mask_box) == 0:
        img.save(args.output_dir + "/" + img_name + ".png")
        with open(args.output_dir + "/" + img_name + ".txt", 'w') as f:
            for n in normal_box:
                f.write(n)
        return

    for m in mask_box:
        # xywh to xyxy
        cx = m[0]
        cy = m[1]
        w = int(m[2] * img.size[0])
        h = int(m[3] * img.size[1])
        m[0] = int((cx - m[2] / 2) * img.size[0])
        m[1] = int((cy - m[3] / 2) * img.size[1])
        m[2] = int((cx + m[2] / 2) * img.size[0])
        m[3] = int((cy + m[3] / 2) * img.size[1])

        mask_img = Image.new(mode="RGB", size=(w, h), color=(128, 128, 128))
        img.paste(mask_img, (m[0], m[1]))
    img.save(args.output_dir + "/" + img_name + ".png")
    with open(args.output_dir + "/" + img_name + ".txt", 'w') as f:
        for n in normal_box:
            f.write(n)
    return


def main():
    imgset = args.data_dir + "/ImageSets/" + args.imgset
    img_folder = args.data_dir + "/JPEGImages/"
    anno_folder = args.data_dir + "/Annotations/"

    img_name = []
    img_list = []
    anno_list = []
    with open(imgset, 'r') as f:
        lines = f.readlines()
        for l in lines:
            img_name += [l.replace("\n", "")]
            img_list += [img_folder + l.replace("\n", ".png")]
            anno_list += [anno_folder + l.replace("\n", ".txt")]

    for i, (img, name, anno) in enumerate(zip(img_list, img_name, anno_list)):
        print(i, img, anno)
        mask(img, name, anno)


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    main()