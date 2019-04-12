# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:31:51 2019

@author: Alexander
"""

#!/usr/bin/env python
#
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Generate training and test images.
"""

__all__ = (
    'generate_ims',
)


#Python 3 librries
import itertools, math, os, random, sys, string, shutil, time, argparse

#Import 3rd party libraries
#Python Image Library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2          #OpenCv
import numpy        #Numpy

#Import my custom libraries
import common

BACKGROUNDS_DIR = ".\\bgs"
FONT_HEIGHT = 16  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (32, 128)

CHARS = open(common.fnCharList).read()

streetNames = list()

def make_char_ims(font_path, output_height):
    "Generate images of characters according to font files provided"
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    "Generate a rotation matrix given yaw, pitch, and roll"
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    "Choose sign and text colors such that contrast is greater than 3"
    first = True
    while first or signColor - text_color < 0.3:
        text_color = random.random()
        signColor = random.random()
        if text_color > signColor:
            text_color, signColor = signColor, text_color
        first = False
    #text_color = 0 #black
    #signColor = 1 #white
    return text_color, signColor


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    """
    Generates the affine transformation matrix.
    Affine transforms translate, scale, and rotate such that
    points, straight lines, and planes are preserved. Sets of 
    parallel lines remain parallel after an affine transformation.
    """
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    #set the scale
    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    #set rotation
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generateSignText():
    streetName = ''
    isNumberedStreet = (True, False)[random.randint(0,1)]
    if(isNumberedStreet):
        streetNum = random.randint(1, 99)
        streetName = str(streetNum) + getSuffix(streetNum%10) + ' '
    # numWords = 2 if isNumberedStreet else 1
    # for i in range(0, numWords):
        #word = ''.join(random.choices(string.ascii_uppercase, k = random.randint(1, 8)))
    f=open(common.fnStreetList)
    for line in f:
        if(not is_ascii(line)):
            continue
        #remove trailing \n
        streetNames.append(line.upper().rstrip())
    word = random.choice(streetNames)
    streetName = streetName + word
    return "{} {}".format(streetName, random.choice(common.STREETS))

def getSuffix(lastDigit):
    switcher = {
        1: 'ST',
        2: 'ND',
        3: 'RD'
    }
    return switcher.get(lastDigit, "TH")

def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generateSign(font_height, char_ims, text = None):
    # h_padding = random.uniform(0.2, 0.4) * font_height
    # v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * 0.05
    h_padding = font_height*0.1
    v_padding = font_height*0.2
    radius = 1 + int(font_height * 0.1)

    if(not text):
        text = generateSignText()
    text_width = sum(char_ims[c].shape[1] for c in text)
    text_width += (len(text) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                  int(text_width + h_padding * 2))

    text_color, signColor = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 

    #apply mask to each character
    for c in text:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    sign = (numpy.ones(out_shape) * signColor * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return sign, rounded_rect(out_shape, radius), text


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "./bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        if not os.path.getsize(fname):
            print("Did not find background {}".format(fname))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)
    #print (bg)

    sign, signMask, code = generateSign(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=sign.shape,
                            to_shape=bg.shape,
                            min_scale=0.6,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.2)
    sign = cv2.warpAffine(sign, M, (bg.shape[1], bg.shape[0]))
    signMask = cv2.warpAffine(signMask, M, (bg.shape[1], bg.shape[0]))

    out = sign * signMask + bg * (1.0 - signMask)
#    cv2.imshow('image', out)
#    cv2.waitKey(0)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    #sign = cv2.resize(sign, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
    #out_of_bounds = False
    return out, code, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.TTF')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    """
    Generate number sign images.
    :return:
        Iterable of number sign images.
    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(common.fontDirectory)
    num_bg_images = len(os.listdir(common.backgroundDirectory))
    #print (num_bg_images)
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)

def setupBatchFileIO():
    if(os.path.exists(common.imageDirectory)):
        print("Deleting old dataset")
        shutil.rmtree(common.imageDirectory)
    os.mkdir(".\data\images")
    if(os.path.isfile(common.fnWords)):
        os.remove(common.fnWords)
    gtFile = open(".\data\words.txt", "w+")
    print("Getting street Names")
    f=open(common.fnStreetList)
    for line in f:
        if(not is_ascii(line)):
            continue
        #remove trailing \n
        streetNames.append(line.upper().rstrip())
    return gtFile

def generate_custom(signText):
    fonts, font_char_ims = load_fonts(common.fontDirectory)
    sign, signMask, code = generateSign(FONT_HEIGHT, font_char_ims[random.choice(fonts)], signText)
    fname = "data/customImages/{}.png".format(args.text)
    cv2.imwrite(fname, sign * 255.)

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


if __name__ == "__main__":
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--count", type=int, help="size of random batch")
    parser.add_argument('-t', "--text", help="text on custom image")
    parser.add_argument("--single", help="generate a single image with custom text", action="store_true")
    parser.add_argument("--batch", help="generate batch of random images", action='store_true')
    args = parser.parse_args()
    
    start_time = time.time()
    if args.batch:
        gtFile = setupBatchFileIO()
        print("Generating Images")
        numImages = int(args.count)
        im_gen = itertools.islice(generate_ims(), numImages)
        common.printProgressBar(0, numImages, prefix = 'Progress:', suffix = 'Complete', length = 50)
        #p is the out of bounds flag
        for img_idx, (im, text, inBounds) in enumerate(im_gen):
            fname = "data/images/{:08d}.png".format(img_idx)
            #update ground truth file
            gtFile.write("{} {}\n".format(img_idx, text)) 
            cv2.imwrite(fname, im * 255.)
            common.printProgressBar(img_idx, numImages, prefix = 'Progress:', suffix = 'Complete', length = 50)
        gtFile.close()
    
    else:
        if(not os.path.exists(".\data\customImages")):
            os.mkdir(".\data\customImages")
        generate_custom(args.text.upper())
        
    elapsedTime = time.time() - start_time
    print('{:02f}:{:02f}:{:02f}'.format(elapsedTime // 3600, (elapsedTime % 3600 // 60), elapsedTime % 60))