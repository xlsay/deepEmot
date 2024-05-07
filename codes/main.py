# coding=utf-8
'''
@File    :   test.py
@Time    :   2024/04/09
@Author  :   xlwang 
'''
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
from ffmpy import FFmpeg
import ffmpeg
import numpy as np
import shutil
import cv2
import os

def get_intersect(rect1, rect2):
    ''' xyxy '''
    left = max(rect1[0], rect2[0])
    right = min(rect1[2], rect2[2])
    top = max(rect1[1], rect2[1])
    bottom = min(rect1[3], rect2[3])
    intersect = max(0, right-left) * max(0, bottom-top)
    area1, area2 = (rect1[2]-rect1[0])*(rect1[3]-rect1[1]), (rect2[2]-rect2[0])*(rect2[3]-rect2[1])
    # union = area1 + area2 - intersect
    union = min(area1,area2)
    sect = np.clip(intersect/union, 0, 1)
    return sect


def ana_img(img, save_dir, is_path=False, fi=0):
    # 'emotion': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emo_zh = {'angry': '生气', 'disgust': '厌恶', 'fear': '害怕', 'happy': '高兴', 'sad': '难过', 'surprise': '惊讶', 'neutral': '自然'}
    ress = DeepFace.analyze(img_path=img, enforce_detection=False,detector_backend='mtcnn',
                            silent=True, actions = ['emotion']) # ["emotion", "age", "gender", "race"]
    ress = [res for res in ress if res['face_confidence']>0.7]
    ress = [(res['region']['w']*res['region']['h'], res) for res in ress]
    ress.sort(key=lambda x: x[0], reverse=True)

    # plot res
    if is_path:
        img = cv2.imread(img)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    color_box, color_txt = (0,255,0), (225,255,255)
    font_file = 'font/Microsoft-YaHei-Semibold.ttc'
    fontsize = max(round(max(img.size)/40), 8)
    font = ImageFont.truetype(font_file, fontsize)
    line_thickness = 2
    plot_boxes = []
    for res_ in ress:
        res = res_[1]
        x,y = res['region']['x'],res['region']['y'] # top-left corner
        w,h = res['region']['w'],res['region']['h']
        box = [int(x),int(y),int(x+w),int(y+h)]
        not_plot = False
        for bx in plot_boxes:
            if get_intersect(box, bx)>0.01:
                not_plot = True
                break
        if not_plot:
            continue
        plot_boxes.append(box)
        emo = res['dominant_emotion']
        txt = '{} {}%'.format(emo_zh[emo], int(res['emotion'][emo]))
        draw.rectangle(box, width=line_thickness, outline=color_box) 
        tw, th = font.getsize(txt)
        draw.rectangle([box[0], box[1]-th+5, box[0]+tw, box[1]], fill=(0,139,0))
        draw.text((box[0], box[1]-th+1), txt, fill=color_txt, font=font)

    cv2.imwrite('{}/{:0>6d}.jpg'.format(save_dir,fi),np.asarray(img))


def extract_mp3(vd_dir, vd_name, au_path):
    ext = au_path.split('.')[-1] # 'mp3'
    ff = FFmpeg(inputs={os.path.join(vd_dir,vd_name): None},
                outputs={au_path: '-f {} -vn'.format(ext)})
    ff.run()


def merge_av(af, vf):
    vd_dir, vd_name = os.path.dirname(vd_path), os.path.basename(vd_path)
    vd_names = vd_name.split('.')
    new_vd = os.path.join(vd_dir,vd_names[0]+'_emot.'+vd_names[-1])
    audio = ffmpeg.input(af)
    video = ffmpeg.input(vf)
    out = ffmpeg.output(video, audio, new_vd)
    out.run()
    print('ok')


def gen_video(img_dir, vd_file, fps=20.0):
    imgs = sorted([img for img in os.listdir(img_dir) if img[-3:]=='jpg'])
    imgs.sort()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h,w = cv2.imread(os.path.join(img_dir,imgs[0])).shape[:2]
    out = cv2.VideoWriter(vd_file, fourcc=fourcc, fps=fps, frameSize=(w,h))
    for img in imgs:
        _img = cv2.imread(os.path.join(img_dir,img))
        out.write(_img)
    out.release()


if __name__=='__main__':
    vd_path = '../data/test.mp4'
    vd_dir, vd_name = os.path.dirname(vd_path), os.path.basename(vd_path)
    orig_audio = os.path.join(vd_dir, '{}.{}'.format(vd_name.split('.')[0], 'mp3'))
    if os.path.exists(orig_audio):
        os.remove(orig_audio)
    print('extracting orig audio...')
    extract_mp3(vd_dir, vd_name, orig_audio)
    img_dir = os.path.join(vd_dir, vd_name.split('.')[0])
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    vs = cv2.VideoCapture(vd_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    num_img = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    h1,h2 = 70,680
    w1,w2 = 20,1240
    fi = 0
    # ana emotions
    while vs.isOpened():
        grabbed,frame = vs.read()
        if not grabbed:
            break
        fi += 1
        print('processing {}/{}...'.format(fi, num_img))
        ana_img(frame[h1:h2, w1:w2, :], img_dir, fi=fi)
    mark_vd = img_dir+'_mark.mp4'
    if os.path.exists(mark_vd):
        os.remove(mark_vd)
    print('making video...')
    gen_video(img_dir,mark_vd,fps=fps)
    print('merging video and orig audioaudio...')
    merge_av(orig_audio, mark_vd)
    # del tmp files
    os.remove(orig_audio)
    os.remove(mark_vd)
    shutil.rmtree(img_dir)
    # print()

