import os
import cv2
import skvideo.io
import numpy as np
import time
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, flash, redirect
from forms import uploadPhotoForm

import dlib
import time
import random
import tensorflow as tf
from models.config import get_config
from models.functions import get_eye_img, generate_agl_dig, generate_rand_seq_dig
conf,_ = get_config()

if conf.mod == 'flx':
    import models.flx as model
elif conf.mod == 'deepwarp':
    import models.deepwarp as model
else:
    sys.exit("Wrong Model selection: flx or deepwarp")
    
model_dir = conf.weight_set+'/warping_model/ckpt/'+conf.mod+'/'+ str(conf.ef_dim) + '/'

app = Flask(__name__)
app.config['SECRET_KEY'] = '79c55a394f9b72dcc886f8259c2d62c8'

# # Load Gaze Redirection Model to GPU
with tf.Graph().as_default() as g:
    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        LE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel], name="input_img") # [None, 41, 51, 3]
        LE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width,conf.ef_dim], name="input_fp") # [None, 41, 51, 14]
        LE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang") ## [None, 41, 51, 2]
        LE_phase_train = tf.placeholder(tf.bool, name='phase_train') # a bool for batch_normalization
    # inference model.
    LE_img_pred, _, _ = model.inference(LE_input_img, LE_input_fp, LE_input_ang,  LE_phase_train, conf)
    # split modle here
    L_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False), graph = g)
    # load model
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir+'L/')
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(L_sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

with tf.Graph().as_default() as g2:
    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        RE_input_img = tf.placeholder(tf.float32, [None, conf.height, conf.width, conf.channel], name="input_img") # [None, 41, 51, 3]
        RE_input_fp = tf.placeholder(tf.float32, [None, conf.height, conf.width,conf.ef_dim], name="input_fp") # [None, 41, 51, 14]
        RE_input_ang = tf.placeholder(tf.float32, [None, conf.agl_dim], name="input_ang") ## [None, 2]
        RE_phase_train = tf.placeholder(tf.bool, name='phase_train') # a bool for batch_normalization
    # inference model.
    RE_img_pred, _, _ = model.inference(RE_input_img, RE_input_fp, RE_input_ang, RE_phase_train, conf)
    # split modle here
    R_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False), graph = g2)
    # load model
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir+'R/')
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(R_sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

# # Load dlib

#Face detector
detector = dlib.get_frontal_face_detector()
#Landmark identifier. Set the filename to whatever you named the downloaded file
predictor = dlib.shape_predictor("./models/lm_feat/shape_predictor_68_face_landmarks.dat")

def save_photo(form_photo):
    photo_format = secure_filename(form_photo.filename)[-4:]
    current_ms = lambda: str(round(time.time() * 1000))
    photo_fn = current_ms()+photo_format
    print(photo_fn)
    photo_path = os.path.join(app.root_path, 'static/uploads', photo_fn)

    file_str = form_photo.read()
    npimg = np.fromstring(file_str, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite(photo_path,img)
    return photo_fn

def convert_images_to_video(photo_file, video_res = [320,240], pixel_cut = [3,4], demo_type = 1, angle = 15, nframe_per_demo = 33):
    face_detect_size = [320,240]
    x_ratio = video_res[0]/face_detect_size[0]
    y_ratio = video_res[1]/face_detect_size[1]
    # user defined agl dif
    new_size = [48,64]
    angle_dif = [generate_agl_dig(0, angle, nframe_per_demo),
                 generate_agl_dig(1, angle, nframe_per_demo), 
                 generate_agl_dig(2, angle, nframe_per_demo)]

    # output_video_path = './static/uploads/' + photo_path[:-4] + '.mp4'
    output_video_path = './static/uploads/' + photo_file[:-4] + '.mp4'
    frame_list = []
    ori_frame = cv2.imread('./static/uploads/'+ photo_file,1)
    print(ori_frame.shape)
    # ori_frame = cv2.imread('static/uploads/' + photo_path)
    ori_frame = cv2.resize(ori_frame,(640,480))
    
    for n_frame in range(nframe_per_demo):
        frame = ori_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray, (face_detect_size[0], face_detect_size[1]))
        # Detect the facial landmarks
        detections = detector(face_detect_gray, 0)
        LE_ach_maps=[]
        RE_ach_maps=[]
        if len(detections) == 0:
            return None
        else:    
            for k,bx in enumerate(detections):
                # Get facial landmarks
                time_start = time.time()
                target_bx = dlib.rectangle(left=int(bx.left()*x_ratio), right =int(bx.right()*x_ratio),
                                            top =int(bx.top()*y_ratio),  bottom=int(bx.bottom()*y_ratio))
                shape = predictor(gray, target_bx)
                # get eye
                LE_img, LE_ach_maps, LE_center, LE_ori_size, LE_LT_coor = get_eye_img(frame, shape, pos="L", new_size=new_size)
                RE_img, RE_ach_maps, RE_center, RE_ori_size, RE_LT_coor = get_eye_img(frame, shape, pos="R", new_size=new_size)
                time_get_eye = time.time() - time_start
                # gaze manipulation
                time_start = time.time()

                if demo_type == 3:
                    px_to_degree = 15/100
                    # dif [v, h]
                    input_L_agl_dif = [int((LE_center[1] - c_c[n_frame % nframe_per_demo,1])*px_to_degree), int((c_c[n_frame % nframe_per_demo,0] - LE_center[0])*px_to_degree)]
                    input_R_agl_dif = [int((RE_center[1] - c_c[n_frame % nframe_per_demo,1])*px_to_degree), int((c_c[n_frame % nframe_per_demo,0] - RE_center[0])*px_to_degree)]
                else:
                    input_L_agl_dif = angle_dif[demo_type][n_frame % nframe_per_demo,:]
                    input_R_agl_dif = angle_dif[demo_type][n_frame % nframe_per_demo,:]

                # gaze redirection
                # left Eye
                LE_infer_img = L_sess.run(LE_img_pred, feed_dict= {
                                                                LE_input_img: np.expand_dims(LE_img, axis = 0),
                                                                LE_input_fp: np.expand_dims(LE_ach_maps, axis = 0),
                                                                LE_input_ang: np.expand_dims(input_L_agl_dif, axis = 0),
                                                                LE_phase_train: False
                                                                })
                LE_infer = cv2.resize(LE_infer_img.reshape(new_size[0],new_size[1],3), (LE_ori_size[1], LE_ori_size[0]))
                # right Eye
                RE_infer_img = R_sess.run(RE_img_pred, feed_dict= {
                                                                RE_input_img: np.expand_dims(RE_img, axis = 0),
                                                                RE_input_fp: np.expand_dims(RE_ach_maps, axis = 0),
                                                                RE_input_ang: np.expand_dims(input_R_agl_dif, axis = 0),
                                                                RE_phase_train: False
                                                                })
                RE_infer = cv2.resize(RE_infer_img.reshape(new_size[0],new_size[1],3), (RE_ori_size[1], RE_ori_size[0]))
                # repace eyes
                frame[(LE_LT_coor[0]+pixel_cut[0]):(LE_LT_coor[0]+LE_ori_size[0]-pixel_cut[0]),
                        (LE_LT_coor[1]+pixel_cut[1]):(LE_LT_coor[1]+LE_ori_size[1]-pixel_cut[1])] = LE_infer[pixel_cut[0]:(-1*pixel_cut[0]), pixel_cut[1]:-1*(pixel_cut[1])]*255
                frame[(RE_LT_coor[0]+pixel_cut[0]):(RE_LT_coor[0]+RE_ori_size[0]-pixel_cut[0]),
                        (RE_LT_coor[1]+pixel_cut[1]):(RE_LT_coor[1]+RE_ori_size[1]-pixel_cut[1])] = RE_infer[pixel_cut[0]:(-1*pixel_cut[0]), pixel_cut[1]:-1*(pixel_cut[1])]*255
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
    print(frame_list[0].shape)
    skvideo.io.vwrite(output_video_path,frame_list,inputdict={},outputdict={'-r':str(nframe_per_demo),'-pix_fmt': 'yuvj420p'},verbosity=1)
    return output_video_path

@app.route("/", methods=['GET','POST'])
def home():
    uploads = os.path.join(app.root_path, 'static/uploads/')
    if not os.path.isdir(uploads):
        os.makedirs(uploads)

    image_file = url_for('static',filename='uploads/test.png')
    form = uploadPhotoForm()
    if form.validate_on_submit():
        if form.photo.data:
            if request.form['submit'] == 'scroll':
                demo_type = 0
            elif request.form['submit'] == 'shift':
                demo_type = 1
            elif request.form['submit'] == 'roll':
                demo_type = 2
            else:
                print('error')
            # video_file = './static/uploads/0013_0P_0H_-30V_scroll.mp4'
            photo_file = save_photo(form.photo.data)
            video_file = convert_images_to_video(photo_file, video_res = [640,480], demo_type = demo_type, angle = conf.angle, nframe_per_demo = conf.nframe_per_demo)
        if video_file:            
            return render_template('home.html', video_file=video_file, form=form)
        else:
            flash(u'Sorry we can not process your image.', 'danger')
            render_template('home.html', form=form)
    # elif request.methods == 'GET':
    else:
        print('fail')

    return render_template('home.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
    L_sess.close()
    R_sess.close()