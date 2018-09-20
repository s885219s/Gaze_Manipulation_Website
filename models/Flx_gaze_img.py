# # Libraries
import cv2
import dlib
import time
import numpy as np
import random
import tensorflow as tf
from functions import get_eye_img, generate_agl_dig, generate_rand_seq_dig
from config import get_config
conf,_ = get_config()

if conf.mod == 'flx':
    import flx as model
elif conf.mod == 'deepwarp':
    import deepwarp as model
else:
    sys.exit("Wrong Model selection: flx or deepwarp")
    
model_dir = '../'+conf.weight_set+'/warping_model/ckpt/'+conf.mod+'/'+ str(conf.ef_dim) + '/'


# # Help

print("┌******** Demo of Flx-gaze :Image ********┐")
print("| Key \"r\" to Start redirecting            |")
print("| Key \"t\" to change demo Type             |")
print("| Key \"c\" to Capture the redirected image |")
print("| Key \"q\" to Quit demo                    |")
print("└*****************************************┘")


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
predictor = dlib.shape_predictor("./lm_feat/shape_predictor_68_face_landmarks.dat")

def show_webcam(video_res = [320,240], pixel_cut = [3,4], demo_type = 1, angle = 15, nframe_per_demo = 33):
    face_detect_size = [320,240]
    x_ratio = video_res[0]/face_detect_size[0]
    y_ratio = video_res[1]/face_detect_size[1]
    # user defined agl dif
    new_size = [48,64]
    start = [random.randint(50, 590), random.randint(30, 450)]
    end = [random.randint(50, 590), random.randint(30, 450)]
    c_c = generate_rand_seq_dig(start, end, nframe_per_demo)
    angle_dif = [generate_agl_dig(0, angle, nframe_per_demo),
                 generate_agl_dig(1, angle, nframe_per_demo), 
                 generate_agl_dig(2, angle, nframe_per_demo)]
    
    ori_frame = cv2.imread('./test_imgs/'+conf.test_img,1)
    if conf.record_time == True:
        # setting output fn
        out_fn = 'times.csv'
        fout= open(out_fn, 'w')
        fout.write(str('get_face,get_eye,infer,total\n'))
    # setting parameter
    gaze_manipulate = False
    get_new_frame = True
    test_img = False
    n_frame = -1
    # start capture video frames
    while True:
        frame = ori_frame.copy()
        if n_frame == 1000:
            n_frame = 0
        n_frame = n_frame + 1
           
        if gaze_manipulate ==True:
            time_infer = 0
            time_get_face = 0
            time_get_eye = 0
            runtime_start = time.time()
            time_start = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detect_gray = cv2.resize(gray, (face_detect_size[0], face_detect_size[1]))
            # Detect the facial landmarks
            detections = detector(face_detect_gray, 0)

            time_get_face = time.time() - time_start

            if len(detections) == 0:
                cv2.circle(frame,(video_res[0]-30, 30), 30, (0,0,255), -1)
                cv2.imshow("infer frame", frame)
            else: # detect faces
                # For each detected face
                LE_ach_maps=[]
                RE_ach_maps=[]
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

                time_infer = time.time() - time_start
                time_tot = time.time() - runtime_start
                # draw user interface
                cv2.circle(frame,(video_res[0]-30, 30), 30, (0,255,0), -1)
                cv2.putText(frame, 'F:'+str(int(time_get_face*1000)) + ' ms',(video_res[0]-45,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(frame, 'E:'+str(int(time_get_eye*1000)) + ' ms',(video_res[0]-45,25), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(frame, 'I:'+str(int(time_infer*1000)) + ' ms',(video_res[0]-45,35), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(frame, 'T:'+str(demo_type),(video_res[0]-45,45), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                if demo_type == 3:
                    cv2.circle(frame,(int(c_c[n_frame % nframe_per_demo,0]), int(c_c[n_frame % nframe_per_demo,1])), 5, (0,255,0), -1)
                cv2.imshow("infer frame", frame)
                if conf.record_time == True:
                    # write time to file
                    fout.write(str('%d,%d,%d,%d\n' % (int(time_get_face*1000),int(time_get_eye*1000),int(time_infer*1000),int(time_tot*1000))))
                    fout.flush()
        else: # gaze_manipulate == False
            cv2.circle(frame,(video_res[0]-30, 30), 30, (0,0,255), -1)
            cv2.imshow("infer frame", frame)
                
        if(n_frame == (nframe_per_demo-3)):
            n_frame = 0
            start = end
            end = [random.randint(200, 440), random.randint(120, 360)]
            c_c = generate_rand_seq_dig(start, end, nframe_per_demo)
            
        k = cv2.waitKey(10)
        if k==ord('q'): # "q"uit
            break
        elif k==ord('r'):  # "r"edirecting gaze
            if gaze_manipulate == False:
                gaze_manipulate = True
            else:
                gaze_manipulate = False
        elif k==ord('t'):  # change demo "t"ype
            print(angle_dif[0].shape)
            demo_type = (demo_type + 1) % 4
        elif k==ord('c'):  # "c"apture iference image
            cv2.imwrite('./saves/infer'+str(n_frame)+'.png',frame)
        else:
            pass
    cv2.destroyAllWindows()
    return True

def main():
    _ = show_webcam(video_res = [640,480], demo_type = 0, angle = conf.angle, nframe_per_demo = conf.nframe_per_demo)
    
if __name__ == '__main__':
    main()
    L_sess.close()
    R_sess.close()

