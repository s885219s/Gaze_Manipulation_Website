# # Load Functions
import cv2
import numpy as np

def get_eye_img(frame, shape, pos = "L", new_size = [48,64]):
    if(pos == "R"):
        lc = 36
        rc = 39
        FP_seq = [36,37,38,39,40,41]
    elif(pos == "L"):
        lc = 42
        rc = 45
        FP_seq = [45,44,43,42,47,46]
    else:
        print("Error: Wrong Eye")
    eye_cx = (shape.part(rc).x+shape.part(lc).x)*0.5
    eye_cy = (shape.part(rc).y+shape.part(lc).y)*0.5
    eye_center = [eye_cx, eye_cy]
    eye_len = np.absolute(shape.part(rc).x - shape.part(lc).x)
    bx_d5w = eye_len*3/4
    bx_h = 1.5*bx_d5w
    sft_up = bx_h*7/12
    sft_low = bx_h*5/12
    img_eye = frame[int(eye_cy-sft_up):int(eye_cy+sft_low),int(eye_cx-bx_d5w):int(eye_cx+bx_d5w)]
    ori_size = [img_eye.shape[0],img_eye.shape[1]]
    LT_coor = [int(eye_cy-sft_up), int(eye_cx-bx_d5w)] # (y,x)    
    img_eye = cv2.resize(img_eye, (new_size[1], new_size[0]))
    # create anchor maps
    ach_map = []
    for i,d in enumerate(FP_seq):
        resize_x = int((shape.part(d).x-LT_coor[1])*new_size[1]/ori_size[1])
        resize_y = int((shape.part(d).y-LT_coor[0])*new_size[0]/ori_size[0])
        # y
        ach_map_y = np.expand_dims(np.expand_dims(np.arange(0, new_size[0]) - resize_y, axis=1), axis=2)
        ach_map_y = np.tile(ach_map_y, [1,new_size[1],1])
        # x
        ach_map_x = np.expand_dims(np.expand_dims(np.arange(0, new_size[1]) - resize_x, axis=0), axis=2)
        ach_map_x = np.tile(ach_map_x, [new_size[0],1,1])
        if (i ==0):
            ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
        else:
            ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

    return img_eye/255, ach_map, eye_center, ori_size, LT_coor

def generate_agl_dig(demo_type=0, angle=15, nframe_per_demo=10):
    theda360 = np.linspace(0.0, 360.0, num=nframe_per_demo)
    theda180 = np.linspace(0.0, 180.0, num=nframe_per_demo)
    theda0 = np.linspace(0.0, 0.0, num=nframe_per_demo)
    if (demo_type == 0):
        '''sroll'''
        av = np.linspace(0.0, 0.0, num=nframe_per_demo)
        ah = angle*np.cos(theda360*np.pi/180)
    elif(demo_type == 1):
        '''shift'''
        av = angle*np.sin(theda360*np.pi/180)
        ah = np.linspace(0.0, 0.0, num=nframe_per_demo)
    elif(demo_type == 2):
        '''roll'''
        av = angle*np.sin(theda360*np.pi/180)
        ah = angle*np.cos(theda360*np.pi/180)
    else:
        print("Wrong demo Type, using roll")
        '''roll'''
        av = np.linspace(0.0, 0.0, num=nframe_per_demo)
        ah = np.linspace(-1*angle, angle, num=nframe_per_demo)
                         
    angle_dif = np.concatenate((np.expand_dims(av,1),np.expand_dims(ah,1)), axis = 1)
    return angle_dif

def generate_rand_seq_dig(start,end,nframe_per_demo=10):
    av = np.linspace(start[0], end[0], num=nframe_per_demo)
    ah = np.linspace(start[1], end[1], num=nframe_per_demo)
    angle_dif = np.concatenate((np.expand_dims(av,1),np.expand_dims(ah,1)), axis = 1)
    return angle_dif
