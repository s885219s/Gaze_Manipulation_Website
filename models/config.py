#-*- coding: utf-8 -*-
import argparse

model_config = argparse.ArgumentParser()

# model
model_config.add_argument('--height', type=eval, default=48, help='')
model_config.add_argument('--width', type=eval, default=64, help='')
model_config.add_argument('--channel', type=eval, default=3, help='')
model_config.add_argument('--ef_dim', type=eval, default=12, help='')
model_config.add_argument('--agl_dim', type=eval, default=2, help='')
model_config.add_argument('--encoded_agl_dim', type=eval, default=16, help='')

#training
model_config.add_argument('--lr', type=eval, default=0.001, help='')
model_config.add_argument('--epochs', type=eval, default=300, help='')
model_config.add_argument('--batch_size', type=eval, default=256, help='')
model_config.add_argument('--dataset', type=str, default='dirl_48x64', help='')
model_config.add_argument('--early_stop', type=eval, default=16, help='')
model_config.add_argument('--validation_split', type=eval, default=0.05, help='')

#load trained weight
model_config.add_argument('--load_weights', type=bool, default=False, help='')
model_config.add_argument('--easy_mode', type=bool, default=True, help='')

#eye
model_config.add_argument('--eye', type=str, default="L", help='')
model_config.add_argument('--pose', type=str, default="0P", help='')

#demo
model_config.add_argument('--mod', type=str, default="flx", help='')
# model_config.add_argument('--mod', type=str, default="deepwarp", help='')
model_config.add_argument('--weight_set', type=str, default="weights_20180413", help='')
model_config.add_argument('--test_img', type=str, default="0013_0P_0H_-30V.png", help='')
model_config.add_argument('--record_time', type=bool, default=False, help='')
model_config.add_argument('--angle', type=int, default=20, help='')
model_config.add_argument('--nframe_per_demo', type=int, default=30, help='')

def get_config():
    config, unparsed = model_config.parse_known_args()
    print(config)
    return config, unparsed
