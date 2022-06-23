import os
import re
import random

import streamlit as st
import pandas as pd
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgba2rgb, hsv2rgb, rgb2gray, gray2rgb
from skimage.draw import circle_perimeter, disk

st.title('Value Recognition Practice')

IMG_DIR = './data/portrait'
seed = 0

@st.cache
def load_random_image(seed):
    fps = os.listdir(IMG_DIR)
    fp = os.path.join(IMG_DIR, random.choice(fps))

    img = skimage.io.imread(fp)

    if img.shape[2] == 4:
        img = rgba2rgb(img)
        
    if np.max(img) > 1:
        img = img / 255.
    
    hsv_img = rgb2hsv(img)
    gray_img = rgb2gray(img)

    return img, hsv_img, gray_img


@st.cache
def get_random_pts(img_shape, n=1):
    return [(random.randint(0, img_shape[0]), random.randint(0, img_shape[1]))
            for i in range(n)]


def get_random_pts_in_radius(pts, shape, circle_radius=10, n=10):
    tups = []
    for r, c in pts:
        rs, cs = disk((r, c), radius=circle_radius, shape=shape)
        tups += [(r, c) for r, c in zip(rs, cs)]

    return random.choices(tups, k=n)


def plot_target_on_image(img, pts, rect_radius=1, circle_radius=10):
    for r, c in pts:
        rs, cs = circle_perimeter(r, c, circle_radius, shape=img.shape)
        img[rs, cs] = [1., 0., 0.]

    return img


def plot_pts_on_img(img, pts, rect_radius=2):
    for r, c in pts:
        r1, r2 = max(0, r - rect_radius), min(img.shape[0], r + rect_radius)
        c1, c2 = max(0, c - rect_radius), min(img.shape[1], c + rect_radius)
        img[r1:r2, c1:c2] = [1., 0., 0.]

    return img


def get_value_rectangle(shape):
    rect = np.zeros(shape)
    rect[:int(shape[0] * .2), :] = 1.
    for c in range(shape[1]):
        rect[:, c] = c / shape[1]
    rect[:int(shape[0] * .2), :] = 1.
    rect[:, 0] = 0.
    rect[:, shape[1] - 1] = 0.
    rect[0, :] = 0.
    rect[shape[0] - 1, :] = 0.


    rect = gray2rgb(rect)

    return rect


def draw_vline_on_img(img, c, color=(1., 0., 0.), width=2):
    img[:, c] = color

    return img


def add_hues(ax):
    for i in range(0, 360, 5):
        h, s, v = i / 360., 1., 1.
        r, g, b = hsv2rgb(np.asarray([h, s, v]).reshape((1, 1, -1))).flatten()

        p = 0.01745329 * (h * 360)
        x, y = s * np.cos(p), s * np.sin(p)

        ax.scatter(x, y, color=(r, g, b))


box = st.sidebar.checkbox('Show Target', value=True)
submit_button = st.sidebar.button('Submit')
refresh_window = st.sidebar.number_input('Refresh', value=0)

rgb_img, hsv_img, gray_img = load_random_image(seed=refresh_window)

rgb_img, hsv_img, gray_img = np.copy(rgb_img), np.copy(hsv_img), np.copy(gray_img)
og_rgb_img = np.copy(rgb_img)

radius = int(rgb_img.shape[0] * .02)

pts = list(get_random_pts(rgb_img.shape))

if box:
    plot_target_on_image(rgb_img, pts, circle_radius=radius)
    plot_pts_on_img(rgb_img, pts, rect_radius=1)

value_rect = get_value_rectangle((100, 500))

st.image(rgb_img)

value_slider = st.slider('Value guess', 0., 1., value=0., step=.01)

guess_c = int(value_slider * value_rect.shape[1])
true_c = int(gray_img[pts[0][0], pts[0][1]] * value_rect.shape[1])
draw_vline_on_img(value_rect, guess_c, color=(1., 0., 0.))

if submit_button:
    draw_vline_on_img(value_rect, true_c, color=tuple(og_rgb_img[pts[0][0], pts[0][1], :]))

st.image(value_rect)


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlim(-1., 1.)
ax.set_ylim(-1., 1.)
ax.set_zlim(-1., 1.)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)

add_hues(ax2)

r, g, b = og_rgb_img[pts[0][0], pts[0][1]]
h, s, v = hsv_img[pts[0][0], pts[0][1]]

p = 0.01745329 * (h * 360)

x, y = s * np.cos(p), s * np.sin(p)

if submit_button:
    ax.scatter(x, y, v, color=(r, g, b))
    ax2.scatter(x, y, color=(r, g, b))

st.pyplot(fig)

if submit_button:
    fig2, axs = plt.subplots(1, 2)
    axs[0].imshow(rgb_img)
    axs[1].imshow(1. - gray_img, cmap='Greys')
    st.pyplot(fig2)

