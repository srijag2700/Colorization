import numpy as np
from numpy.lib.function_base import diff
from skimage import io, color
import matplotlib.pyplot as plt
from k_means import get_colors
from sklearn.neighbors import NearestNeighbors #for KDTree
from collections import Counter

five_colors = get_colors('coast120.jpg', 5)
print(five_colors) # - format: list of lists

img = io.imread('coast120.jpg')
img = color.convert_colorspace(img, 'RGB', 'RGB')

def color_distance(rgb1, rgb2):
    # formula for color distance from https://www.compuphase.com/cmetric.htm
    rm = (rgb1[0] + rgb2[0])/2
    r = rgb1[0] - rgb2[0]
    g = rgb1[1] - rgb2[1]
    b = rgb1[2] - rgb2[1]
    rgb1 = np.array(rgb1)
    rgb2 = np.array(rgb2)
    return sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5

def five_color(pic):
    five_pic = pic
    for i in range(five_pic.shape[0]):
        for j in range(five_pic.shape[1]):
            min_dist = float('inf')
            min_col = []
            rgb = [five_pic[i,j,0], five_pic[i,j,1], five_pic[i,j,2]]
            for c in five_colors:
                np_c = np.array(c)
                dist = color_distance(np_c, rgb)
                if dist < min_dist:
                    min_dist = dist
                    min_col = c
            five_pic[i,j,0] = min_col[0]
            five_pic[i,j,1] = min_col[1]
            five_pic[i,j,2] = min_col[2]
    return five_pic

img_five = five_color(img)

train_bw = color.rgb2gray(img_five[:, 0:int(img_five.shape[1]/2), :])
train_color = img_five[:, 0:int(img_five.shape[1]/2), :] #this needs to be k-means-ed
predict_bw = color.rgb2gray(img[:, int(img.shape[1]/2):img.shape[1], :])
actual_color = img[:, int(img.shape[1]/2):img.shape[1], :]
predict_color = actual_color.copy()

def get_avg_color(pic, x, y):
    # using surrounding pixels, get average color for that "patch"
    num_in_patch = 1
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    total_r = pic[x,y,0]
    total_g = pic[x,y,1]
    total_b = pic[x,y,2]
    pix_len = pic.shape[0]
    pix_wid = pic.shape[1]
    for d in directions:
        new_x = x-d[0]
        new_y = y-d[1]
        if new_x < 0 or new_x >= pix_len or new_y < 0 or new_y >= pix_wid:
            continue
        total_r += pic[new_x, new_y, 0]
        total_g += pic[new_x, new_y, 1]
        total_b += pic[new_x, new_y, 2]
        num_in_patch += 1
    avg_r = int(total_r / num_in_patch)
    avg_g = int(total_g / num_in_patch)
    avg_b = int(total_b / num_in_patch)
    return [avg_r, avg_g, avg_b]

def get_avg_bw(pic, x, y):
    # using surrounding pixels, get average color for that "patch"
    num_in_patch = 1
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    total_r = pic[x,y]*255 
    total_g = pic[x,y]*255 
    total_b = pic[x,y]*255
    pix_len = pic.shape[0]
    pix_wid = pic.shape[1]
    for d in directions:
        new_x = x+d[0]
        new_y = y+d[1]
        if new_x < 0 or new_x >= pix_len or new_y < 0 or new_y >= pix_wid:
            continue
        new_val = pic[new_x, new_y]*255
        total_r += new_val
        total_g += new_val
        total_b += new_val
        num_in_patch += 1
    avg_r = int(total_r / num_in_patch)
    avg_g = int(total_g / num_in_patch)
    avg_b = int(total_b / num_in_patch)
    return [avg_r, avg_g, avg_b]

def get_all_avgs(pic):
    avgs = {}
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            avgs[(i,j)] = get_avg_bw(pic,i,j)
    return avgs

# quantitatively calculate difference between expected and result for report
def difference(og, pre):
    wrong = 0
    for i in range(og.shape[0]):
        for j in range(og.shape[1]):
            if not np.array_equal(og[i,j],pre[i,j]):
                wrong += 1
    total = og.shape[0] * og.shape[1]
    return wrong/total

def agent(p, train_bw, train_color, predict_color):
    # for each patch in predict_bw...
    # find 6 most similar patches from train_bw (nearest neighbor / kdtree or whatever)
    # then look @ those same center pixels in train_color
    # figure out which one appears the most and assign that color to the center pixel in the current patch

    print("Basic agent started")
    bw_avgs = get_all_avgs(train_bw)
    list_bw_avgs = np.array(list(bw_avgs.keys()))
    tree = NearestNeighbors(n_neighbors=6, algorithm='kd_tree')

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            r = int(p[i,j]*255)
            g = int(p[i,j]*255)
            b = int(p[i,j]*255)
            passed = False
            num_loops = 0
            sample_avgs = []
            closest_colors = []

            while not passed:
                #sample_avgs = random.sample(bw_avgs.keys(), 250)
                #closest_colors = [a for a in list_bw_avgs if color_distance(bw_avgs.get(tuple(a)), get_avg_bw(p,i,j)) <= 15]
                closest_colors = [a for a in list_bw_avgs if color_distance(bw_avgs.get(tuple(a)), [r,g,b]) <= 12]
                if len(closest_colors) >= 6:
                    passed = True
                else:
                    if num_loops > 10:
                        closest_colors = [a for a in list_bw_avgs if color_distance(bw_avgs.get(tuple(a)), [r,g,b]) <= 30]
                        break
                num_loops += 1

            tree.fit(closest_colors)
            near_neighbors = tree.kneighbors([[i,j]], 6, return_distance=False)
            near_data = [list_bw_avgs[y] for (x,y) in np.ndenumerate(near_neighbors)]
            near_colors = [bw_avgs.get(tuple(x))[0] for x in near_data]

            color_counts = Counter(near_colors)
            popular_color = max(color_counts, key=lambda key: color_counts[key])

            squares = [tuple(x) for x in near_data if bw_avgs.get(tuple(x))[0] == popular_color]
            squares.sort()

            target_square_r = train_color[squares[0][0], squares[0][1], 0]
            target_square_g = train_color[squares[0][0], squares[0][1], 1]
            target_square_b = train_color[squares[0][0], squares[0][1], 2]

            predict_color[i,j,0] = target_square_r
            predict_color[i,j,1] = target_square_g
            predict_color[i,j,2] = target_square_b
    
    return predict_color

if __name__ == '__main__':
    predict_color = agent(predict_bw, train_bw, train_color, predict_color)
    print("Color predicted")

    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.title('Recolored Pic')
    plt.imshow(img_five)

    plt.subplot(2,2,3)
    plt.title('Right Side - BW')
    #plt.imshow(train_color)
    plt.imshow(predict_bw, cmap="gray")

    plt.subplot(2,2,4)
    plt.title('Right Side - Test Data')
    plt.imshow(predict_color)

    print("Difference: {}".format(difference(actual_color, predict_color)))
    plt.show()