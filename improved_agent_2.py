import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from k_means import get_colors
from neural_network import NeuralNet
from basic_agent import five_color, color_distance

img = io.imread('coast.jpg')

colors = get_colors('coast.jpg')

# recoloring based on five representative colors
def five_color(pic):
    five_pic = pic
    for i in range(five_pic.shape[0]):
        for j in range(five_pic.shape[1]):
            min_dist = float('inf')
            min_col = []
            rgb = [five_pic[i,j,0], five_pic[i,j,1], five_pic[i,j,2]]
            for c in colors:
                np_c = np.array(c)
                dist = color_distance(np_c, rgb)
                if dist < min_dist:
                    min_dist = dist
                    min_col = c
            five_pic[i,j,0] = min_col[0]
            five_pic[i,j,1] = min_col[1]
            five_pic[i,j,2] = min_col[2]
    return five_pic

img_rec = five_color(img)

def get_patch(pic, x, y):
    # get 3x3 patch of pixels -- improved from basic agent!
    patch = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            patch.append(pic[i,j])
    return patch

# one hot codes to make things simple!
# 0 = [1 0 0 0 0]
# 1 = [0 1 0 0 0]
# 2 = [0 0 1 0 0]
# 3 = [0 0 0 1 0]
# 4 = [0 0 0 0 1]

def one_hot(n):
    oh = [0, 0, 0, 0, 0]
    ind = colors.index(n.tolist())
    oh[ind] = 1
    return oh

# and change back to a normal color
def oh_to_reg(n):
    one = n.tolist().index(max(n))
    return colors[one]

# quantitatively calculate difference between expected and result for report
def difference(og, pre):
    wrong = 0
    for i in range(og.shape[0]):
        for j in range(og.shape[1]):
            if not np.array_equal(og[i,j],pre[i,j]):
                wrong += 1
    total = og.shape[0] * og.shape[1]
    return wrong/total

layers = [9,9,5]
# input layer: 9, for the 9 pixels in the patch
# output layer: 5, for the one hot code
learning_rate = 0.1
epochs = 20000

train_bw = color.rgb2gray(img_rec[:, 0:int(img_rec.shape[1]/2), :])
train_color = img_rec[:, 0:int(img_rec.shape[1]/2), :] #this needs to be k-means-ed
predict_bw = color.rgb2gray(img_rec[:, int(img_rec.shape[1]/2):img_rec.shape[1], :])
predict_color = img_rec[:, int(img_rec.shape[1]/2):img_rec.shape[1], :]

# its NEURAL NETWORK time
in_data = []
all_NN = NeuralNet(layers, 'tanh')
all_data = []

# train data & labels
for i in range(1, train_bw.shape[0]-1):
    for j in range(1, train_bw.shape[1]-1):
        in_data.append(get_patch(train_bw,i,j))
        all_data.append(one_hot(train_color[i,j]))

#print(in_data)

# its training time :D
all_NN.fit(in_data, all_data, learning_rate, epochs)

# aaaand now prediction time
print("predict: ", all_NN.predict(get_patch(predict_bw, 100, 100)))

res = np.zeros([predict_bw.shape[0], predict_bw.shape[1], 3], dtype = np.uint8)

for i in range(1, predict_bw.shape[0]-1):
    for j in range(1, predict_bw.shape[1]-1):
        res[i,j,0] = oh_to_reg(all_NN.predict(get_patch(predict_bw, i, j)))[0]
        res[i,j,1] = oh_to_reg(all_NN.predict(get_patch(predict_bw, i, j)))[1]
        res[i,j,2] = oh_to_reg(all_NN.predict(get_patch(predict_bw, i, j)))[2]

res.astype(np.int)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.title('Recolored Pic')
plt.imshow(img_rec)

plt.subplot(2,2,3)
plt.title('Right Side - Grayscale')
#plt.imshow(train_color)
plt.imshow(predict_bw, cmap="gray")

plt.subplot(2,2,4)
plt.title('Right Side - Predicted')
plt.imshow(res)

print("Difference: {}".format(difference(predict_color, res)))
plt.show()