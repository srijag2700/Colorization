import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from neural_network import NeuralNet

img = io.imread('coast.jpg')

# recoloring based on "bins" of colors
# possible RGB values will be 0, 64, 128, 192 because they are multiples & easy division
# wow I love numpy arrays
def recolor(pic):
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            pic[i,j,0] = (pic[i,j,0]//64)*64
            pic[i,j,1] = (pic[i,j,1]//64)*64
            pic[i,j,2] = (pic[i,j,2]//64)*64
    return pic

def get_patch(pic, x, y):
    # get 3x3 patch of pixels -- improved from basic agent!
    patch = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            patch.append(pic[i,j])
    return patch

# one hot codes to make things simple!
# 0 (0)   = [1 0 0 0]
# 1 (64)  = [0 1 0 0]
# 2 (128) = [0 0 1 0]
# 3 (192) = [0 0 0 1]

def one_hot(n):
    oh = []
    for i in range(4):
        if i == n:
            oh.append(1)
        else:
            oh.append(0)
    return oh

# and change back to a normal number
def oh_to_reg(n):
    ret = 0
    for i in range(4):
        if round(n[i]) == 1:
            ret = i
    return ret

# quantitatively calculate difference between expected and result for report
def difference(og, pre):
    wrong = 0
    for i in range(og.shape[0]):
        for j in range(og.shape[1]):
            if not np.array_equal(og[i,j],pre[i,j]):
                wrong += 1
    total = og.shape[0] * og.shape[1]
    return wrong/total

img_rec = recolor(img)

layers = [9,9,4]
# input layer: 9, for the 9 pixels in the patch
# output layer: 4, for the one hot code
learning_rate = 0.1
epochs = 20000

train_bw = color.rgb2gray(img_rec[:, 0:int(img_rec.shape[1]/2), :])
train_color = img_rec[:, 0:int(img_rec.shape[1]/2), :] #this needs to be k-means-ed
predict_bw = color.rgb2gray(img_rec[:, int(img_rec.shape[1]/2):img_rec.shape[1], :])
predict_color = img_rec[:, int(img_rec.shape[1]/2):img_rec.shape[1], :]

# its NEURAL NETWORK time
in_data = []
red_NN = NeuralNet(layers, 'tanh')
red_data = []

gre_NN = NeuralNet(layers, 'tanh')
gre_data = []

blu_NN = NeuralNet(layers, 'tanh')
blu_data = []

# train data & labels
for i in range(1, train_bw.shape[0]-1):
    for j in range(1, train_bw.shape[1]-1):
        in_data.append(get_patch(train_bw,i,j))
        red_data.append(one_hot(train_color[i,j,0]/64))
        gre_data.append(one_hot(train_color[i,j,1]/64))
        blu_data.append(one_hot(train_color[i,j,2]/64))

#print(in_data)

# its training time :D
red_NN.fit(in_data, red_data, learning_rate, epochs)
gre_NN.fit(in_data, gre_data, learning_rate, epochs)
blu_NN.fit(in_data, blu_data, learning_rate, epochs)

# aaaand now prediction time
print("red predict: ", red_NN.predict(get_patch(predict_bw, 100, 100))*64)
print("green predict: ", gre_NN.predict(get_patch(predict_bw, 100, 100))*64)
print("blue predict: ", blu_NN.predict(get_patch(predict_bw, 100, 100))*64)

res = np.zeros([predict_bw.shape[0], predict_bw.shape[1], 3], dtype = np.uint8)

for i in range(1, predict_bw.shape[0]-1):
    for j in range(1, predict_bw.shape[1]-1):
        res[i,j,0] = oh_to_reg(red_NN.predict(get_patch(predict_bw, i, j))) * 64
        res[i,j,1] = oh_to_reg(gre_NN.predict(get_patch(predict_bw, i, j))) * 64
        res[i,j,2] = oh_to_reg(blu_NN.predict(get_patch(predict_bw, i, j))) * 64

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