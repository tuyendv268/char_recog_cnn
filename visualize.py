from matplotlib import pyplot as plt
import numpy as np

h1=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_1_block\\history_mdl1.npy',allow_pickle='TRUE').item()
h2=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_2_block\\history_mdl2.npy',allow_pickle='TRUE').item()
h3=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_3_block\\history_mdl3.npy',allow_pickle='TRUE').item()
h4=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\model_1\\model_4_block\\history_mdl4.npy',allow_pickle='TRUE').item()

x = [i for i in range(1, 11)]

# plt.title("training accuracy")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.plot(x,h1['accuracy'], label = "1 conv-batch-maxp")
# plt.plot(x,h2['accuracy'], label = "2 conv-batch-maxp")
# plt.plot(x,h3['accuracy'], label = "3 conv-batch-maxp")
# plt.plot(x,h4['accuracy'], label = "4 conv-batch-maxp")
# plt.legend()
# plt.show()

# m1=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_3_block\\history_mdl3.npy',allow_pickle='TRUE').item()
# m2=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\tuning\\maxpooling\\history_no_maxpooling.npy',allow_pickle='TRUE').item()

# plt.title("loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.plot(x,m1['loss'],color="blue", label = "train_loss(conv-batch-maxp)")
# plt.plot(x,m2['loss'],color="red",label = "train_loss(conv-batch)")
# # plt.plot(x,m1['val_loss'],color="blue", label = "val_loss(conv-batch-maxp)",linestyle="--")
# # plt.plot(x,m2['val_loss'],color="red",label = "val_loss(conv-batch)",linestyle="--")
# plt.plot(x,m1['val_loss'],color="blue", label = "val_loss(conv-batch-maxp)",linestyle="--")
# plt.plot(x,m2['val_loss'],color="red",label = "val_loss(conv-batch)",linestyle="--")
# plt.legend()

# plt.show()

# m1=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_3_block\\history_mdl3.npy',allow_pickle='TRUE').item()
# m2=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\tuning\\batchnorm\\history_no_batchnorm.npy',allow_pickle='TRUE').item()

# plt.title("loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.plot(x,m1['loss'],color="blue", label = "train_loss(conv-batch-maxp)")
# plt.plot(x,m2['loss'],color="red",label = "train_loss(conv-maxp)")
# plt.plot(x,m1['val_loss'],color="blue", label = "val_loss(conv-batch-maxp)",linestyle="--")
# plt.plot(x,m2['val_loss'],color="red",label = "val_loss(conv-batch)",linestyle="--")
# # plt.plot(x,m1['val_accuracy'],color="blue", label = "val_accuracy(conv-batch-maxp)",linestyle="--")
# # plt.plot(x,m2['val_accuracy'],color="red",label = "val_accuracy(conv-maxp)",linestyle="--")
# plt.legend()

# plt.show()

m1=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\training\\model_1\\model_3_block\\history_mdl3.npy',allow_pickle='TRUE').item()
m2=np.load('C:\\Users\\tuyen.dv\\OneDrive\\Máy tính\\NMAI\\tuning\\dropout\\history_no_dropout.npy',allow_pickle='TRUE').item()

plt.title("accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x,m1['accuracy'],color="blue", label = "train_accuracy(có dropout)")
plt.plot(x,m2['accuracy'],color="red",label = "train_accuracy(không dropout)")
# plt.plot(x,m1['val_loss'],color="blue", label = "val_loss(conv-batch-maxp)",linestyle="--")
# plt.plot(x,m2['val_loss'],color="red",label = "val_loss(conv-batch)",linestyle="--")
plt.plot(x,m1['val_accuracy'],color="blue", label = "val_accuracy(có dropout)",linestyle="--")
plt.plot(x,m2['val_accuracy'],color="red",label = "val_accuracy(không dropout)",linestyle="--")
plt.legend()

plt.show()