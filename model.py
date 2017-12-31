import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_processing

data = data_processing.load_data(download=True)
new_data = data_processing.convert2onehot(data)

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)
seq = int(0.7 * len(new_data))
train_data = new_data[:seq]
test_data = new_data[seq:]

tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 400, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accurary = tf.metrics.accuracy(
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1)
)[1]

opt = tf.train.GradientDescentOptimizer(0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

#for t in range(4000):
#    batch_index = np.random.randint(len(train_data), size=32)
#    sess.run(train_op, {tf_input: train_data[batch_index]})

#    if t % 50 == 0:
#        acc_, pred_, loss_ = sess.run([accurary, prediction, loss], {tf_input: test_data})
#        print("step: %i" % t, "| Accurate:%.2f" % acc_, "| Loss:%.2f" % loss_, )

plt.ion()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
accuraries,steps = [],[]
for t in range(4000):
    #training
    batch_index = np.random.randint(len(train_data), size=32)
    sess.run(train_op, {tf_input: train_data[batch_index]})

    if t % 50 ==0:
        acc_, pred_, loss_ = sess.run([accurary, prediction, loss], {tf_input: test_data})
        accuraries.append(acc_)
        steps.append(t)
        print("step: %i" % t, "| Accurate:%.2f" % acc_, "| Loss:%.2f" % loss_, )

        #visualiziong testing
        ax1.cla()
        for c in range(4):
            bp = ax1.bar(left=c+0.1,height=sum((np.argmax(pred_,axis=1)==c)),width=0.2,color='red')
            bt = ax1.bar(left=c-0.1,height=sum((np.argmax(test_data[:,21:],axis=1)==c)),width=0.2,color='blue')

        ax1.set_xticks(range(4),["accepted","good","unaccepted","very good"])
        ax1.legend(handles=[bp,bt],labels=["prediction","target"])
        ax1.set_ylim((0,400))
        ax2.plot(steps,accuraries,label="accuracy")
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)

plt.ioff()
plt.show()