import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#Input and Output placeholders
x=tf.placeholder('float',[None,28*28])
y=tf.placeholder('float',[None,10])

modelPath='/tmp/model.ckpt'

#design network
def neuralNetwork(x):
	#Variable weights and biases
	weights={'hid1':tf.Variable(tf.random_normal([28*28,500])),
			 'hid2':tf.Variable(tf.random_normal([500,500])),
			 'out':tf.Variable(tf.random_normal([500,10])),}

	biases={'b1':tf.Variable(tf.random_normal([500])),
			 'b2':tf.Variable(tf.random_normal([500])),
			 'out':tf.Variable(tf.random_normal([10])),}

	layer1=tf.add(tf.matmul(x,weights['hid1']),biases['b1'])
	layer1=tf.nn.relu(layer1)

	layer2=tf.add(tf.matmul(layer1,weights['hid2']),biases['b2'])
	layer2=tf.nn.relu(layer2)

	output=tf.add(tf.matmul(layer2,weights['out']),biases['out'])
	return output

#train model
def trainModel(x,learnRate=0.001,batch=100,epochs=10):
	prediction=neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))#Activation Fn
	optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)#GD
	
	init=tf.global_variables_initializer()

	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)	

		#Training
		for epoch in range(epochs):
			loss=0

			totalBatch=int(mnist.train.num_examples/batch)
			for i in range(totalBatch):
				batch_x,batch_y = mnist.train.next_batch(batch)

				_,c=sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y})
				loss+=c
			print("Epoch ",(epoch+1)," Loss: ",loss)

		#Evaluate accuracy and save
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ",accuracy.eval({x: mnist.test.images[:256],y: mnist.test.labels[:256]}))
		saver.save(sess,modelPath)

	#test model
	with tf.Session() as sess:
		sess.run(init)	
		saver.restore(sess, modelPath)

		#Evaluate accuracy
		print(tf.argmax(prediction,1).eval({x: mnist.test.images[:1]}))
		print(tf.argmax(mnist.test.labels[:1],1)).eval()
		img=np.array(mnist.test.images[:1]).reshape(28,28)
		cv2.imshow("Result", img)
		cv2.waitKey(0)

def main():
	trainModel(x)

if __name__=="__main__":
	main()