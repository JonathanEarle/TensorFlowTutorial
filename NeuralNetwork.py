import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data

#One hot encoding means [0,0,0,1,0,0,0,0,0,0,0] represents 3 [0,0,0,0,1,0,0,0,0,0,0] represents 4 and so forth
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#Input and Output placeholders
x=tf.placeholder('float',[None,28*28])#28*28 size images are the inputs
y=tf.placeholder('float',[None,10])#An array of size 10 stores the one hot encoded results

#Path to save the model to
modelPath='/tmp/model.ckpt'

#Model of the neural net
def neuralNetwork(x):
	#Variable weights and biases, randomly initalized
	weights={'hid1':tf.Variable(tf.random_normal([28*28,500])),
			 'hid2':tf.Variable(tf.random_normal([500,500])),
			 'out':tf.Variable(tf.random_normal([500,10])),}

	biases={'b1':tf.Variable(tf.random_normal([500])),
			 'b2':tf.Variable(tf.random_normal([500])),
			 'out':tf.Variable(tf.random_normal([10])),}

	#A single layer multiplies its input by the weights and adds the bias
	#relu (Rectified linear unit) is our activation function
	layer1=tf.add(tf.matmul(x,weights['hid1']),biases['b1'])
	layer1=tf.nn.relu(layer1)

	#Each layer takes in the output from the previous layer and applies the operations
	layer2=tf.add(tf.matmul(layer1,weights['hid2']),biases['b2'])
	layer2=tf.nn.relu(layer2)

	output=tf.add(tf.matmul(layer2,weights['out']),biases['out'])
	return output

#Training the network on data (x),segmenting the data in 100 batches and repeating this for 10 epochs
def trainModel(x,learnRate=0.001,batch=100,epochs=10):
	prediction=neuralNetwork(x)

	#Activation Function applied to the output layer, cross entropy is our cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#Adam Optimizer used for cost minimization
	optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)
	
	init=tf.global_variables_initializer()

	#Saver to save our model
	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)	

		#Training
		for epoch in range(epochs):
			loss=0#stores our total loss at each epoch
			totalBatch=int(mnist.train.num_examples/batch)
			for i in range(totalBatch):
				batch_x,batch_y = mnist.train.next_batch(batch)

				#Actually running the process with a batch of data and store the loss (_ results from optimizer)
				_,c=sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y})
				loss+=c
			print("Epoch ",(epoch+1)," Loss: ",loss)

		#Evaluate accuracy and save
		#argmax returns index of largest value in the output (highest confidence score) along axis 1 (each row)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy: ",accuracy.eval({x: mnist.test.images[:256],y: mnist.test.labels[:256]}))

		#Aave the session, with the trained weights for reuse
		saver.save(sess,modelPath)

	#Open a new session and attempt to restore the model
	with tf.Session() as sess:
		sess.run(init)	
		saver.restore(sess, modelPath)

		#See the result given by the model on one test image
		print(tf.argmax(prediction,1).eval({x: mnist.test.images[:1]}))#Neural network answer
		print(tf.argmax(mnist.test.labels[:1],1)).eval()#Actual answer
		
		#Display the image
		img=np.array(mnist.test.images[:1]).reshape(28,28)
		cv2.imshow("Result", img)
		cv2.waitKey(0)

def main():
	trainModel(x)

if __name__=="__main__":
	main()