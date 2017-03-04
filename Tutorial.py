#Nodes in the graph represent mathematical operations,
#while the graph edges represent the multidimensional data arrays (tensors) communicated between them.
import tensorflow as tf

#---------------------------------
#TF represents computations
x=tf.constant(5)

#These variables represent various computations
y=tf.Variable(x*2)
z=tf.Variable(tf.mul(x,2))

y2=tf.Variable(x+2)
z2=tf.Variable(tf.add(x,2))

#x and y are tensors print they need to be run in a session
print(x)#Outputs: Tensor("Const:0", shape=(), dtype=int32)
print(y)#Outputs: Tensor("Variable/read:0", shape=(), dtype=int32)


#With statement used to open a session, it is automatically closed when finished
with tf.Session() as sess:
	#This statement initalizes all our variables in the current session
	sess.run(tf.global_variables_initializer())

	#Run a particular computation using eval or run
	print(sess.run(y))#Outputs: 10
	print(z.eval())#Outputs: 10

	#An important difference between eval and run is that run can work with multiple inputs
	print(sess.run([y2,z2]))#Outputs: [7, 7] 
	print(sess.run(z2))#Outputs: 7

#Without with the session must be explicitly closed
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y))
print(sess.run(z))

print(sess.run(y2))
print(sess.run(z2))
sess.close()#Close session when finished

#---------------------------------
#A placeholder is a variable that will be assigned data to later
#Usually used to represent inputs and outputs

x=tf.placeholder("float",3)
y=5*(x*x)-(3*x)+15

with tf.Session() as sess:
	print(sess.run(y,feed_dict=({x:[0,1,2]})))#Outputs: [ 15.  17.  29.]

#---------------------------------
#If we don't know the size being put in the placeholder, None is used
x=tf.placeholder("float",[None,3])
y=5*(x*x)-(3*x)+15

with tf.Session() as session:
	#The input data can now consist of many arrays or size 3 rather than just one
    x_data = [[0, 1, 2],
              [3, 4, 5],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)#Outputs: [[  15.   17.   29.] [  51.   83.  125.]]

#---------------------------------
#Simple addition

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	result=sess.run(add,feed_dict={a:2,b:3})#Outputs: 5
	print(result)

#---------------------------------
#Simple addition with tensorboard
#Tensorboard allows us to visualize how the data is 
#flowing in the computational graph through each mathematical operation 

#Adding name scopes allows us to identify what operation tf is refering to
#Names are defaulted to ones which aren't easy to identify
with tf.name_scope("A"):
	a=tf.placeholder(tf.int16)

with tf.name_scope("B"):
	b=tf.placeholder(tf.int16)

with tf.name_scope("ADDER"):
	add=tf.add(a,b)

#Scalars graph the progress of computations
tf.summary.scalar("Result", add)

#All the summaries are merged
mergedSummary = tf.summary.merge_all()

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#Writer to write our graph to a temp location
	summaryWriter = tf.summary.FileWriter("/tmp/tensorflow_logs/example", graph=tf.get_default_graph())

	result,summary=sess.run([add,mergedSummary],feed_dict={a:2,b:3})

	#Summary is added to our temp location
	summaryWriter.add_summary(summary)

	#Run the following command in console after running script and put the link returned into a browser
	#tensorboard --logdir=/tmp/tensorflow_logs

#Saving models covered in NeuralNetwork.py