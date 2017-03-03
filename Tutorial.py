#Nodes in the graph represent mathematical operations,
#while the graph edges represent the multidimensional data arrays (tensors) communicated between them.
import tensorflow as tf

#---------------------------------
#TF represents computations
x=tf.constant(5)

y=tf.Variable(x*2)
z=tf.Variable(tf.mul(x,2))

y2=tf.Variable(x+2)
z2=tf.Variable(tf.add(x,2))

print(x)
print(y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(y))
	print(z.eval())

	print(sess.run(y2))
	print(sess.run(z2))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y))
print(sess.run(z))

print(sess.run(y2))
print(sess.run(z2))
sess.close()

#---------------------------------
#A placeholder is a variable that will be assigned data to later
#Usually used to represent inputs and outputs

x=tf.placeholder("float",3)
y=5*(x*x)-(3*x)+15

with tf.Session() as sess:
	print(sess.run(y,feed_dict=({x:[0,1,2]})))

#---------------------------------
#If we don't know the size

x=tf.placeholder("float",[None,3])
y=5*(x*x)-(3*x)+15

with tf.Session() as session:
    x_data = [[0, 1, 2],
              [3, 4, 5],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)

#---------------------------------
#Simple addition

a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
add=tf.add(a,b)

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	result=sess.run(add,feed_dict={a:2,b:3})
	print(result)

#---------------------------------
#Simple addition tensorboard

with tf.name_scope("A"):
	a=tf.placeholder(tf.int16)

with tf.name_scope("B"):
	b=tf.placeholder(tf.int16)

with tf.name_scope("ADDER"):
	add=tf.add(a,b)

tf.summary.scalar("Result", add)
merged_summary_op = tf.summary.merge_all()

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	summary_writer = tf.summary.FileWriter("/tmp/tensorflow_logs/example", graph=tf.get_default_graph())

	result,summary=sess.run([add,merged_summary_op],feed_dict={a:2,b:3})

	summary_writer.add_summary(summary)

	#tensorboard --logdir=/tmp/tensorflow_logs

#Saving models covered in MLP