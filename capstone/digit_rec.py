import data_d

#Read Dataset from the path
mnist = data_d.read_data_sets("/root/Desktop/projects/capstone/project/data", one_hot=True)
print "Data Reading Done"

import tensorflow as tflow


#HyperParameters for model Tuning and optimising it for best accuracy
learning_rate=0.01
training_itrn = 50
batch_size = 64
display_step = 2

#TensorFlow graph input
x = tflow.placeholder("float", [None,784])
y = tflow.placeholder("float", [None,10])
print "Graph input Done"


########################################################################################

#Now Creating the Model

#Initilizing Model Wights and Grpahs to Zero
W = tflow.Variable(tflow.zeros([784,10]))
b = tflow.Variable(tflow.zeros([10]))

print "Model Weight and baises initialization to Zero Done"


with tflow.name_scope("Wx_b") as scope:
#using Softmax function
    model = tflow.nn.softmax(tflow.matmul(x, W)+b)

print "Softmax Done"

#Histogram Data
w_h = tflow.summary.histogram("weights", W)
b_h = tflow.summary.histogram("biases", b)

print "Summary Histogram done"

#Making more name scops for better graph Representation
with tflow.name_scope("cost_function") as scope:
    #Using Cross Entropy for Error Minimisation

    cost_function = -tflow.reduce_sum(y*tflow.log(model))

    #Monitoring cost function using summary monitor

    tflow.summary.scalar("cost_function",cost_function)

print "Cost Function Done"

with tflow.name_scope("train") as scope:
    #using Gradinent Descent optimiser
    optimizer = tflow.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

print "Gradient Descent Optimizer Done"


#Now Lets intitalize the variables
init = tflow.initialize_all_variables()

print "Initialising the Variables Done"

#Summaries into a single operator Merged
m_summ_op = tflow.summary.merge_all()

#Its the fu part now lets launch tha graph
with tflow.Session() as sson:
    sson.run(init)

    #Setting the path to log writer
    summ_wr = tflow.summary.FileWriter('/root/Desktop/projects/capstone/project/graph/', graph_def=sson.graph_def)

    #Training Cycle
    for itrn in range(training_itrn):
        av_cst = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            btch_x, btch_y = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sson.run(optimizer, feed_dict={x: btch_x, y: btch_y})
            # Compute the average loss
            av_cst += sson.run(cost_function, feed_dict={x: btch_x, y: btch_y})/total_batch
            # Write logs for each itrn
            summ_st = sson.run(m_summ_op, feed_dict={x: btch_x, y: btch_y})
            summ_wr.add_summary(summ_st, itrn*total_batch + i)
        # Display logs per itrn step
        if itrn % display_step == 0:
            print "iteration:", '%04d' % (itrn + 1), "cost=", "{:.9f}".format(av_cst)
	

    print "Tuning completed!"

    # Test the model
    predictions = tflow.equal(tflow.argmax(model, 1), tflow.argmax(y, 1))
	
    print "Predictions Done"

    # Calculate accuracy
    accuracy = tflow.reduce_mean(tflow.cast(predictions, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
