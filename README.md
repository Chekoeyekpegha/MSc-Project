# MSc-Project

This project is all about data streams. Imaging streams of images coming in to a neural network one by one for classification. The model, exstream, is built to prevent the neural network from forgetting experiences or knowledge from old data. This is one of the major challenges of using deep learning on on data streams, This phenonemon is called catastrophic forgetting.





Exstream tries to mix old data with new data in one or more buffers or memory system. when all buffers get full, a consolidation technique is used to compress only useful information, the irrevalnt ones gets discarded. so it stores information about the training experiences occuring in thenetwork. There are several experiences because new data arrives and new conclusion ought to be derived to prevent bias. 
