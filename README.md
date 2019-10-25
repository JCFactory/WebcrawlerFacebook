# WebcrawlerFacebook
WebcrawlerFacebook for lecture "Big Data Analyseprojekt"


## Link to Tutorial for Sentiment Analysis with Neural Network:

https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456


## Commands to setup Tensorflow (MacOS)
see also: https://www.tensorflow.org/install/pip

1) virtualenv --system-site-packages -p python3 ./venv
2) source ./venv/bin/activate  # sh, bash, ksh, or zsh
3) pip install --upgrade pip
4) pip list  # show packages installed within the virtual environment

After using Tensorflow, deactivate the virtual env: 
5) deactivate  # don't exit until you're done using TensorFlow

6) pip inst all --upgrade tensorflow
7) Verify install: 
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"



If the above commands do not work, use the following:

1) mkdir ts1
2) cd ts2
3) python3.7 -m venv .
4) pip install --upgrade pip
5) pip install tensorflow
