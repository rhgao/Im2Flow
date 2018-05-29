FILE=$1
URL=http://vision.cs.utexas.edu/projects/im2flow/models/$FILE.t7
MODEL_FILE=./model/$FILE.t7
wget -N $URL -O $MODEL_FILE
