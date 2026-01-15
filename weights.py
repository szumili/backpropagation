import numpy as np
from tqdm import tqdm

def load_weights(weights_file):

    W = []

    file = open((weights_file), 'r')

    data = file.read()
    data = data.strip("\n")
    split_str = data.split("[")
    
    #for i, s in enumerate(tqdm(split_str[1:], desc="Loading weights: %s" % weights_file)):
    for s in split_str[1:]:
        temp_list = []
        split_s = s.split("]")
        #print(len(split_s[0]))
        if len(split_s) > 1:
            split_s[0] = split_s[0].replace('\n', '')
            #print(len(split_s[0]))
            temp_list.append(split_s[0].strip('\n').split(" "))
            #print(len(temp_list[0]))
            temp_list_2 = []
            #temp_list_2 = [float(el) for el in temp_list[0] if len(el) > 0]
            #temp_list_2 = [float(el) for el in temp_list[0] if el]
            for el in temp_list[0]:
                if len(el)>0:
                    temp_list_2.append(float(el))
            W.append(np.array(temp_list_2))

    #print(weights_file)
    #print(len(W)) #1568, 130, 64, 32
    #print(len(W[0])) #130, 64, 32, 10

    return np.array(W)
