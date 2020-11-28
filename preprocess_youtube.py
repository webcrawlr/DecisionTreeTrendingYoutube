import csv
import os
import numpy as np
import pandas as pd
from statistics import mean; #Simple mean function to ease implementations

# Read into pandas dataframe
proc_frames = pd.read_csv("USvideos.csv") 

#  Drop data that will be irrelevant to the decision tree
proc_frames = proc_frames.drop(['video_id', 'trending_date', 'title', 'publish_time', 'description', 'thumbnail_link'], axis = 1)

# Reorganize tags by have or have not
for i, row in proc_frames.iterrows():
    tag_val = 1
    if row['tags'] == "[none]":
        tag_val = 0
    proc_frames.at[i,'tags'] = tag_val

# Cut the views column into 2
# View count above 682,000 will be considered "popular" (it is the median value)
# View count below or equal to that will not be considered popular
bin_result = pd.cut(proc_frames['views'], [0, 682000, 225000000], labels=[0, 1])
proc_frames['views'] = bin_result.tolist()

proc_frames = proc_frames.rename(columns={'views': 'decision'})
# Split likes, dislikes, and comment count into bins by quantiles
bin_result = pd.cut(proc_frames['likes'], [0, 5424, 18100, 55400, 5610000], labels=[0, 1, 2, 3])
proc_frames['likes'] = bin_result.tolist()

bin_result = pd.cut(proc_frames['dislikes'], [0, 202, 631, 1938, 1670000], labels=[0, 1, 2, 3])
proc_frames['dislikes'] = bin_result.tolist()

bin_result = pd.cut(proc_frames['comment_count'], [0, 614, 1856, 5755, 1360000], labels=[0, 1, 2, 3])
proc_frames['comment_count'] = bin_result.tolist()

# Export to set
test_set = proc_frames.sample(random_state = 69, frac = 0.2)
training_set = proc_frames.drop(test_set.index)
test_set.to_csv("testSet-ChannelSpec.csv", index = False, header = True)
training_set.to_csv("trainingSet-ChannelSpec.csv", index = False, header = True)

# Also do ANOTHER set that's not channel specific
proc_frames = proc_frames.drop(['channel_title'], axis = 1)
test_set = proc_frames.sample(random_state = 69, frac = 0.2)
training_set = proc_frames.drop(test_set.index)
test_set.to_csv("testSet-NoChannelSpec.csv", index = False, header = True)
training_set.to_csv("trainingSet-NoChannelSpec.csv", index = False, header = True)


        



