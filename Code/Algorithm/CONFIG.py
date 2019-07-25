#best model
MODEL = 'best_169.h5'

#the number of data you want to predict
TEST_NUM = 11516

#please be sure all data are in the same format
TEST_FORMAT  = 'jpg'

#where to find your data 
TEST_DIR = '../Input/val2014'

#the size of the image
IMAGE_SIZE = 300

#please do not change the LABELS unless you retrained the model
#if model retrained, use (your data generator).class_indices to get the new LABELS
LABELS = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '15', 8: '16', 9: '17', 10: '18', 11: '19', 12: '2', 13: '3', 14: '4', 15: '5', 16: '6', 17: '7', 18: '8', 19: '9'}