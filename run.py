#!/usr/bin/env python

import sys
import os
import json



#import model
#import optimal trans
#import testmetric



# 
# parser.add_argument('--batch_size', type = int, default = 1,
#         help = 'input batch size for training (default: 1)')

# parser.add_argument('--z_score',dest = 'normalization', action='store_const',default = data.min_max_normalize, const = data.z_score_normalize,help = 'use z-score normalization on the dataset, default is min-max normalization')


def main(targets):
	#target=targets.target
    data_to_use=targets[0]
    if data_to_use =='test':
		#TODO: load test data
        print('test on test data')
    elif data_to_use == 'all':
	#TODO: load all data

        print('run on all data')
    elif data_to_use == 'clean':
	#TODO: clean all the generate result files

        print('clear to clean repo')
    else:
        print('No clear instruction')

	#TODO: 
	#train classifier on given data
	#apply classifier on given data

	#train OT on given data
	#apply OT on given data

	#test, validate result, produce output


# # RUN Way1
#import argparse
#parser = argparse.ArgumentParser(description = 'DSC180A Project')
#parser.add_argument("target", type=str)
#main(parser.parse_args())

## Run Way2
if __name__ == '__main__': #if run from command line
    targets = sys.argv[1:]
    main(targets)
