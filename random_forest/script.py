import pandas as pd
import sys

def main(inputfile, class_name, output):
	class_col = pd.read_csv(inputfile, usecols=[class_name],sep=',')
	data = pd.read_csv(inputfile, sep=',')
	data = data.drop(class_name, axis=1)
	data[class_name]=class_col
	data.to_csv(output, sep=",")

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
