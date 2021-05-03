from sys import argv
import matplotlib.pyplot as plt
from utils import load_csv, filter_dataframe

def scatter_plot(courses_df):
	"Scatter plot all combinations of courses"
	courses = list(courses_df.columns)
	for course in courses:
		c_lst = courses[1:]
		for second_course in c_lst:
			plt.scatter(courses_df[course], courses_df[second_course], label='Students', s=20, alpha=0.8)
			plt.xlabel(course)
			plt.ylabel(second_course)
			plt.legend()
			plt.title(f'Scatter plot of {course} vs {second_course}')
			plt.show()
		courses = courses[1:]

def	main():
	if len(argv) == 2:
		print (argv[1])
		dataframe = load_csv(argv[1])
		if dataframe is None:
			print ("Input a valid file to run the program")
			return
	else:
		print ("Input the dataset to run the program.")
		return
	# Astronomy', 'Defense Against the Dark Arts
	courses_df = filter_dataframe(dataframe)
	courses_df = normalize_dataframe(courses_df)
	scatter_plot(courses_df)
	return

if __name__ == "__main__":
	main()
