import data

dict_regressors = {"1": "ExtraTreeRegressor", "2": "RandomForestRegressor", "3": "DecisionTreeRegressor",
                   "4": "MLPRegressor", "5": "Lasso", "6": "Ridge", "7": "LinearRegressor"}


def collect_input_one():
    i = raw_input("Input: ")
    while i != str(1) and i != str(2):
        print ("Please insert a valid option")
        i = raw_input("Input: ")

    return int(i)


def collect_input_two():
    i = raw_input("Input: ")
    i = i.lower()
    while i != "y" and i != "n":
        print ("Please insert y or n")
        print ("")
        i = raw_input("Input: ")
        i.lower()

    return 1 if i == "y" else 0


def collect_input_three():

    df = None
    valid = False
    while not valid:
        try:
            df = data.import_test(raw_input("Input:"))
            valid = True
        except IOError:
            print ("The inserted file was not found...")
            print ("Please insert the name of a valid file")
            print ("")

    return df


def collect_input_four():
    i = raw_input("File name: ")

    while i not in dict_regressors.keys():
        print ("Please insert one of the supplied options")
        print ("")
        i = raw_input("Input: ")

    return i



print ("")
print ("************************************************************************************************************")
print ("")
print ("WELCOME TO THE SONAE BIT PROMOTION ESTIMATOR")
print ("")
print ("made with <3 in FEUP")
print ("June 2017")
print ("************************************************************************************************************")
print ("Instructions:")
print ("This is a simple command line tool that provides estimations for a specific problem presented by BIT SONAE.")
print ("The file with the test data shall be put inside the /src/test folder.")
print ("What would you like to accomplish?")
print ("")
print ("1 - Generate training data (preprocessing)")
print ("2 - Make an estimation, given some test data ")
print ("")
if collect_input_one() == 1:
    print ("")
    print ("Does your model allow nominal attributes? (y/n)")
    print ("")
    if collect_input_two() == 1:
        data.export_train(True)
    else:
        data.export_train(False)
else:
    print ("")
    print ("Please insert the name of the file with the test data (eg. test.csv)")
    print ("Remember, the file must be inserted in the src/test directory")
    print ("")
    df = collect_input_three()

    print ("")
    print ("Select one of the following estimators:")
    print ("1 - Extremely Randomized Trees")
    print ("2 - Random Forests")
    print ("3 - Decision Tree")
    print ("4 - Multi Layered Perceptron")
    print ("5 - LASSO")
    print ("6 - Ridge")
    print ("7 - Linear Regression")
    print ("8 - SVM")
    print ("9 - Gradient Tree Boosting")

    print ("")
    opt = collect_input_four()
    model = data.import_model(dict_regressors.get(opt))
    print (model)
