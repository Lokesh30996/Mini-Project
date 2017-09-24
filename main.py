import argparse
import readPTB
import utils
from Train_LBFGS import Train_LBFGS
import pickle
import time
from Test import Test
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNTN')
    parser.add_argument('--trainPath', type=str,
                        default='data/train.txt')
    parser.add_argument('--devPath', type=str,
                        default='data/dev.txt')
    parser.add_argument('--testPath', type=str,
                        default='data/test.txt')
    args = parser.parse_args()
    trees_train = readPTB.parser(args.trainPath)
    trees_dev = readPTB.parser(args.devPath)
    trees_test = readPTB.parser(args.testPath)
    print '[Read] parsed train:%d, dev:%d, test:%d sentences' \
          %(len(trees_train), len(trees_dev), len(trees_test))
    dictionary = utils.constructCompactDictionary(trees_train)
    print '[Read] built dictionary of size %d' % (len(dictionary))
    # Train
    trainObj = Train_LBFGS(dictionary, trees_train)
    optResult = trainObj.train()
    # Save final model
    savefilename = "optResult-RNTN-LBFGS-Final-" + time.strftime("%Y%m%d-%H%M%S")
    with open(savefilename, 'wb') as output:
        pickle.dump(optResult, output, -1)
    # Test on train data
    theta_opt = optResult.x
    testObj_train = Test(dictionary, trees_train)
    tree_accuracy_train, root_accuracy_train = testObj_train.test(theta_opt)
    print "[Train accuracy] tree: %.2f, root: %.2f" %\
          (tree_accuracy_train, root_accuracy_train)
    # Test on test data
    testObj_test = Test(dictionary, trees_test)
    tree_accuracy_test, root_accuracy_test = testObj_test.test(theta_opt)
    print "[Test accuracy] tree: %.2f, root: %.2f" %\
          (tree_accuracy_test, root_accuracy_test)

