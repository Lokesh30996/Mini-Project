import numpy as np
import utils
from RNTNModel import RNTNModel
class ComputeCostAndGradMiniBatch:
    def __init__(self):
        self.dictionary = None
        self.trees_train = None
        self.trees_dev = None
        self.loss = 0.0
        self.dJ_dWs = None
        self.dJ_dL = None
        self.dJ_dW = None
        self.dJ_dV = None
    def compute(self, theta, dictionary, trees_train, trees_dev=None):
        self.dictionary = dictionary
        self.trees_train = trees_train
        self.trees_dev = trees_dev
        model = RNTNModel(self.dictionary)
        if theta is not None:
            model.updateParamsGivenTheta(theta)
        cost = 0.0
        grad = np.zeros(model.num_parameters)
        self.loss = 0.0
        self.dJ_dWs = np.zeros(model.Ws.shape)
        self.dJ_dL = np.zeros(model.L.shape)
        self.dJ_dW = np.zeros(model.W.shape)
        self.dJ_dV = np.zeros(model.V.shape)
        tree_train_clone = []
        for tree in self.trees_train:
            cloned_tree = tree.clone()
            self.forwardPass(model, cloned_tree)
            tree_train_clone.append(cloned_tree)
        scaler = 1.0 / len(self.trees_train)
        cost = self.loss*scaler + self.calculateRegularizationCost(model)
        # Backprop on cloned trees
        for tree in tree_train_clone:
            dJ_dz_prop = np.zeros(model.dim)
            self.backwardPass(model, tree, dJ_dz_prop)
        grad = self.calculateTotalGradient(model, scaler)
        return cost, grad
    def forwardPass(self, model, tree):
        if tree.is_leaf():
            word_index = self.getWordIndex(model, tree.word)
            tree.word_vector = model.L[:, word_index]
        else:
            left_child = tree.subtrees[0]
            right_child = tree.subtrees[1]
            self.forwardPass(model, left_child)
            self.forwardPass(model, right_child)
            tree.word_vector = self.composition(
                model, left_child.word_vector, right_child.word_vector)
        tree.word_vector = np.tanh(tree.word_vector)
        tree.prediction = utils.softmax(model.Ws.dot(np.append(tree.word_vector, [1])))
        label_vector = self.getLabelVector(model, tree.label)
        self.loss += -1*label_vector.dot(np.log(tree.prediction))
    def backwardPass(self, model, tree, dJ_dz_prop):
        word_vector_with_bias = np.append(tree.word_vector, [1])
        prediction_diff = tree.prediction - self.getLabelVector(model, tree.label)
        self.dJ_dWs += np.outer(prediction_diff, word_vector_with_bias)
        assert self.dJ_dWs.shape == model.Ws.shape,"classification matrix dim is incorrect"
        dJ_dz_pred = model.Ws[:, :-1].T.dot(prediction_diff)*(1 - tree.word_vector**2)
        dJ_dz_full = dJ_dz_pred + dJ_dz_prop
        if tree.is_leaf():
            word_index = self.getWordIndex(model, tree.word)
            self.dJ_dL[:, word_index] += dJ_dz_full
        else:
            c_vector = np.hstack([tree.subtrees[0].word_vector, tree.subtrees[1].word_vector])
            self.dJ_dW += np.outer(dJ_dz_full, np.append(c_vector, [1]))
            assert self.dJ_dW.shape == model.W.shape,"composition W dim is incorrect"
            if model.use_tensor:
                self.dJ_dV += np.tensordot(dJ_dz_full, np.outer(c_vector, c_vector), axes=0).T
            dJ_dz_down = model.W[:, :-1].T.dot(dJ_dz_full)
            if model.use_tensor:
                dJ_dz_down += (model.V + np.transpose(model.V, axes=[1,0,2])).T.dot(c_vector).T.dot(dJ_dz_full)
            assert dJ_dz_down.size == model.dim*2,"down gradient dim is incorrect"
            dJ_dz_down = dJ_dz_down * (1 - c_vector**2)
            dJ_dz_down_left = dJ_dz_down[:model.dim]
            dJ_dz_down_right = dJ_dz_down[model.dim:]
            assert dJ_dz_down_left.size == dJ_dz_down_right.size,"down gradient left&right dim mismatch"
            self.backwardPass(model, tree.subtrees[0], dJ_dz_down_left)
            self.backwardPass(model, tree.subtrees[1], dJ_dz_down_right)
    def calculateRegularizationCost(self, model):
        reg = 0.0
        reg = model.lambda_Ws/2 * np.linalg.norm(model.Ws)**2
        reg += model.lambda_L/2 * np.linalg.norm(model.L)**2
        reg += model.lambda_W/2 * np.linalg.norm(model.W)**2
        if model.use_tensor:
            reg += model.lambda_V/2 * np.linalg.norm(model.V)**2
        return reg
    def calculateTotalGradient(self, model, scaler):
        grad = np.zeros(model.num_parameters)
        self.dJ_dWs *= scaler
        self.dJ_dL *= scaler
        self.dJ_dW *= scaler
        self.dJ_dWs += model.lambda_Ws * model.Ws
        self.dJ_dL += model.lambda_L * model.L
        self.dJ_dW += model.lambda_W * model.W
        if model.use_tensor:
            self.dJ_dV *= scaler
            self.dJ_dV += model.lambda_V * model.V
            grad = utils.vectorizeParams(
                self.dJ_dWs, self.dJ_dL, self.dJ_dW, self.dJ_dV)
        else:
            grad = utils.vectorizeParams(
                self.dJ_dWs, self.dJ_dL, self.dJ_dW)

        return grad
    def composition(self, model, child1, child2):
        c_vector = np.hstack([child1, child2])
        word_vector = model.W.dot(np.append(c_vector, [1]))
        if model.use_tensor:
            word_vector += c_vector.T.dot(model.V.T).dot(c_vector)
        return word_vector
    def getLabelVector(self, model, label):
        label_vector = np.zeros(model.K)
        node_label = (int)(label)
        label_vector[node_label] = 1
        return label_vector
    def getWordIndex(self, model, word):
        if word in model.word_lookup:
            word_index = model.word_lookup[word]
        else:
            word_index = model.word_lookup[model.UNKNOWN_WORD]
        return word_index