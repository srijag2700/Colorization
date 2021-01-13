import numpy as np

# ACTIVATION FUNCTION STUFF

def logistic(a):
        return 1/(1+np.exp(-a)) 

def tanh(a):
    return np.tanh(a)

def logistic_der(a): #derivative
    return logistic(a)*(1-logistic(a))

def tanh_der(a): #derivative
    return 1.0 - np.tanh(a)**2

# NEURAL NETWORK STUFF

class NeuralNet:
    def __init__(self, list_layers, ac='tanh'):
        # list_layers = number of units per layer
        # a = activation function

        if ac == 'logistic':
            self.activation = logistic
            self.activation_der = logistic_der
        elif ac == 'tanh':
            self.activation = tanh
            self.activation_der = tanh_der
        
        self.weights = []
        # initializing random but small weights
        for i in range(1,len(list_layers)):
            if i == len(list_layers)-1:
                self.weights.append((2*np.random.random((list_layers[i-1] + 1, list_layers[i]))-1)*0.25)
            else:
                self.weights.append((2*np.random.random((list_layers[i-1]+1, list_layers[i]+1))-1)*0.25)
    
    def fit(self, input_data, label, learning_rate=0.2, epochs=10000):
        # all parameter names are self explanatory tbh
        input_data = np.atleast_2d(input_data)
        t = np.ones([input_data.shape[0], input_data.shape[1]+1])
        t[:, 0:-1] = input_data #add bias
        input_data = t
        label = np.array(label)

        for i in range(epochs):
            ran = np.random.randint(input_data.shape[0])
            curr = [input_data[ran]]

            for j in range(len(self.weights)):
                #forward
                curr.append(self.activation(np.dot(curr[j], self.weights[j])))
            
            err = label[ran] - curr[-1]
            delt = [err * self.activation_der(curr[-1])]

            for j in range(len(curr)-2, 0, -1):
                #back
                delt.append(delt[-1].dot(self.weights[1].T)*self.activation_der(curr[1]))
            
            delt.reverse()

            for k in range(len(self.weights)):
                #update
                l = np.atleast_2d(curr[k]) #layer
                d = np.atleast_2d(delt[k]) #delta
                self.weights[k] += learning_rate * l.T.dot(d)
    
    def predict(self, input_data):
        input_data = np.array(input_data)
        t = np.ones(input_data.shape[0]+1)
        t[0:-1] = input_data
        prediction = t
        for i in range(0, len(self.weights)):
            prediction = self.activation(np.dot(prediction, self.weights[i]))
        return prediction
