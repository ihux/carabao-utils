#============================================================================
# Toy Class
# usage: toy = Toy()
#        V = toy.vocab()
#        mary = V("Mary")
#============================================================================

from carabao.htm import Layer

class Toy:
    def vocab(self):
        V = {"Mary": [0,0,1,0,0,1,1,1,0,0],
             "John": [1,0,0,1,0,1,1,0,0,0],
             "likes":[0,1,0,0,0,0,0,1,1,1],
             "to":   [0,1,0,1,1,0,0,0,0,1],
            }
        return V

    def layer(self):
        return Layer(4,10,5,2)

def demo():
    print("toy = Toy()")
    toy = Toy()
    print("V = toy.vocab()")
    V = toy.vocab()
    print("token = V['Mary']")
    token = V['Mary']
    print("print(token)")
    print(token)
