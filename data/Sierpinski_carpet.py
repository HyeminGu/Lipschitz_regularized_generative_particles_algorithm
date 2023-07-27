#!/usr/bin/env python3
## 2D samples from Sierpinski carpet
## Code from https://www.geeksforgeeks.org/python-sierpinski-carpet/
import numpy as np

if __name__ == "__main__":
    # %%
    # importing necessary modules
    from PIL import Image

    # total number of times the process will be repeated
    total = 5

    # size of the image
    size = 3**total

    # creating an image
    square = np.empty([size, size, 3], dtype = np.uint8)
    color = np.array([255, 255, 255], dtype = np.uint8)

    # filling it black
    square.fill(0)

    for i in range(0, total + 1):
        stepdown = 3**(total - i)
        for x in range(0, 3**i):
            
            # checking for the centremost square
            if x % 3 == 1:
                for y in range(0, 3**i):
                    if y % 3 == 1:
                        
                        # changing its color
                        square[y * stepdown:(y + 1)*stepdown, x * stepdown:(x + 1)*stepdown] = color

    # saving the image produced
    save_file = "sierpinski.jpg"
    Image.fromarray(square).save(save_file)

    # displaying it in console
    i = Image.open("sierpinski.jpg")
    print(square[0,0], square[1,1])
    i.show()


    # %%

def calc_sierpinski(steps):
    # size of the image
    size = 3**steps

    # creating an image
    square = np.empty([size, size, 3], dtype = np.uint8)
    color = np.array([255, 255, 255], dtype = np.uint8)

    # filling it black
    square.fill(0)

    for i in range(0, steps + 1):
        stepdown = 3**(steps - i)
        for x in range(0, 3**i):
            
            # checking for the centremost square
            if x % 3 == 1:
                for y in range(0, 3**i):
                    if y % 3 == 1:
                        
                        # changing its color
                        square[y * stepdown:(y + 1)*stepdown, x * stepdown:(x + 1)*stepdown] = color
                        
    return square

'''
def generate_sierpinski(
    size, #: tuple[int], # (N, d)
    steps, #: int,
    scale, #: float,
    random_seed: int,
):
    # sampling region: [0, 3^steps/scale) x [0, 3^steps/scale)
    X = np.zeros(size)
    square = calc_sierpinski(steps)
    
    n = 0
    while n < size[0]:
        trial = 3**steps * np.random.random(size[1]) #/scale
        x1, x2 = np.floor(trial)
        if square[int(x1), int(x2), 0] == 0: # *scale
            X[n] = trial/scale
            n += 1
            
    
    return X
'''
def generate_sierpinski(
    size, #: tuple[int], # (N, d)
    steps, #: int,
    scale, #: float,
    random_seed: int,
):
    # sampling region: [0, 3^steps/scale) x [0, 3^steps/scale)
    square = calc_sierpinski(steps)
    
    N = int(sum(sum(1-square[:,:,0]/255)))
    X = np.zeros((N, size[1]))
    n=0
            
    for i in range(square.shape[0]):
        for j in range(square.shape[1]):
            if square[i, j, 0] == 0:
                X[n] = scale/3**steps *(np.random.random(size[1]) + np.array([i,j]))
                n += 1
    
    return X

def embed_data(
    x,
    di: int,
    df: int,
    offset: float
):
    X = np.concatenate((offset*np.ones([x.shape[0],di]),x),axis=1)
    X=np.concatenate((X,offset*np.ones([x.shape[0],df])),axis=1)
    return X

def generate_embedded_sierpinski(
    N:int,
    di:int,
    df:int,
    offset:float,
    random_seed:int
):
    X = generate_sierpinski(size=(N, 2), steps = 4, scale = 10, random_seed=random_seed)
    X = embed_data(X, di=di, df=df, offset=offset)# target
    return X
