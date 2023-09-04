import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

def compute_invarianceP4_accurary(model_to_test,x,y,verbose=False):
    if verbose:
        res=[]
        for k in range(0,4):
            res.append(tf.experimental.numpy.rot90(x[0:1],k=k,axes=(1,2)))
        return res
    else:
        res=[]
        for k in range(0,4):
            print('Rot90 times:',k)
            x_rot=tf.experimental.numpy.rot90(x,k=k,axes=(1,2))
            y_rot=model_to_test.predict(x_rot)
            y_rot=np.argmax(y_rot,axis=-1)
            perf=np.sum(y==y_rot)/len(y)
            res.append(perf)
            print('Accurary in',k,perf)
        print('Performance variation on P4:',np.std(res))
        return res


def compute_invariance_translation_accurary(model_to_test,x,y,verbose=False):
    if verbose:
        x_expand=np.pad(x[0:1],((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(0))
        res=[]
        for kx in range(-1,2,1):
            for ky in range(-1,2,1):
                x_rot=tf.roll(x_expand,shift=kx,axis=1)
                x_rot=tf.roll(x_rot,shift=ky,axis=2)
                res.append(x_rot[:,2:-2,2:-2,:])
        return res
    else:
        x_expand=np.pad(x,((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(0))
        res=[]
        for kx in range(-1,2,1):
            for ky in range(-1,2,1):
                print('Translation:',kx,ky)
                x_rot=tf.roll(x_expand,shift=kx,axis=1)
                x_rot=tf.roll(x_rot,shift=ky,axis=2)
                x_rot=x_rot[:,2:-2,2:-2,:]
                y_rot=model_to_test.predict(x_rot)
                y_rot=np.argmax(y_rot,axis=-1)
                perf=np.sum(y==y_rot)/len(y)
                res.append(perf)
                print('Accuracy in',kx,ky,perf)
        print('Performance variation on Translation:',np.std(res))
        return res