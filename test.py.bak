
import numpy as np

from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
print [x.name for x in local_device_protos]

quit()


preds = np.mat([[0, 1, 2, 1, 6, 0]])
themes = ['a', 'b', 'c', 'd', 'e', 'f']



print np.argsort(preds)
print np.argsort(preds).shape
print np.argsort(preds)[::-1][:, range(0, 5)]
print np.fliplr(np.argsort(preds))



for id in  np.argsort(preds)[:, range(0, 5)].tolist()[0]:
    print 'id', id
quit()


m = np.arange(9).reshape(3, 3)

y = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))

top = np.argmax(m, axis=1)

selected = np.argmax(y, axis=1)



print top, selected

corr = np.equal(top, selected)

print corr

print np.where(corr)

print len(np.where(np.equal(np.argmax(m, axis=1), np.argmax(y, axis=1))))
