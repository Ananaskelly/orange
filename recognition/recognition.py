import tensorflow as tf
import cv2
import numpy as np
import recognition.img_processing as ip
import os

sess = tf.Session()
saver = tf.train.import_meta_graph('./model2/trained_model.meta')
saver.restore(sess, './model2/trained_model')


def get_class_no(img):
    inp = ip.load_and_process_image(img, False)
    inp = inp.reshape(1, 32, 32, 3)
    cv2.waitKey()
    prediction = tf.get_default_graph().get_tensor_by_name("softmax:0")
    a = sess.run(prediction, feed_dict={"x:0": inp})

    # test purpose
    # dir = os.listdir(os.path.join(os.getcwd(), 'sign'))
    # ind = len(dir)
    # cv2.imwrite(os.getcwd() + '/sign/' + str(ind) + '_' + str(np.argmax(a)) + '_' + str(np.max(a[0])) + '.jpg', img)

    return np.argmax(a), np.max(a[0])
