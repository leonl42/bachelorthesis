import tensorflow_datasets as tfds
import tensorflow as tf

tfds.display_progress_bar(enable=False)

def get_cifar(name,data_dir="./datasets/",batch_size = 128, shuffle_buffer = 5000, prefetch= tf.data.AUTOTUNE,repeats=-1):
    builder = tfds.builder(name,data_dir=data_dir)
    builder.download_and_prepare()
    ds_train,ds_test = builder.as_dataset(split=["train", "test"])

    solve_dict = lambda elem : (elem["image"],elem["label"])
    ds_train,ds_test = ds_train.map(solve_dict),ds_test.map(solve_dict)

    cast = lambda img,lbl : (tf.cast(img,tf.dtypes.float32),lbl)
    ds_train,ds_test = ds_train.map(cast),ds_test.map(cast)

    mean = tf.convert_to_tensor([0.32768, 0.32768, 0.32768])[None,None,:]
    std = tf.convert_to_tensor([0.27755222, 0.26925606, 0.2683012 ])[None,None,:]

    normalize = lambda img,lbl : ((img/255-mean)/std,lbl)

    ds_train,ds_test = ds_train.map(normalize),ds_test.map(normalize)

    prepare = lambda ds : ds.repeat(repeats).shuffle(buffer_size=shuffle_buffer).batch(batch_size,drop_remainder=True).prefetch(prefetch)
    ds_train,ds_test = prepare(ds_train), prepare(ds_test)

    return ds_train.as_numpy_iterator(),ds_test.as_numpy_iterator()


