import tensorflow as tf

def get_prediction(img, model, labels=[
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
  ], target_size=(180, 180)):
  class_names = labels
  img = tf.keras.utils.load_img(img, target_size=target_size)
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = model.predict(img_array)
  score = tf.nn.softmax(prediction)

  return sorted([(class_names[i], 100 * score[0][i].numpy()) for i in range(len(class_names))], key=lambda x: x[1], reverse=True)