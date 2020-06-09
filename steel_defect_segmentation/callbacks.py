from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath=model_file,
                     monitor='val_dice_coef',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)
