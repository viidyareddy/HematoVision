import os
import random
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Disable oneDNN optimization logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Directory structure
dataset_dir = './output_dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Display random images from each class
def show_random_image(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if image_files:
        selected_image = random.choice(image_files)
        display(Image(filename=os.path.join(folder_path, selected_image)))

show_random_image(os.path.join(train_dir, 'Biodegradable Images'))
show_random_image(os.path.join(test_dir, 'Recyclable Images'))
show_random_image(os.path.join(test_dir, 'Trash Images'))
show_random_image(os.path.join(train_dir, 'Trash Images'))

# Build VGG16 model
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False  # Freeze VGG16 layers

x = Flatten()(vgg.output)
output = Dense(3, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=output)

# Compile
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("waste_classifier_vgg16.h5")
print("✅ Model saved as waste_classifier_vgg16.h5")
