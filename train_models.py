
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.resnet50_model import build_resnet50
from models.inceptionv3_model import build_inceptionv3
from models.densenet201_model import build_densenet201

BATCH_SIZE = 32
EPOCHS = 30

train_dir = "dataset/train"
test_dir = "dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("Training ResNet50")
model = build_resnet50()
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

print("Training DenseNet201")
model = build_densenet201()
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

print("Training InceptionV3")
model = build_inceptionv3()
model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)
