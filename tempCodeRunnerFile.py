def create_gender_model():
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
#         MaxPooling2D(),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(2, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model