# === Program 5(a): L2 Regularization (Ridge Regression) ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

X = np.arange(1, 6).reshape(-1, 1).astype(float)
y = 2 * X.flatten()

# Fit both models
lr  = LinearRegression().fit(X, y)
rid = Ridge(alpha=1.0).fit(X, y)

print("Without L2 -> Weight:", round(lr.coef_[0], 4),  "Bias:", round(lr.intercept_, 4))
print("With L2    -> Weight:", round(rid.coef_[0], 4), "Bias:", round(rid.intercept_, 4))

# Plot
x_range = np.linspace(0, 6, 100).reshape(-1, 1)
plt.plot(x_range, 2 * x_range,             label="Original y=2x",   linestyle='--', color='black')
plt.plot(x_range, lr.predict(x_range),     label="Without L2",      color='blue')
plt.plot(x_range, rid.predict(x_range),    label="With L2 (Ridge)", color='red')
plt.scatter(X, y, color='black', zorder=5)
plt.legend(); plt.title("Linear Regression vs Ridge"); plt.grid(True)
plt.show()


# === Program 5(b): Image Augmentation ===
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

img = load_img("sample.jpg", target_size=(128, 128))
x = img_to_array(img).reshape((1, 128, 128, 3))

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    fill_mode='nearest'
)

plt.figure(figsize=(14, 4))
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 4, i + 1)
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    i += 1
    if i == 8:
        break

plt.suptitle("Augmented Images")
plt.tight_layout(); plt.show()