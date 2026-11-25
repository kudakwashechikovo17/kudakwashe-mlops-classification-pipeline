from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


def sample_image() -> bytes:
    image = Image.new("RGB", (32, 32), color=(150, 90, 70))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


SAMPLE_IMAGE = sample_image()


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def health(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        self.client.post(
            "/predict",
            files={"file": ("sample.png", SAMPLE_IMAGE, "image/png")},
        )
