from locust import HttpUser, task, between


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def health(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        image = b"\xff\xd8\xff\xd9"
        self.client.post("/predict", files={"file": ("tiny.jpg", image, "image/jpeg")})
