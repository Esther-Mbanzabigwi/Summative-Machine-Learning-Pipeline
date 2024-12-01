from locust import HttpUser, task, between

class PlantDiseaseRecognitionUser(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def predict_disease(self):
        with open("home_page.jpeg", "rb") as image_file:
            self.client.post(
                "/predict",
                files={"file": ("home_page.jpeg", image_file, "image/jpeg")},
            )