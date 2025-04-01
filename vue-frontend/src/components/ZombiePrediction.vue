<template>
  <div
    class="min-h-screen flex items-center justify-center bg-cover bg-center"
    :style="`background-image: url(${require('@/assets/rick_grimes.jpg')})`"
  >
    <div class="container mx-auto p-4 bg-white bg-opacity-85 rounded-xl shadow-lg max-w-xl">
      <h1 class="text-2xl font-bold text-center mb-4">Zombie Data Science - Image Classifier</h1>

      <form @submit.prevent="handleFileUpload">
        <div class="mb-4">
          <label for="file" class="block text-lg">Upload Image</label>
          <input
            type="file"
            id="file"
            class="mt-2"
            ref="fileInput"
            accept="image/*"
            @change="onFileChange"
          />
        </div>

        <!-- Camera controls -->
        <div class="mb-4">
          <button
            type="button"
            class="bg-green-600 text-white px-4 py-2 rounded-lg"
            @click="startCamera"
          >
            Use Camera
          </button>
        </div>

        <div v-if="showCamera">
          <video ref="video" autoplay class="w-full rounded-lg mb-2"></video>
          <button
            type="button"
            class="bg-yellow-500 text-white px-4 py-2 rounded-lg mb-4"
            @click="captureImage"
          >
            Capture
          </button>
          <canvas ref="canvas" class="hidden"></canvas>
        </div>

        <button
          type="submit"
          class="w-full bg-blue-500 text-white py-2 px-4 rounded-lg mt-4"
        >
          Upload and Predict
        </button>
      </form>

      <!-- Displaying the result image -->
      <div v-if="predictedImage" class="mt-6">
        <h2 class="text-xl font-semibold text-center">Predicted Image</h2>
        <img :src="predictedImage" alt="Predicted Zombies" class="mx-auto mt-4 rounded-lg" />
      </div>

      <!-- Display predictions -->
      <div v-if="predictions.length" class="mt-4">
        <h3 class="text-xl font-semibold">Predictions:</h3>
        <ul>
          <li v-for="(prediction, index) in predictions" :key="index">
            <p><strong>Zombie {{ index + 1 }}:</strong> {{ prediction.class }} - Confidence: {{ prediction.confidence }}</p>
            <p><strong>Bounding Box:</strong> {{ prediction.bbox.join(', ') }}</p>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  data() {
    return {
      selectedFile: null,
      predictedImage: "",
      predictions: [],
      showCamera: false,
    };
  },
  methods: {
    onFileChange(event) {
      this.selectedFile = event.target.files[0];
    },

    startCamera() {
      this.showCamera = true;
      const video = this.$refs.video;
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          video.srcObject = stream;
        })
        .catch((err) => {
          console.error("Camera access denied:", err);
          alert("Could not access camera.");
        });
    },

    captureImage() {
      const video = this.$refs.video;
      const canvas = this.$refs.canvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        this.selectedFile = new File([blob], "captured.png", { type: "image/png" });
        // Stop the camera after capture
        video.srcObject.getTracks().forEach((track) => track.stop());
        this.showCamera = false;
      });
    },

    async handleFileUpload() {
      if (!this.selectedFile) {
        alert("Please select or capture a file.");
        return;
      }

      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        const response = await fetch("http://127.0.0.1:8000/predict/", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Failed to get prediction from API");
        }

        const result = await response.json();
        this.predictedImage = result.image;
        this.predictions = result.predictions;
      } catch (error) {
        console.error("Error uploading file:", error);
        alert("Something went wrong. Please try again.");
      }
    },
  },
};
</script>


<style scoped>
.container {
  background-color: rgba(255, 255, 255, 0.85);
  padding: 2rem;
  border-radius: 1rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
</style>
