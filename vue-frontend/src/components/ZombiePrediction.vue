<template>
  <div class="container mx-auto p-4">
    <h1 class="text-2xl font-bold text-center mb-4">Zombie Data Science - Image Classifier</h1>
    <div class="max-w-xl mx-auto">
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
      selectedFile: "",
      predictedImage: "",
      predictions: [], 
    };
  },
  methods: {
    onFileChange(event) {
      this.selectedFile = event.target.files[0];
    },

    async handleFileUpload() {
      if (!this.selectedFile) {
        alert("Please select a file to upload.");
        return;
      }

      const formData = new FormData();
      formData.append("file", this.selectedFile);

      try {
        // Make the API request to predict
        const response = await fetch("http://127.0.0.1:8000/predict/", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Failed to get prediction from API");
        }

        const result = await response.json();

        // Handle the result (base64 image and predictions)
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
  max-width: 600px;
}
</style>
