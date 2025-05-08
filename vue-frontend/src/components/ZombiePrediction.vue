<template>
  <div
    class="min-h-screen flex items-center justify-center bg-cover bg-center"
    :style="`background-image: url(${require('@/assets/rick_grimes.jpg')})`"
  >
    <div class="bg-black bg-opacity-80 text-white p-6 rounded-2xl shadow-xl w-full max-w-lg">
      <h1 class="text-3xl font-bold text-center mb-6">Zombie Classifier</h1>

      <!-- Tabs -->
      <div class="flex mb-4 rounded-lg overflow-hidden border border-gray-700">
        <button @click="activeTab = 'upload'" :class="tabClass('upload')">
          <i class="fas fa-upload mr-2"></i> Upload
        </button>
        <button @click="activeTab = 'camera'" :class="tabClass('camera')">
          <i class="fas fa-camera mr-2"></i> Camera
        </button>
      </div>

      <!-- Upload Tab -->
      <div v-if="activeTab === 'upload'" class="space-y-4">
        <label class="block text-sm font-medium text-gray-300">Choose an image</label>
        <input
          type="file"
          accept="image/*"
          @change="onFileChange"
          class="w-full bg-gray-800 text-white border border-gray-600 rounded-lg p-2 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-green-600 hover:file:bg-green-700 file:text-white"
        />
      </div>

      <!-- Camera Tab -->
      <div v-if="activeTab === 'camera'" class="space-y-4">
        <video ref="video" autoplay playsinline class="rounded-xl w-full border border-gray-600"></video>
        <button
          @click="captureImage"
          class="bg-green-600 hover:bg-green-700 w-full py-2 px-4 rounded-lg text-white font-semibold"
        >
          Capture Image
        </button>
        <canvas ref="canvas" class="hidden"></canvas>
      </div>

      <!-- Image Preview -->
      <div v-if="imagePreview" class="mt-6">
        <img :src="imagePreview" class="w-full rounded-xl" />
      </div>

        <!-- Result Section -->
      <div v-if="loading" class="text-center mt-4 text-green-400 font-semibold">
        Detecting...
      </div>

      <div v-if="result" class="mt-4 text-center bg-gray-800 rounded-xl p-4">
        <p class="text-lg font-bold text-white">Result:</p>
        <p class="text-xl mt-2">
          <span :class="result.class === 'zombie' ? 'text-red-500' : 'text-green-500'">
            {{ result.class.toUpperCase() }}
          </span>
          <br />
          <span class="text-sm text-gray-400">Confidence: {{ (result.confidence * 100).toFixed(1) }}%</span>
        </p>

        <div v-if="result.image" class="mt-4">
          <img :src="result.image" class="rounded-xl border border-gray-600" />
        </div>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      activeTab: 'upload',
      imagePreview: null,
      loading: false,
      result: null,
    };
  },
  methods: {
    tabClass(tab) {
      return [
        'flex-1 text-center py-2 px-4 transition font-semibold',
        this.activeTab === tab
          ? 'bg-green-600 text-white'
          : 'bg-gray-700 hover:bg-gray-600 text-gray-300',
      ];
    },
    onFileChange(e) {
      const file = e.target.files[0];
      if (file) {
        this.imagePreview = URL.createObjectURL(file);
        this.sendToModel(file);
      }
    },
    async sendToModel(imageFile) {
      this.loading = true;
      this.result = null;

      const formData = new FormData();
      formData.append("file", imageFile);

      try {
        const res = await fetch("https://three25-data-science-image-classifier.onrender.com/predict/", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        if (data.predictions && data.predictions.length > 0) {
          this.result = {
            class: data.predictions[0].class,
            confidence: data.predictions[0].confidence,
            image: data.image,
          };
        } else {
          this.result = {
            class: "unknown",
            confidence: 0,
            image: data.image,
          };
        }
      } catch (err) {
        console.error("Prediction failed:", err);
      } finally {
        this.loading = false;
      }
    },
    async startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        this.$refs.video.srcObject = stream;
      } catch (err) {
        console.error('Camera access denied:', err);
      }
    },
    captureImage() {
      const video = this.$refs.video;
      const canvas = this.$refs.canvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      canvas.toBlob((blob) => {
        this.imagePreview = canvas.toDataURL('image/png');
        if (blob) {
          const file = new File([blob], 'capture.png', { type: 'image/png' });
          this.sendToModel(file);
        }
      });
    },
  },
  watch: {
    activeTab(val) {
      if (val === 'camera') {
        this.startCamera();
      }
    },
  },
};
</script>

<!-- Font Awesome for icons -->
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');
</style>
