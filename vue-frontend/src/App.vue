<template>
  <div class="container mx-auto p-4">
    <ZombiePrediction />
  </div>
</template>

<script>
// Import the ZombiePrediction component
import ZombiePrediction from './components/ZombiePrediction.vue'

export default {
  name: 'App',
  components: {
    ZombiePrediction  // Register the component
  }
}
</script>