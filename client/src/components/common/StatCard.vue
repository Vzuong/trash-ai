<template>
  <div class="eco-card h-100 p-3 p-xl-4 stat-card-wrapper position-relative overflow-hidden">
    <div class="d-flex align-items-center justify-content-between mb-2">
      <span class="text-muted small fw-semibold text-uppercase letter-spacing">{{ title }}</span>
      <div 
        class="stat-icon-box d-flex align-items-center justify-content-center"
        :style="{ backgroundColor: iconBg || '#ecfdf5', color: iconColor || '#10b981' }"
      >
        <i :class="['bi', icon || 'bi-pie-chart', 'fs-5']"></i>
      </div>
    </div>

    <div class="d-flex align-items-baseline gap-2 mb-1">
      <h3 class="fw-bold mb-0 text-dark">{{ value }}</h3>
      <span v-if="unit" class="text-muted small">{{ unit }}</span>
    </div>

    <div v-if="subtitle" class="d-flex align-items-center gap-1 text-muted small mt-1">
      <span v-if="trend" :class="['fw-semibold', trendClass]">
        <i :class="['bi', trendIcon]"></i> {{ trend }}
      </span>
      <span>{{ subtitle }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  title: { type: String, required: true },
  value: { type: [Number, String], required: true },
  unit: { type: String, default: '' },
  icon: { type: String, default: 'bi-grid' },
  iconBg: { type: String, default: '' },
  iconColor: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  trend: { type: String, default: '' },
  trendType: { type: String, default: 'up' }
});

const trendClass = computed(() => {
  return props.trendType === 'up' ? 'text-success' : 'text-danger';
});

const trendIcon = computed(() => {
  return props.trendType === 'up' ? 'bi-arrow-up-short' : 'bi-arrow-down-short';
});
</script>

<style scoped>
.letter-spacing {
  letter-spacing: 0.04em;
  font-size: 0.75rem;
}

.stat-icon-box {
  width: 44px;
  height: 44px;
  border-radius: 12px;
}

.stat-card-wrapper:hover .stat-icon-box {
  transform: scale(1.08);
  transition: transform 0.2s ease;
}
</style>
