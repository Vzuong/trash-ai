import { createApp } from 'vue';
import App from './App.vue';
import router from './router';

// Bootstrap 5 & Bootstrap Icons CSS & JS
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import 'bootstrap-icons/font/bootstrap-icons.css';

// Main Environmental Design System CSS
import './assets/css/main.css';

const app = createApp(App);
app.use(router);
app.mount('#app');
